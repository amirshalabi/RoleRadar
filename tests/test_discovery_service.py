"""
Tests for backend.services.discovery. LLM extraction is always
monkeypatched (no real OpenAI calls); Postgres reads/writes go through
the in-memory FakeSupabaseClient for real read-after-write behavior.
"""

from __future__ import annotations

import pytest

from backend.db import applications as applications_db
from backend.db import candidates as candidates_db
from backend.db import favorites as favorites_db
from backend.db import rationales as rationales_db
from backend.db import role_requirements as role_requirements_db
from backend.db import roles as roles_db
from backend.db import upserts as upserts_db
from backend.llm.extract_requirements import RoleRequirement
from backend.llm.rationale import FitRationale
from backend.services import discovery
from tests._fake_supabase import FakeSupabaseClient

SAMPLE_REQUIREMENTS = [
    RoleRequirement(skill="Probability", normalized_skill="probability", target_level=8.0, importance=9.0, required=True, evidence=["Need probability."]),
    RoleRequirement(skill="Python", normalized_skill="python", target_level=7.0, importance=7.0, required=True, evidence=["Build trading signals."]),
]


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> FakeSupabaseClient:
    client = FakeSupabaseClient()
    for module in (roles_db, candidates_db, favorites_db, applications_db, role_requirements_db, rationales_db, upserts_db):
        monkeypatch.setattr(module, "get_client", lambda c=client: c)
    return client


def _fake_rationale(overall_score: float = 50.0) -> FitRationale:
    return FitRationale(overall_score=overall_score, overall_explanation="x")


def _seed_role(role_family="quant", description="Build trading signals. Need probability and Python."):
    return roles_db.upsert_role(
        company="Meridian Capital", title="Quant Research Intern", location="NYC",
        description=description, role_family=role_family,
    )


# ---------------------------------------------------------------------
# list_role_cards
# ---------------------------------------------------------------------


def test_list_role_cards_empty_when_no_roles(fake_client: FakeSupabaseClient) -> None:
    assert discovery.list_role_cards("u1") == []


def test_unanalyzed_role_has_no_fit_score_or_gaps(fake_client: FakeSupabaseClient) -> None:
    _seed_role()

    [card] = discovery.list_role_cards("u1")

    assert card.analyzed is False
    assert card.fit_score is None
    assert card.top_strengths == []
    assert card.top_gap is None


def test_readiness_computed_live_without_analysis(fake_client: FakeSupabaseClient) -> None:
    _seed_role(role_family="quant")
    candidates_db.upsert_candidate_skill("u1", "probability", estimated_level=5.0, confidence=0.5, display_name="Probability")

    [card] = discovery.list_role_cards("u1")

    assert card.analyzed is False
    assert card.readiness_score is not None  # no LLM/requirements needed for this


def test_readiness_none_without_skills(fake_client: FakeSupabaseClient) -> None:
    _seed_role(role_family="quant")

    [card] = discovery.list_role_cards("u1")

    assert card.readiness_score is None


def test_readiness_none_without_role_family(fake_client: FakeSupabaseClient) -> None:
    _seed_role(role_family=None)
    candidates_db.upsert_candidate_skill("u1", "python", estimated_level=5.0, confidence=0.5, display_name="Python")

    [card] = discovery.list_role_cards("u1")

    assert card.readiness_score is None


def test_card_reflects_favorite_and_application_state(fake_client: FakeSupabaseClient) -> None:
    role_row = _seed_role()
    favorites_db.save_favorite("u1", role_row["id"], priority="dream")
    applications_db.upsert_application("u1", role_row["id"], status="applied", deadline="2026-09-20")

    [card] = discovery.list_role_cards("u1")

    assert card.is_saved is True
    assert card.priority == "dream"
    assert card.application_status == "applied"
    assert card.deadline == "2026-09-20"


def test_card_defaults_when_not_saved_or_tracked(fake_client: FakeSupabaseClient) -> None:
    _seed_role()

    [card] = discovery.list_role_cards("u1")

    assert card.is_saved is False
    assert card.priority is None
    assert card.application_status is None


# ---------------------------------------------------------------------
# analyze_role
# ---------------------------------------------------------------------


def test_analyze_role_extracts_requirements_when_none_persisted(monkeypatch: pytest.MonkeyPatch, fake_client: FakeSupabaseClient) -> None:
    role_row = _seed_role()
    monkeypatch.setattr(discovery, "extract_role_requirements", lambda role, model=None: SAMPLE_REQUIREMENTS)

    analysis = discovery.analyze_role("u1", role_row)

    assert {r.normalized_skill for r in analysis.requirements} == {"probability", "python"}
    assert role_requirements_db.list_role_requirements(role_row["id"])  # persisted


def test_analyze_role_handles_zero_extracted_requirements(monkeypatch: pytest.MonkeyPatch, fake_client: FakeSupabaseClient) -> None:
    """The LLM legitimately finds nothing extractable in a posting - analyze_role must still produce a valid, in-bounds analysis, not crash."""
    role_row = _seed_role()
    monkeypatch.setattr(discovery, "extract_role_requirements", lambda role, model=None: [])

    analysis = discovery.analyze_role("u1", role_row)

    assert analysis.requirements == []
    assert analysis.gaps == []
    assert 0.0 <= analysis.fit_result.overall_score <= 100.0
    assert role_requirements_db.list_role_requirements(role_row["id"]) == []


def test_analyze_role_reuses_persisted_requirements_without_calling_llm(monkeypatch: pytest.MonkeyPatch, fake_client: FakeSupabaseClient) -> None:
    role_row = _seed_role()
    role_requirements_db.upsert_role_requirements(
        role_row["id"],
        [{"normalized_skill_name": "python", "display_name": "Python", "target_level": 6.0, "importance": 7.0, "is_required": True, "evidence": []}],
    )
    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("LLM extraction should not be called when requirements are already persisted")

    monkeypatch.setattr(discovery, "extract_role_requirements", fail_if_called)

    analysis = discovery.analyze_role("u1", role_row)

    assert called is False
    assert len(analysis.requirements) == 1


def test_analyze_role_persists_fit_score(monkeypatch: pytest.MonkeyPatch, fake_client: FakeSupabaseClient) -> None:
    role_row = _seed_role()
    monkeypatch.setattr(discovery, "extract_role_requirements", lambda role, model=None: SAMPLE_REQUIREMENTS)

    analysis = discovery.analyze_role("u1", role_row)

    fit_rows = roles_db.list_fit_scores_for_user("u1")
    assert len(fit_rows) == 1
    assert fit_rows[0]["overall_score"] == pytest.approx(analysis.fit_result.overall_score)


def test_analyze_role_computes_gaps_against_candidate_skills(monkeypatch: pytest.MonkeyPatch, fake_client: FakeSupabaseClient) -> None:
    role_row = _seed_role()
    candidates_db.upsert_candidate_skill("u1", "python", estimated_level=8.0, confidence=0.8, display_name="Python")
    monkeypatch.setattr(discovery, "extract_role_requirements", lambda role, model=None: SAMPLE_REQUIREMENTS)

    analysis = discovery.analyze_role("u1", role_row)

    by_skill = {g.normalized_skill: g for g in analysis.gaps}
    assert by_skill["python"].satisfaction_ratio == 1.0
    assert by_skill["probability"].satisfaction_ratio == 0.0


def test_analyze_role_computes_readiness_when_role_family_present(monkeypatch: pytest.MonkeyPatch, fake_client: FakeSupabaseClient) -> None:
    role_row = _seed_role(role_family="quant")
    monkeypatch.setattr(discovery, "extract_role_requirements", lambda role, model=None: SAMPLE_REQUIREMENTS)

    analysis = discovery.analyze_role("u1", role_row)

    assert analysis.readiness is not None


def test_analyze_role_readiness_none_without_role_family(monkeypatch: pytest.MonkeyPatch, fake_client: FakeSupabaseClient) -> None:
    role_row = _seed_role(role_family=None)
    monkeypatch.setattr(discovery, "extract_role_requirements", lambda role, model=None: SAMPLE_REQUIREMENTS)

    analysis = discovery.analyze_role("u1", role_row)

    assert analysis.readiness is None


def test_analyze_role_raises_without_description_or_cached_requirements(fake_client: FakeSupabaseClient) -> None:
    role_row = _seed_role(description=None)

    with pytest.raises(ValueError):
        discovery.analyze_role("u1", role_row)


def test_analyze_role_raises_with_empty_string_description(fake_client: FakeSupabaseClient) -> None:
    """An empty string is as "no description" as None is - both must hit the same guard, not slip through as truthy."""
    role_row = _seed_role(description="")

    with pytest.raises(ValueError):
        discovery.analyze_role("u1", role_row)


def test_repeated_scoring_does_not_duplicate_fit_score_row(
    monkeypatch: pytest.MonkeyPatch, fake_client: FakeSupabaseClient
) -> None:
    """Calling analyze_role() for the same (user, role) repeatedly must update the one fit_scores row, never insert additional rows."""
    role_row = _seed_role()
    monkeypatch.setattr(discovery, "extract_role_requirements", lambda role, model=None: SAMPLE_REQUIREMENTS)

    discovery.analyze_role("u1", role_row)
    discovery.analyze_role("u1", role_row)
    discovery.analyze_role("u1", role_row)

    fit_rows = roles_db.list_fit_scores_for_user("u1", limit=50)
    assert len(fit_rows) == 1


def test_repeated_scoring_reflects_updated_candidate_skills(
    monkeypatch: pytest.MonkeyPatch, fake_client: FakeSupabaseClient
) -> None:
    """Re-scoring after a skill estimate changes must produce a different score, not a stale cached one."""
    role_row = _seed_role()
    monkeypatch.setattr(discovery, "extract_role_requirements", lambda role, model=None: SAMPLE_REQUIREMENTS)

    candidates_db.upsert_candidate_skill("u1", "python", estimated_level=1.0, confidence=0.2, display_name="Python")
    first = discovery.analyze_role("u1", role_row)

    candidates_db.upsert_candidate_skill("u1", "python", estimated_level=9.0, confidence=0.9, display_name="Python")
    second = discovery.analyze_role("u1", role_row)

    assert second.fit_result.overall_score > first.fit_result.overall_score


def test_card_reflects_analysis_after_it_runs(monkeypatch: pytest.MonkeyPatch, fake_client: FakeSupabaseClient) -> None:
    role_row = _seed_role()
    candidates_db.upsert_candidate_skill("u1", "python", estimated_level=8.0, confidence=0.8, display_name="Python")
    monkeypatch.setattr(discovery, "extract_role_requirements", lambda role, model=None: SAMPLE_REQUIREMENTS)

    discovery.analyze_role("u1", role_row)
    [card] = discovery.list_role_cards("u1")

    assert card.analyzed is True
    assert card.fit_score is not None
    assert "Python" in card.top_strengths
    assert card.top_gap == "Probability"


# ---------------------------------------------------------------------
# get_or_generate_rationale
# ---------------------------------------------------------------------


def test_get_or_generate_rationale_calls_llm_and_persists_on_first_run(
    monkeypatch: pytest.MonkeyPatch, fake_client: FakeSupabaseClient
) -> None:
    role_row = _seed_role()
    monkeypatch.setattr(discovery, "extract_role_requirements", lambda role, model=None: SAMPLE_REQUIREMENTS)
    analysis = discovery.analyze_role("u1", role_row)

    calls = []
    monkeypatch.setattr(
        discovery,
        "generate_fit_rationale",
        lambda *a, **kw: (calls.append(1), _fake_rationale(analysis.fit_result.overall_score))[1],
    )

    rationale = discovery.get_or_generate_rationale("u1", role_row, analysis)

    assert len(calls) == 1
    assert rationale.overall_score == analysis.fit_result.overall_score
    assert rationales_db.get_rationale("u1", role_row["id"]) is not None


def test_get_or_generate_rationale_reuses_cache_when_inputs_unchanged(
    monkeypatch: pytest.MonkeyPatch, fake_client: FakeSupabaseClient
) -> None:
    role_row = _seed_role()
    monkeypatch.setattr(discovery, "extract_role_requirements", lambda role, model=None: SAMPLE_REQUIREMENTS)
    analysis = discovery.analyze_role("u1", role_row)

    calls = []
    monkeypatch.setattr(
        discovery, "generate_fit_rationale", lambda *a, **kw: (calls.append(1), _fake_rationale())[1]
    )

    first = discovery.get_or_generate_rationale("u1", role_row, analysis)
    second = discovery.get_or_generate_rationale("u1", role_row, analysis)

    assert len(calls) == 1  # the second call was a pure cache read - no LLM/Qdrant call
    assert first.overall_score == second.overall_score


def test_get_or_generate_rationale_regenerates_when_candidate_skills_change(
    monkeypatch: pytest.MonkeyPatch, fake_client: FakeSupabaseClient
) -> None:
    role_row = _seed_role()
    monkeypatch.setattr(discovery, "extract_role_requirements", lambda role, model=None: SAMPLE_REQUIREMENTS)
    analysis = discovery.analyze_role("u1", role_row)

    calls = []
    monkeypatch.setattr(
        discovery, "generate_fit_rationale", lambda *a, **kw: (calls.append(1), _fake_rationale())[1]
    )

    discovery.get_or_generate_rationale("u1", role_row, analysis)
    candidates_db.upsert_candidate_skill("u1", "python", estimated_level=9.0, confidence=0.9, display_name="Python")
    discovery.get_or_generate_rationale("u1", role_row, analysis)

    assert len(calls) == 2  # candidate skills changed since the cached rationale was generated


def test_get_or_generate_rationale_regenerates_when_requirements_change(
    monkeypatch: pytest.MonkeyPatch, fake_client: FakeSupabaseClient
) -> None:
    role_row = _seed_role()
    monkeypatch.setattr(discovery, "extract_role_requirements", lambda role, model=None: SAMPLE_REQUIREMENTS)
    analysis = discovery.analyze_role("u1", role_row)

    calls = []
    monkeypatch.setattr(
        discovery, "generate_fit_rationale", lambda *a, **kw: (calls.append(1), _fake_rationale())[1]
    )
    discovery.get_or_generate_rationale("u1", role_row, analysis)

    role_requirements_db.upsert_role_requirements(
        role_row["id"],
        [{"normalized_skill_name": "python", "display_name": "Python", "target_level": 9.0, "importance": 9.0, "is_required": True, "evidence": []}],
    )
    changed_analysis = discovery.analyze_role("u1", role_row)
    discovery.get_or_generate_rationale("u1", role_row, changed_analysis)

    assert len(calls) == 2  # requirement target_level changed since the cached rationale was generated


# ---------------------------------------------------------------------
# build_favorite_contexts / compare_roles / get_skill_roi_for_favorites
# ---------------------------------------------------------------------


def _seed_favorite(title, company, role_family, requirements, priority="interested"):
    role_row = roles_db.upsert_role(company=company, title=title, role_family=role_family, description="desc")
    favorites_db.save_favorite("u1", role_row["id"], priority=priority)
    if requirements:
        role_requirements_db.upsert_role_requirements(
            role_row["id"],
            [
                {
                    "normalized_skill_name": r.normalized_skill, "display_name": r.skill,
                    "target_level": r.target_level, "importance": r.importance,
                    "is_required": r.required, "evidence": r.evidence,
                }
                for r in requirements
            ],
        )
    return role_row


def test_build_favorite_contexts_reuses_persisted_requirements(fake_client: FakeSupabaseClient) -> None:
    role_row = _seed_favorite(
        "Quant Intern", "Meridian", "quant", [SAMPLE_REQUIREMENTS[0]], priority="dream"
    )

    [context] = discovery.build_favorite_contexts("u1", [role_row["id"]])

    assert context.priority == "dream"
    assert context.requirements[0].normalized_skill == "probability"


def test_build_favorite_contexts_empty_requirements_for_unanalyzed_role(fake_client: FakeSupabaseClient) -> None:
    role_row = _seed_favorite("SWE Intern", "Acme", "swe", [])

    [context] = discovery.build_favorite_contexts("u1", [role_row["id"]])

    assert context.requirements == []


def test_build_favorite_contexts_carries_role_id(fake_client: FakeSupabaseClient) -> None:
    role_row = _seed_favorite("Quant Intern", "Meridian", "quant", [])

    [context] = discovery.build_favorite_contexts("u1", [role_row["id"]])

    assert context.role_id == role_row["id"]


def test_compare_roles_returns_comparisons_and_summary(fake_client: FakeSupabaseClient) -> None:
    role_a = _seed_favorite("Quant Intern", "Meridian", "quant", SAMPLE_REQUIREMENTS, priority="dream")
    role_b = _seed_favorite(
        "SWE Intern", "Acme", "swe",
        [RoleRequirement(skill="Python", normalized_skill="python", target_level=9.0, importance=9.0, required=True, evidence=["x"])],
        priority="backup",
    )
    candidates_db.upsert_candidate_skill("u1", "python", estimated_level=8.0, confidence=0.8, display_name="Python")

    comparisons, summary = discovery.compare_roles("u1", [role_a["id"], role_b["id"]])

    assert len(comparisons) == 2
    assert summary.best_current_match is not None
    assert summary.largest_prep_burden is not None


def test_get_skill_roi_for_favorites_covers_all_saved_roles(fake_client: FakeSupabaseClient) -> None:
    _seed_favorite("Quant Intern", "Meridian", "quant", SAMPLE_REQUIREMENTS, priority="dream")
    _seed_favorite(
        "SWE Intern", "Acme", "swe",
        [RoleRequirement(skill="Python", normalized_skill="python", target_level=9.0, importance=9.0, required=True, evidence=["x"])],
        priority="backup",
    )

    results = discovery.get_skill_roi_for_favorites("u1")

    by_skill = {r.normalized_skill: r for r in results}
    assert by_skill["python"].roles_requiring_it == 2
    assert by_skill["probability"].roles_requiring_it == 1


def test_get_skill_roi_for_favorites_empty_when_no_favorites(fake_client: FakeSupabaseClient) -> None:
    assert discovery.get_skill_roi_for_favorites("u1") == []


# ---------------------------------------------------------------------
# build_study_plan_for_application
# ---------------------------------------------------------------------


def test_build_study_plan_raises_without_interview_date(monkeypatch: pytest.MonkeyPatch, fake_client: FakeSupabaseClient) -> None:
    role_row = _seed_role()
    monkeypatch.setattr(discovery, "extract_role_requirements", lambda role, model=None: SAMPLE_REQUIREMENTS)
    applications_db.upsert_application("u1", role_row["id"], status="applied")

    with pytest.raises(ValueError):
        discovery.build_study_plan_for_application("u1", role_row["id"])


def test_build_study_plan_raises_without_any_application(fake_client: FakeSupabaseClient) -> None:
    role_row = _seed_role()

    with pytest.raises(ValueError):
        discovery.build_study_plan_for_application("u1", role_row["id"])


def test_build_study_plan_uses_interview_date_and_default_hours(
    monkeypatch: pytest.MonkeyPatch, fake_client: FakeSupabaseClient
) -> None:
    role_row = _seed_role(role_family="quant")
    monkeypatch.setattr(discovery, "extract_role_requirements", lambda role, model=None: SAMPLE_REQUIREMENTS)
    applications_db.upsert_application("u1", role_row["id"], status="interview", interview_date="2026-09-20")

    plan = discovery.build_study_plan_for_application("u1", role_row["id"])

    assert plan.interview_date.isoformat() == "2026-09-20"
    assert plan.hours_available_per_day == discovery.DEFAULT_PREP_HOURS_PER_DAY


def test_build_study_plan_uses_applications_own_hours_available(
    monkeypatch: pytest.MonkeyPatch, fake_client: FakeSupabaseClient
) -> None:
    role_row = _seed_role(role_family="quant")
    monkeypatch.setattr(discovery, "extract_role_requirements", lambda role, model=None: SAMPLE_REQUIREMENTS)
    applications_db.upsert_application(
        "u1", role_row["id"], status="interview", interview_date="2026-09-20", hours_available_per_day=5.0
    )

    plan = discovery.build_study_plan_for_application("u1", role_row["id"])

    assert plan.hours_available_per_day == 5.0


def test_build_study_plan_applies_favorite_priority_multiplier(
    monkeypatch: pytest.MonkeyPatch, fake_client: FakeSupabaseClient
) -> None:
    role_a = _seed_role(role_family="quant")
    role_b = roles_db.upsert_role(company="Other Co", title="Other Intern", role_family="quant", description="Build trading signals. Need probability and Python.")
    monkeypatch.setattr(discovery, "extract_role_requirements", lambda role, model=None: SAMPLE_REQUIREMENTS)

    applications_db.upsert_application("u1", role_a["id"], status="interview", interview_date="2026-09-20")
    applications_db.upsert_application("u1", role_b["id"], status="interview", interview_date="2026-09-20")
    favorites_db.save_favorite("u1", role_a["id"], priority="dream")
    favorites_db.save_favorite("u1", role_b["id"], priority="backup")

    plan_a = discovery.build_study_plan_for_application("u1", role_a["id"])
    plan_b = discovery.build_study_plan_for_application("u1", role_b["id"])

    priority_a = [item.priority_score for item in plan_a.prep_items]
    priority_b = [item.priority_score for item in plan_b.prep_items]
    assert sum(priority_a) > sum(priority_b)  # dream priority multiplier > backup


# ---------------------------------------------------------------------
# get_requirement_evidence_for_skill
# ---------------------------------------------------------------------


def test_get_requirement_evidence_for_skill_returns_persisted_quotes(fake_client: FakeSupabaseClient) -> None:
    role_row = _seed_role()
    role_requirements_db.upsert_role_requirements(
        role_row["id"],
        [{"normalized_skill_name": "python", "display_name": "Python", "target_level": 7.0, "importance": 7.0, "is_required": True, "evidence": ["Proficiency in Python required."]}],
    )

    evidence = discovery.get_requirement_evidence_for_skill(role_row["id"], "python")

    assert evidence == ["Proficiency in Python required."]


def test_get_requirement_evidence_for_skill_empty_when_skill_not_required(fake_client: FakeSupabaseClient) -> None:
    role_row = _seed_role()
    role_requirements_db.upsert_role_requirements(
        role_row["id"],
        [{"normalized_skill_name": "python", "display_name": "Python", "target_level": 7.0, "importance": 7.0, "is_required": True, "evidence": ["x"]}],
    )

    assert discovery.get_requirement_evidence_for_skill(role_row["id"], "probability") == []


def test_get_requirement_evidence_for_skill_empty_when_role_never_analyzed(fake_client: FakeSupabaseClient) -> None:
    role_row = _seed_role()
    assert discovery.get_requirement_evidence_for_skill(role_row["id"], "python") == []
