"""
Tests for backend.llm.rationale. No real OpenAI or Qdrant calls happen:
retrieve_skill_evidence / retrieve_candidate_evidence and parse_structured
are all monkeypatched. The fit score and skill gaps themselves are
computed for real via backend.matching.scorer / backend.matching.gaps
(both pure Python, no mocking needed) so these tests exercise a
realistic, internally-consistent FitScoreResult.
"""

from __future__ import annotations

import pytest

from backend.candidate.profile import CandidateProfile, CandidateSkillEstimate
from backend.ingestion.normalize import normalize_role
from backend.llm import rationale as rationale_module
from backend.llm.extract_requirements import RoleRequirement
from backend.llm.rationale import (
    FitRationale,
    _LLMDimensionNarrative,
    _LLMSkillNarrative,
    _RationaleNarrative,
    generate_fit_rationale,
)
from backend.matching.scorer import calculate_fit_score
from backend.rag.retrieval import SkillEvidenceBundle
from backend.rag.vector_store import RetrievedChunk


def _build_scenario():
    """A candidate who meets 'python' but is missing the required 'probability' skill."""
    profile = CandidateProfile(
        coursework=["Algorithms"],
        programming_languages=["Python"],
        domain_experience=["trading systems"],
        skills=[
            CandidateSkillEstimate(
                normalized_skill_name="python",
                display_name="Python",
                estimated_level=8.0,
                confidence=0.8,
                evidence_snippets=["Built Python ETL pipelines"],
            )
        ],
    )
    role = normalize_role({"title": "Quant Intern", "company": "Meridian Capital", "role_family": "quant"})
    requirements = [
        RoleRequirement(
            skill="Python", normalized_skill="python", target_level=7.0, importance=8.0, required=True,
            evidence=["Proficiency in Python"],
        ),
        RoleRequirement(
            skill="Probability", normalized_skill="probability", target_level=6.0, importance=9.0, required=True,
            evidence=["Strong foundation in probability"],
        ),
    ]
    fit_score = calculate_fit_score(profile, role, requirements)
    skill_gaps = fit_score.technical_detail.gaps
    return profile, role, requirements, fit_score, skill_gaps


def _chunk(text: str, source_type: str = "skill", skill: str | None = None, score: float = 0.9) -> RetrievedChunk:
    payload = {"text": text, "source_type": source_type, "skill": skill}
    return RetrievedChunk(id="id", score=score, payload=payload)


def _fake_narrative(
    skills: list[str],
    insufficient_for: set[str] | None = None,
    dimensions: list[str] | None = None,
) -> _RationaleNarrative:
    insufficient_for = insufficient_for or set()
    return _RationaleNarrative(
        overall_explanation="Solid technical alignment with one notable gap.",
        strengths=["Strong Python evidence from a real ETL project."],
        weaknesses=["No demonstrated probability experience."],
        risks=["May struggle with probability-heavy interview questions."],
        uncertain_areas=["Domain fit is only weakly evidenced."],
        recommended_actions=["Review core probability concepts before interviewing."],
        skill_narratives=[
            _LLMSkillNarrative(
                skill=skill,
                assessment="Strong alignment." if skill not in insufficient_for else "Cannot assess.",
                risk="Low." if skill not in insufficient_for else "Unknown.",
                insufficient_evidence=skill in insufficient_for,
            )
            for skill in skills
        ],
        dimension_narratives=[
            _LLMDimensionNarrative(
                dimension=dimension,
                rationale=f"{dimension} rationale text.",
                strengths=[f"{dimension} strength."],
                weaknesses=[f"{dimension} weakness."],
                risks=[f"{dimension} risk."],
                recommended_action=f"{dimension} action.",
            )
            for dimension in (dimensions or [])
        ],
        why_this_role="This role plays to the candidate's strongest evidenced skill.",
        why_not_this_role="The candidate has no evidenced experience with the role's core knowledge area.",
    )


@pytest.fixture
def scenario():
    return _build_scenario()


@pytest.fixture(autouse=True)
def _no_real_role_content_retrieval(monkeypatch: pytest.MonkeyPatch):
    """
    Every test below drives generate_fit_rationale(), which now also
    retrieves role-side evidence per fit-score dimension
    (retrieve_role_content) in addition to the candidate-side calls
    tests already patch. Default it to "nothing retrieved" here so
    existing tests that don't care about dimension evidence don't need
    to add their own patch (real Qdrant/embeddings aren't configured in
    tests); tests that DO care override it explicitly.
    """
    monkeypatch.setattr(rationale_module, "retrieve_role_content", lambda *a, **kw: [])


def test_overall_score_is_preserved_exactly(monkeypatch: pytest.MonkeyPatch, scenario) -> None:
    profile, role, requirements, fit_score, skill_gaps = scenario

    monkeypatch.setattr(
        rationale_module,
        "retrieve_skill_evidence",
        lambda user_id, role_id, skill, top_k=3, provider=None: SkillEvidenceBundle(skill=skill),
    )
    monkeypatch.setattr(
        rationale_module, "retrieve_candidate_evidence", lambda *a, **kw: []
    )
    monkeypatch.setattr(
        rationale_module,
        "parse_structured",
        lambda **kwargs: _fake_narrative(["python", "probability"]),
    )

    result = generate_fit_rationale(profile, role, requirements, fit_score, skill_gaps, "u1", "r1")

    assert result.overall_score == fit_score.overall_score
    assert isinstance(result, FitRationale)


def test_skill_score_contribution_and_confidence_come_from_gaps_not_llm(
    monkeypatch: pytest.MonkeyPatch, scenario
) -> None:
    profile, role, requirements, fit_score, skill_gaps = scenario

    monkeypatch.setattr(
        rationale_module,
        "retrieve_skill_evidence",
        lambda user_id, role_id, skill, top_k=3, provider=None: SkillEvidenceBundle(
            skill=skill,
            candidate_evidence=[_chunk("some candidate text", skill=skill)],
            role_evidence=[_chunk("some role text", source_type="requirement", skill=skill)],
        ),
    )
    monkeypatch.setattr(rationale_module, "retrieve_candidate_evidence", lambda *a, **kw: [])
    monkeypatch.setattr(
        rationale_module,
        "parse_structured",
        # LLM tries to claim a different confidence/score - it has no such fields, so it can't;
        # this also proves the schema itself makes tampering impossible.
        lambda **kwargs: _fake_narrative(["python", "probability"]),
    )

    result = generate_fit_rationale(profile, role, requirements, fit_score, skill_gaps, "u1", "r1")

    by_skill = {sr.skill: sr for sr in result.skill_rationales}
    for gap in skill_gaps:
        assert by_skill[gap.display_skill].score_contribution == gap.satisfaction_ratio
        assert by_skill[gap.display_skill].confidence == gap.confidence


def test_retrieval_happens_before_generation(monkeypatch: pytest.MonkeyPatch, scenario) -> None:
    profile, role, requirements, fit_score, skill_gaps = scenario
    call_order: list[str] = []

    def fake_retrieve_skill_evidence(user_id, role_id, skill, top_k=3, provider=None):
        call_order.append(f"retrieve_skill_evidence:{skill}")
        return SkillEvidenceBundle(skill=skill)

    def fake_retrieve_candidate_evidence(*args, **kwargs):
        call_order.append("retrieve_candidate_evidence")
        return []

    def fake_parse_structured(**kwargs):
        call_order.append("parse_structured")
        return _fake_narrative(["python", "probability"])

    monkeypatch.setattr(rationale_module, "retrieve_skill_evidence", fake_retrieve_skill_evidence)
    monkeypatch.setattr(rationale_module, "retrieve_candidate_evidence", fake_retrieve_candidate_evidence)
    monkeypatch.setattr(rationale_module, "parse_structured", fake_parse_structured)

    generate_fit_rationale(profile, role, requirements, fit_score, skill_gaps, "u1", "r1")

    generation_index = call_order.index("parse_structured")
    retrieval_indices = [i for i, call in enumerate(call_order) if call != "parse_structured"]
    assert retrieval_indices, "expected at least one retrieval call"
    assert max(retrieval_indices) < generation_index


def test_missing_requirements_is_computed_from_gaps_not_llm(monkeypatch: pytest.MonkeyPatch, scenario) -> None:
    profile, role, requirements, fit_score, skill_gaps = scenario

    monkeypatch.setattr(
        rationale_module,
        "retrieve_skill_evidence",
        lambda user_id, role_id, skill, top_k=3, provider=None: SkillEvidenceBundle(skill=skill),
    )
    monkeypatch.setattr(rationale_module, "retrieve_candidate_evidence", lambda *a, **kw: [])
    monkeypatch.setattr(
        rationale_module, "parse_structured", lambda **kwargs: _fake_narrative(["python", "probability"])
    )

    result = generate_fit_rationale(profile, role, requirements, fit_score, skill_gaps, "u1", "r1")

    assert len(result.missing_requirements) == 1
    assert "Probability" in result.missing_requirements[0]
    assert "Python" not in " ".join(result.missing_requirements)


def test_insufficient_evidence_forced_true_when_nothing_retrieved_even_if_llm_disagrees(
    monkeypatch: pytest.MonkeyPatch, scenario
) -> None:
    profile, role, requirements, fit_score, skill_gaps = scenario

    monkeypatch.setattr(
        rationale_module,
        "retrieve_skill_evidence",
        lambda user_id, role_id, skill, top_k=3, provider=None: SkillEvidenceBundle(skill=skill),  # no evidence
    )
    monkeypatch.setattr(rationale_module, "retrieve_candidate_evidence", lambda *a, **kw: [])
    # LLM (incorrectly) claims it CAN assess despite zero retrieved evidence.
    monkeypatch.setattr(
        rationale_module,
        "parse_structured",
        lambda **kwargs: _fake_narrative(["python", "probability"], insufficient_for=set()),
    )

    result = generate_fit_rationale(profile, role, requirements, fit_score, skill_gaps, "u1", "r1")

    for skill_rationale in result.skill_rationales:
        assert skill_rationale.insufficient_evidence is True


def test_evidence_present_allows_llm_reported_sufficiency_through(monkeypatch: pytest.MonkeyPatch, scenario) -> None:
    profile, role, requirements, fit_score, skill_gaps = scenario

    monkeypatch.setattr(
        rationale_module,
        "retrieve_skill_evidence",
        lambda user_id, role_id, skill, top_k=3, provider=None: SkillEvidenceBundle(
            skill=skill,
            candidate_evidence=[_chunk("evidence", skill=skill)],
            role_evidence=[_chunk("requirement evidence", source_type="requirement", skill=skill)],
        ),
    )
    monkeypatch.setattr(rationale_module, "retrieve_candidate_evidence", lambda *a, **kw: [])
    monkeypatch.setattr(
        rationale_module, "parse_structured", lambda **kwargs: _fake_narrative(["python", "probability"])
    )

    result = generate_fit_rationale(profile, role, requirements, fit_score, skill_gaps, "u1", "r1")

    for skill_rationale in result.skill_rationales:
        assert skill_rationale.insufficient_evidence is False


def test_evidence_references_are_deduplicated(monkeypatch: pytest.MonkeyPatch, scenario) -> None:
    profile, role, requirements, fit_score, skill_gaps = scenario
    duplicate_chunk = _chunk("duplicated evidence text", source_type="skill", skill="python")

    monkeypatch.setattr(
        rationale_module,
        "retrieve_skill_evidence",
        lambda user_id, role_id, skill, top_k=3, provider=None: SkillEvidenceBundle(
            skill=skill, candidate_evidence=[duplicate_chunk], role_evidence=[]
        ),
    )
    # Every dimension query happens to return the exact same chunk too.
    monkeypatch.setattr(rationale_module, "retrieve_candidate_evidence", lambda *a, **kw: [duplicate_chunk])
    monkeypatch.setattr(
        rationale_module, "parse_structured", lambda **kwargs: _fake_narrative(["python", "probability"])
    )

    result = generate_fit_rationale(profile, role, requirements, fit_score, skill_gaps, "u1", "r1")

    matching = [r for r in result.evidence_references if r.text == "duplicated evidence text"]
    assert len(matching) == 1


def test_skill_rationale_matched_by_normalized_skill_identifier(monkeypatch: pytest.MonkeyPatch, scenario) -> None:
    profile, role, requirements, fit_score, skill_gaps = scenario

    monkeypatch.setattr(
        rationale_module,
        "retrieve_skill_evidence",
        lambda user_id, role_id, skill, top_k=3, provider=None: SkillEvidenceBundle(
            skill=skill, candidate_evidence=[_chunk("evidence", skill=skill)]
        ),
    )
    monkeypatch.setattr(rationale_module, "retrieve_candidate_evidence", lambda *a, **kw: [])
    monkeypatch.setattr(
        rationale_module,
        "parse_structured",
        # Only provide a narrative for "python" - "probability" should fall back gracefully.
        lambda **kwargs: _RationaleNarrative(
            overall_explanation="x",
            skill_narratives=[_LLMSkillNarrative(skill="python", assessment="Great fit.", risk="Low.")],
        ),
    )

    result = generate_fit_rationale(profile, role, requirements, fit_score, skill_gaps, "u1", "r1")

    by_skill = {sr.skill: sr for sr in result.skill_rationales}
    assert by_skill["Python"].assessment == "Great fit."
    assert by_skill["Probability"].assessment == "Insufficient evidence to assess this skill."
    assert by_skill["Probability"].insufficient_evidence is True


# ---------------------------------------------------------------------
# Per-dimension rationale (FIT BREAKDOWN)
# ---------------------------------------------------------------------

_ALL_DIMENSIONS = ["technical", "experience", "coursework", "domain", "interest", "constraints"]


def test_dimension_rationales_cover_all_six_dimensions(monkeypatch: pytest.MonkeyPatch, scenario) -> None:
    profile, role, requirements, fit_score, skill_gaps = scenario
    monkeypatch.setattr(
        rationale_module,
        "retrieve_skill_evidence",
        lambda user_id, role_id, skill, top_k=3, provider=None: SkillEvidenceBundle(skill=skill),
    )
    monkeypatch.setattr(rationale_module, "retrieve_candidate_evidence", lambda *a, **kw: [])
    monkeypatch.setattr(
        rationale_module,
        "parse_structured",
        lambda **kwargs: _fake_narrative(["python", "probability"], dimensions=_ALL_DIMENSIONS),
    )

    result = generate_fit_rationale(profile, role, requirements, fit_score, skill_gaps, "u1", "r1")

    assert {d.dimension for d in result.dimension_rationales} == set(_ALL_DIMENSIONS)


def test_dimension_score_copied_from_fit_score_components_not_llm(monkeypatch: pytest.MonkeyPatch, scenario) -> None:
    profile, role, requirements, fit_score, skill_gaps = scenario
    monkeypatch.setattr(
        rationale_module,
        "retrieve_skill_evidence",
        lambda user_id, role_id, skill, top_k=3, provider=None: SkillEvidenceBundle(skill=skill),
    )
    monkeypatch.setattr(rationale_module, "retrieve_candidate_evidence", lambda *a, **kw: [])
    monkeypatch.setattr(
        rationale_module,
        "parse_structured",
        lambda **kwargs: _fake_narrative(["python", "probability"], dimensions=_ALL_DIMENSIONS),
    )

    result = generate_fit_rationale(profile, role, requirements, fit_score, skill_gaps, "u1", "r1")

    by_dimension = {d.dimension: d for d in result.dimension_rationales}
    for dimension in _ALL_DIMENSIONS:
        assert by_dimension[dimension].score == getattr(fit_score.components, dimension)


def test_technical_dimension_confidence_uses_aggregate_confidence(monkeypatch: pytest.MonkeyPatch, scenario) -> None:
    profile, role, requirements, fit_score, skill_gaps = scenario
    monkeypatch.setattr(
        rationale_module,
        "retrieve_skill_evidence",
        lambda user_id, role_id, skill, top_k=3, provider=None: SkillEvidenceBundle(skill=skill),
    )
    monkeypatch.setattr(rationale_module, "retrieve_candidate_evidence", lambda *a, **kw: [])
    monkeypatch.setattr(
        rationale_module,
        "parse_structured",
        lambda **kwargs: _fake_narrative(["python", "probability"], dimensions=_ALL_DIMENSIONS),
    )

    result = generate_fit_rationale(profile, role, requirements, fit_score, skill_gaps, "u1", "r1")

    technical = next(d for d in result.dimension_rationales if d.dimension == "technical")
    assert technical.confidence == fit_score.technical_detail.aggregate_confidence


def test_non_technical_dimension_confidence_derived_from_evidence_scores(
    monkeypatch: pytest.MonkeyPatch, scenario
) -> None:
    profile, role, requirements, fit_score, skill_gaps = scenario
    monkeypatch.setattr(
        rationale_module,
        "retrieve_skill_evidence",
        lambda user_id, role_id, skill, top_k=3, provider=None: SkillEvidenceBundle(skill=skill),
    )
    # Every non-technical dimension gets one candidate-side chunk at score 0.6.
    monkeypatch.setattr(
        rationale_module, "retrieve_candidate_evidence", lambda *a, **kw: [_chunk("evidence", score=0.6)]
    )
    monkeypatch.setattr(
        rationale_module,
        "parse_structured",
        lambda **kwargs: _fake_narrative(["python", "probability"], dimensions=_ALL_DIMENSIONS),
    )

    result = generate_fit_rationale(profile, role, requirements, fit_score, skill_gaps, "u1", "r1")

    by_dimension = {d.dimension: d for d in result.dimension_rationales}
    assert by_dimension["experience"].confidence == pytest.approx(0.6)
    assert by_dimension["domain"].confidence == pytest.approx(0.6)


def test_dimension_with_no_llm_entry_falls_back_gracefully(monkeypatch: pytest.MonkeyPatch, scenario) -> None:
    profile, role, requirements, fit_score, skill_gaps = scenario
    monkeypatch.setattr(
        rationale_module,
        "retrieve_skill_evidence",
        lambda user_id, role_id, skill, top_k=3, provider=None: SkillEvidenceBundle(skill=skill),
    )
    monkeypatch.setattr(rationale_module, "retrieve_candidate_evidence", lambda *a, **kw: [])
    monkeypatch.setattr(
        rationale_module,
        "parse_structured",
        # Only "technical" gets a narrative entry - every other dimension should fall back.
        lambda **kwargs: _fake_narrative(["python", "probability"], dimensions=["technical"]),
    )

    result = generate_fit_rationale(profile, role, requirements, fit_score, skill_gaps, "u1", "r1")

    by_dimension = {d.dimension: d for d in result.dimension_rationales}
    assert by_dimension["technical"].rationale == "technical rationale text."
    assert by_dimension["domain"].rationale == "Insufficient evidence to assess this dimension."
    assert by_dimension["domain"].insufficient_evidence is True


# ---------------------------------------------------------------------
# why_this_role / why_not_this_role / biggest_risk / highest_impact_action
# ---------------------------------------------------------------------


def test_why_this_role_and_why_not_are_passed_through(monkeypatch: pytest.MonkeyPatch, scenario) -> None:
    profile, role, requirements, fit_score, skill_gaps = scenario
    monkeypatch.setattr(
        rationale_module,
        "retrieve_skill_evidence",
        lambda user_id, role_id, skill, top_k=3, provider=None: SkillEvidenceBundle(skill=skill),
    )
    monkeypatch.setattr(rationale_module, "retrieve_candidate_evidence", lambda *a, **kw: [])
    monkeypatch.setattr(
        rationale_module, "parse_structured", lambda **kwargs: _fake_narrative(["python", "probability"])
    )

    result = generate_fit_rationale(profile, role, requirements, fit_score, skill_gaps, "u1", "r1")

    assert result.why_this_role == "This role plays to the candidate's strongest evidenced skill."
    assert result.why_not_this_role == "The candidate has no evidenced experience with the role's core knowledge area."


def test_biggest_risk_and_highest_impact_action_take_first_llm_entry(
    monkeypatch: pytest.MonkeyPatch, scenario
) -> None:
    profile, role, requirements, fit_score, skill_gaps = scenario
    monkeypatch.setattr(
        rationale_module,
        "retrieve_skill_evidence",
        lambda user_id, role_id, skill, top_k=3, provider=None: SkillEvidenceBundle(skill=skill),
    )
    monkeypatch.setattr(rationale_module, "retrieve_candidate_evidence", lambda *a, **kw: [])
    monkeypatch.setattr(
        rationale_module, "parse_structured", lambda **kwargs: _fake_narrative(["python", "probability"])
    )

    result = generate_fit_rationale(profile, role, requirements, fit_score, skill_gaps, "u1", "r1")

    assert result.biggest_risk == "May struggle with probability-heavy interview questions."
    assert result.highest_impact_action == "Review core probability concepts before interviewing."


def test_biggest_risk_and_highest_impact_action_fall_back_when_llm_gives_none(
    monkeypatch: pytest.MonkeyPatch, scenario
) -> None:
    profile, role, requirements, fit_score, skill_gaps = scenario
    monkeypatch.setattr(
        rationale_module,
        "retrieve_skill_evidence",
        lambda user_id, role_id, skill, top_k=3, provider=None: SkillEvidenceBundle(skill=skill),
    )
    monkeypatch.setattr(rationale_module, "retrieve_candidate_evidence", lambda *a, **kw: [])
    monkeypatch.setattr(
        rationale_module,
        "parse_structured",
        lambda **kwargs: _RationaleNarrative(overall_explanation="x"),
    )

    result = generate_fit_rationale(profile, role, requirements, fit_score, skill_gaps, "u1", "r1")

    assert result.biggest_risk == "No significant risk identified from the available evidence."
    assert result.highest_impact_action == "Gather more evidence (resume detail, coursework, projects) before acting."
