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


def _fake_narrative(skills: list[str], insufficient_for: set[str] | None = None) -> _RationaleNarrative:
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
    )


@pytest.fixture
def scenario():
    return _build_scenario()


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
