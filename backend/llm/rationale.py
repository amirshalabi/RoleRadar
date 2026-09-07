"""
Evidence-backed rationale generation.

generate_fit_rationale() explains a fit score that Python has ALREADY
computed (backend.matching.scorer.calculate_fit_score() /
backend.matching.gaps.calculate_skill_gaps()) - it never computes,
replaces, or adjusts that score. Every numeric value in the returned
FitRationale (overall_score, and each skill's score_contribution and
confidence) is copied directly from the FitScoreResult/SkillGapResult
passed in; the LLM only ever produces narrative text
(_RationaleNarrative below), which is spliced together with those
already-final numbers during assembly, well after the LLM call returns.

Order of operations is retrieval-then-generation, enforced structurally:
_retrieve_evidence() (all Qdrant calls) runs first and its result is
handed to _generate_narrative() (the only LLM call) - there is no path
through generate_fit_rationale() that calls the LLM before retrieval.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from backend.candidate.profile import CandidateProfile
from backend.ingestion.normalize import Role
from backend.llm.client import parse_structured
from backend.llm.extract_requirements import RoleRequirement
from backend.llm.prompts import RATIONALE_SYSTEM_PROMPT, build_rationale_user_prompt
from backend.matching.gaps import SkillGapResult
from backend.matching.scorer import FitScoreResult
from backend.rag.retrieval import (
    SkillEvidenceBundle,
    retrieve_candidate_evidence,
    retrieve_role_content,
    retrieve_skill_evidence,
)
from backend.rag.vector_store import RetrievedChunk

logger = logging.getLogger(__name__)

# The six backend.matching.scorer.FitScoreComponents dimensions, in
# display order. "technical" has no entry in _DIMENSION_QUERY_TEMPLATE
# below because its evidence is the union of the per-skill evidence
# already gathered for skill_bundles, not a fresh retrieval query - see
# _technical_dimension_bundle().
_ALL_DIMENSIONS = ["technical", "experience", "coursework", "domain", "interest", "constraints"]


class EvidenceReference(BaseModel):
    """A citation back to one retrieved Qdrant chunk that backed a rationale claim."""

    source_type: str
    skill: str | None = None
    text: str
    score: float


class SkillRationale(BaseModel):
    """
    Rationale for one skill. `score_contribution` and `confidence` are
    copied directly from that skill's SkillGapResult - the LLM never
    produces these numbers, only `assessment`, `risk`, and
    `insufficient_evidence`.
    """

    skill: str
    candidate_evidence: list[str] = Field(default_factory=list)
    role_requirement_evidence: list[str] = Field(default_factory=list)
    assessment: str
    score_contribution: float = Field(
        ge=0, le=1, description="gap.satisfaction_ratio: fraction of the target level met, 0-1."
    )
    confidence: float = Field(ge=0, le=1)
    risk: str
    insufficient_evidence: bool = False


class DimensionRationale(BaseModel):
    """
    Rationale for one of the six FitScoreComponents dimensions
    (technical, experience, coursework, domain, interest, constraints).
    `score` and `confidence` are Python-computed - `score` is copied
    directly from the already-computed FitScoreResult, and `confidence`
    is either that dimension's own aggregate confidence (technical) or a
    deterministic function of retrieval-evidence quality (every other
    dimension, which has no native confidence measure) - the LLM never
    produces either number, only the narrative fields below.
    """

    dimension: str
    score: float
    rationale: str
    candidate_evidence: list[str] = Field(default_factory=list)
    role_evidence: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    recommended_action: str
    insufficient_evidence: bool = False


class FitRationale(BaseModel):
    """
    Complete rationale output. `overall_score` is copied verbatim from
    the FitScoreResult passed to generate_fit_rationale() - never
    produced or altered by the LLM - so any consumer can verify it
    matches the score that was actually computed. `biggest_risk` and
    `highest_impact_action` are likewise Python-derived (the first entry
    of `risks`/`recommended_actions`, with a fixed fallback string when
    empty), not separate LLM-authored fields, since "which risk/action is
    most important" is nothing the LLM was asked to rank independently.
    """

    overall_score: float
    overall_explanation: str
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    missing_requirements: list[str] = Field(default_factory=list)
    uncertain_areas: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    evidence_references: list[EvidenceReference] = Field(default_factory=list)
    skill_rationales: list[SkillRationale] = Field(default_factory=list)
    dimension_rationales: list[DimensionRationale] = Field(default_factory=list)
    why_this_role: str = ""
    why_not_this_role: str = ""
    biggest_risk: str = ""
    highest_impact_action: str = ""


class _LLMSkillNarrative(BaseModel):
    """Internal: the LLM's narrative-only contribution for one skill. Carries no score."""

    skill: str = Field(description="Must exactly match the skill identifier given in the prompt for this skill.")
    assessment: str
    risk: str
    insufficient_evidence: bool = False


class _LLMDimensionNarrative(BaseModel):
    """Internal: the LLM's narrative-only contribution for one fit-score dimension. Carries no score/confidence."""

    dimension: str = Field(
        description="Must exactly match the dimension identifier given in the prompt for this dimension."
    )
    rationale: str
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    recommended_action: str


class _RationaleNarrative(BaseModel):
    """
    Internal: the LLM's complete narrative-only output, before real
    numbers are spliced in. Deliberately has NO missing_requirements
    field - unlike the other list fields, "which required skills are
    unmet" is 100% derivable from already-computed SkillGapResult data
    (required=True and raw_gap>0), so generate_fit_rationale() computes
    it directly via _compute_missing_requirements() rather than trusting
    the LLM to enumerate it correctly. Likewise has no
    biggest_risk/highest_impact_action fields - see FitRationale's
    docstring for why those are Python-derived from `risks`/
    `recommended_actions` instead.
    """

    overall_explanation: str
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    uncertain_areas: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    skill_narratives: list[_LLMSkillNarrative] = Field(default_factory=list)
    dimension_narratives: list[_LLMDimensionNarrative] = Field(default_factory=list)
    why_this_role: str = ""
    why_not_this_role: str = ""


_DIMENSION_QUERY_TEMPLATE = {
    "experience": "candidate's internship, project, and research experience",
    "coursework": "candidate's academic coursework relevant to {title}",
    "domain": "candidate's domain experience relevant to {title} at {company}",
    "interest": "candidate's interest in working on {title}",
    "constraints": "candidate's location and scheduling constraints",
}


@dataclass
class _GatheredEvidence:
    """Everything retrieved from Qdrant for one rationale, before any LLM call is made."""

    skill_bundles: dict[str, SkillEvidenceBundle] = field(default_factory=dict)
    dimension_bundles: dict[str, SkillEvidenceBundle] = field(default_factory=dict)


def _technical_dimension_bundle(skill_bundles: dict[str, SkillEvidenceBundle]) -> SkillEvidenceBundle:
    """
    The "technical" dimension's evidence is the union of every skill's
    already-retrieved evidence - not a fresh Qdrant query, since it
    would just re-retrieve the same per-skill chunks under a broader
    query. Built here, at zero extra retrieval cost, from data
    _retrieve_evidence() already gathered.
    """
    candidate_evidence = [chunk for bundle in skill_bundles.values() for chunk in bundle.candidate_evidence]
    role_evidence = [chunk for bundle in skill_bundles.values() for chunk in bundle.role_evidence]
    return SkillEvidenceBundle(skill="technical", candidate_evidence=candidate_evidence, role_evidence=role_evidence)


def _retrieve_evidence(
    role: Role,
    skill_gaps: list[SkillGapResult],
    user_id: str,
    role_id: str,
    top_k_per_skill: int,
    top_k_per_dimension: int,
) -> _GatheredEvidence:
    """All Qdrant retrieval for a rationale happens here, and only here - called before _generate_narrative()."""
    skill_bundles = {
        gap.normalized_skill: retrieve_skill_evidence(user_id, role_id, gap.normalized_skill, top_k=top_k_per_skill)
        for gap in skill_gaps
    }
    dimension_bundles: dict[str, SkillEvidenceBundle] = {}
    for dimension, template in _DIMENSION_QUERY_TEMPLATE.items():
        query_text = template.format(title=role.title, company=role.company)
        dimension_bundles[dimension] = SkillEvidenceBundle(
            skill=dimension,
            candidate_evidence=retrieve_candidate_evidence(
                query_text, user_id, role_id=role_id, top_k=top_k_per_dimension
            ),
            role_evidence=retrieve_role_content(query_text, role_id, top_k=top_k_per_dimension),
        )
    dimension_bundles["technical"] = _technical_dimension_bundle(skill_bundles)
    return _GatheredEvidence(skill_bundles=skill_bundles, dimension_bundles=dimension_bundles)


def _build_skill_contexts(skill_gaps: list[SkillGapResult], skill_bundles: dict[str, SkillEvidenceBundle]) -> list[dict]:
    contexts = []
    for gap in skill_gaps:
        bundle = skill_bundles.get(gap.normalized_skill)
        candidate_chunks = bundle.candidate_evidence if bundle else []
        role_chunks = bundle.role_evidence if bundle else []
        contexts.append(
            {
                "skill_display": gap.display_skill,
                "normalized_skill": gap.normalized_skill,
                "target_level": gap.target_level,
                "candidate_level": gap.candidate_level,
                "importance": gap.importance,
                "required": gap.required,
                "satisfaction_ratio": gap.satisfaction_ratio,
                "confidence": gap.confidence,
                "candidate_evidence": [chunk.payload.get("text", "") for chunk in candidate_chunks],
                "role_evidence": [chunk.payload.get("text", "") for chunk in role_chunks],
            }
        )
    return contexts


def _build_dimension_contexts(dimension_bundles: dict[str, SkillEvidenceBundle]) -> list[dict[str, Any]]:
    contexts = []
    for dimension in _ALL_DIMENSIONS:
        bundle = dimension_bundles.get(dimension)
        candidate_chunks = bundle.candidate_evidence if bundle else []
        role_chunks = bundle.role_evidence if bundle else []
        contexts.append(
            {
                "dimension": dimension,
                "candidate_evidence": [chunk.payload.get("text", "") for chunk in candidate_chunks],
                "role_evidence": [chunk.payload.get("text", "") for chunk in role_chunks],
            }
        )
    return contexts


def _generate_narrative(
    role: Role,
    profile: CandidateProfile,
    fit_score: FitScoreResult,
    skill_gaps: list[SkillGapResult],
    evidence: _GatheredEvidence,
    model: str | None,
) -> _RationaleNarrative:
    """The only LLM call in this module. Produces narrative text only - never a score."""
    skill_contexts = _build_skill_contexts(skill_gaps, evidence.skill_bundles)
    dimension_contexts = _build_dimension_contexts(evidence.dimension_bundles)
    user_prompt = build_rationale_user_prompt(
        role_title=role.title,
        role_company=role.company,
        overall_score=fit_score.overall_score,
        component_scores=fit_score.components.model_dump(),
        profile_facts={
            "coursework": profile.coursework,
            "domain_experience": profile.domain_experience,
            "programming_languages": profile.programming_languages,
        },
        skill_contexts=skill_contexts,
        dimension_contexts=dimension_contexts,
    )
    logger.info("Generating fit rationale for role '%s' at '%s'", role.title, role.company)
    return parse_structured(
        system_prompt=RATIONALE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_model=_RationaleNarrative,
        model=model,
    )


def _to_evidence_reference(chunk: RetrievedChunk) -> EvidenceReference:
    return EvidenceReference(
        source_type=str(chunk.payload.get("source_type", "unknown")),
        skill=chunk.payload.get("skill"),
        text=str(chunk.payload.get("text", "")),
        score=chunk.score,
    )


def _compute_missing_requirements(skill_gaps: list[SkillGapResult]) -> list[str]:
    """
    Deterministic, Python-computed list of required skills the
    candidate does not yet meet - a direct readout of already-computed
    gap data (required=True and raw_gap>0), not something the LLM is
    asked to enumerate.
    """
    return [
        f"{gap.display_skill} (target level {gap.target_level:g}, candidate level {gap.candidate_level:g})"
        for gap in skill_gaps
        if gap.required and gap.raw_gap > 0
    ]


def _dedupe_evidence(references: list[EvidenceReference]) -> list[EvidenceReference]:
    seen: set[tuple[str, str | None, str]] = set()
    deduped: list[EvidenceReference] = []
    for reference in references:
        key = (reference.source_type, reference.skill, reference.text)
        if key not in seen:
            seen.add(key)
            deduped.append(reference)
    return deduped


def _first_or_fallback(items: list[str], fallback: str) -> str:
    """The first entry of an LLM-produced list, or a fixed fallback string if the list is empty."""
    return items[0] if items else fallback


def _dimension_confidence(chunks: list[RetrievedChunk]) -> float:
    """
    Deterministic confidence for a non-technical dimension: the average
    Qdrant similarity score across every chunk retrieved for it
    (candidate-side and role-side combined), or 0.0 if nothing was
    retrieved. The "technical" dimension does not use this - it has a
    real aggregate_confidence already computed by
    backend.matching.scorer.calculate_technical_fit().
    """
    if not chunks:
        return 0.0
    return round(sum(chunk.score for chunk in chunks) / len(chunks), 4)


def _assemble_dimension_rationales(
    fit_score: FitScoreResult,
    dimension_bundles: dict[str, SkillEvidenceBundle],
    narrative_by_dimension: dict[str, _LLMDimensionNarrative],
) -> list[DimensionRationale]:
    """Splice each dimension's already-computed score/confidence together with its narrative and evidence."""
    results: list[DimensionRationale] = []
    for dimension in _ALL_DIMENSIONS:
        bundle = dimension_bundles.get(dimension)
        candidate_chunks = bundle.candidate_evidence if bundle else []
        role_chunks = bundle.role_evidence if bundle else []
        llm_entry = narrative_by_dimension.get(dimension)
        has_evidence = bool(candidate_chunks) or bool(role_chunks)

        if dimension == "technical":
            confidence = fit_score.technical_detail.aggregate_confidence
        else:
            confidence = _dimension_confidence(candidate_chunks + role_chunks)

        results.append(
            DimensionRationale(
                dimension=dimension,
                score=getattr(fit_score.components, dimension),
                rationale=llm_entry.rationale if llm_entry else "Insufficient evidence to assess this dimension.",
                candidate_evidence=[chunk.payload.get("text", "") for chunk in candidate_chunks],
                role_evidence=[chunk.payload.get("text", "") for chunk in role_chunks],
                confidence=confidence,
                strengths=llm_entry.strengths if llm_entry else [],
                weaknesses=llm_entry.weaknesses if llm_entry else [],
                risks=llm_entry.risks if llm_entry else [],
                recommended_action=(
                    llm_entry.recommended_action if llm_entry else "Gather more evidence for this dimension."
                ),
                insufficient_evidence=not has_evidence,
            )
        )
    return results


def generate_fit_rationale(
    profile: CandidateProfile,
    role: Role,
    requirements: list[RoleRequirement],
    fit_score: FitScoreResult,
    skill_gaps: list[SkillGapResult],
    user_id: str,
    role_id: str,
    *,
    top_k_per_skill: int = 3,
    top_k_per_dimension: int = 3,
    model: str | None = None,
) -> FitRationale:
    """
    Retrieve grounding evidence from Qdrant for every skill in
    `skill_gaps` and for each of the six fit-score dimensions (retrieval
    happens entirely inside _retrieve_evidence(), before any LLM call),
    then generate narrative text explaining `fit_score` - which is
    treated as fixed, already-final input, never something the LLM can
    change.

    `requirements` is accepted (matching the full set of inputs a
    rationale conceptually depends on) but not read directly here: every
    number this function needs about a requirement is already present in
    `skill_gaps` (each SkillGapResult carries target_level, importance,
    required, etc.), so there's nothing further to pull from the raw
    RoleRequirement list.
    """
    evidence = _retrieve_evidence(role, skill_gaps, user_id, role_id, top_k_per_skill, top_k_per_dimension)
    narrative = _generate_narrative(role, profile, fit_score, skill_gaps, evidence, model)
    narrative_by_skill = {entry.skill: entry for entry in narrative.skill_narratives}

    skill_rationales: list[SkillRationale] = []
    evidence_references: list[EvidenceReference] = []

    for gap in skill_gaps:
        bundle = evidence.skill_bundles.get(gap.normalized_skill)
        candidate_chunks = bundle.candidate_evidence if bundle else []
        role_chunks = bundle.role_evidence if bundle else []
        llm_entry = narrative_by_skill.get(gap.normalized_skill)
        has_evidence = bool(candidate_chunks) or bool(role_chunks)

        skill_rationales.append(
            SkillRationale(
                skill=gap.display_skill,
                candidate_evidence=[chunk.payload.get("text", "") for chunk in candidate_chunks],
                role_requirement_evidence=[chunk.payload.get("text", "") for chunk in role_chunks],
                assessment=llm_entry.assessment if llm_entry else "Insufficient evidence to assess this skill.",
                score_contribution=gap.satisfaction_ratio,
                confidence=gap.confidence,
                risk=llm_entry.risk if llm_entry else "Unknown - insufficient evidence.",
                insufficient_evidence=(not has_evidence) or (llm_entry.insufficient_evidence if llm_entry else True),
            )
        )
        evidence_references.extend(_to_evidence_reference(chunk) for chunk in candidate_chunks)
        evidence_references.extend(_to_evidence_reference(chunk) for chunk in role_chunks)

    narrative_by_dimension = {entry.dimension: entry for entry in narrative.dimension_narratives}
    dimension_rationales = _assemble_dimension_rationales(fit_score, evidence.dimension_bundles, narrative_by_dimension)
    for bundle in evidence.dimension_bundles.values():
        evidence_references.extend(_to_evidence_reference(chunk) for chunk in bundle.candidate_evidence)
        evidence_references.extend(_to_evidence_reference(chunk) for chunk in bundle.role_evidence)

    return FitRationale(
        overall_score=fit_score.overall_score,
        overall_explanation=narrative.overall_explanation,
        strengths=narrative.strengths,
        weaknesses=narrative.weaknesses,
        risks=narrative.risks,
        missing_requirements=_compute_missing_requirements(skill_gaps),
        uncertain_areas=narrative.uncertain_areas,
        recommended_actions=narrative.recommended_actions,
        evidence_references=_dedupe_evidence(evidence_references),
        skill_rationales=skill_rationales,
        dimension_rationales=dimension_rationales,
        why_this_role=narrative.why_this_role or "Not enough evidence was retrieved to make a strong case for this role yet.",
        why_not_this_role=narrative.why_not_this_role
        or "Not enough evidence was retrieved to identify a strong concern about this role yet.",
        biggest_risk=_first_or_fallback(narrative.risks, "No significant risk identified from the available evidence."),
        highest_impact_action=_first_or_fallback(
            narrative.recommended_actions, "Gather more evidence (resume detail, coursework, projects) before acting."
        ),
    )
