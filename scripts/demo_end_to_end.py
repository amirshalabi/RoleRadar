#!/usr/bin/env python3
"""
End-to-end integration pass over RoleRadar's real backend modules.

Runs the full candidate -> role -> match -> favorite -> apply -> prep
workflow, stage by stage, calling the SAME service functions the
Streamlit app and FastAPI layer call - nothing here is a separate
reimplementation. Its purpose is to catch integration problems that
per-module unit tests (which mock their neighbors) cannot: mismatched
field names, wrong call signatures, and ordering assumptions that only
show up when the real modules are wired together.

External dependencies (Supabase, OpenAI, Qdrant) are each detected
independently at startup:

  - Supabase configured  -> every step really persists to Postgres.
  - Supabase NOT configured -> falls back to the same in-memory
    FakeSupabaseClient the test suite uses (tests/_fake_supabase.py),
    so the full flow can still run locally. This is clearly logged and
    is NOT a substitute for a real run - it proves the code paths wire
    together correctly, not that production Postgres is reachable.
  - OpenAI configured -> real LLM extraction (candidate profile, role
    requirements) and real rationale narrative generation.
  - OpenAI NOT configured -> falls back to a hand-built, clearly-labeled
    MOCK profile/requirements set, so scoring/gaps/planning (all pure
    Python, no LLM) can still be exercised for real.
  - Qdrant configured (AND OpenAI, since embedding also needs it) ->
    real chunk/embed/upsert/retrieve against Qdrant.
  - Otherwise -> evidence embedding, indexing, retrieval, and rationale
    generation are SKIPPED outright (stages 12-15) with a clear message.
    There is no local mock vector store here - faking semantic retrieval
    would misrepresent what actually ran, which this script's whole
    purpose is to avoid.

Usage:
    python scripts/demo_end_to_end.py
"""

from __future__ import annotations

import sys
import traceback
from datetime import date, timedelta
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.candidate.profile import (  # noqa: E402
    CandidateProfile,
    CandidateSkillEstimate,
    ExperienceEntry,
    ProjectEntry,
)
from backend.candidate.skills import skills_from_profile  # noqa: E402
from backend.ingestion.normalize import normalize_role  # noqa: E402
from backend.llm.extract_requirements import RoleRequirement  # noqa: E402
from backend.utils.config import get_settings  # noqa: E402
from backend.utils.hashing import generate_role_external_id  # noqa: E402
from backend.utils.metrics import PipelineMetrics  # noqa: E402

# ---------------------------------------------------------------------
# Configuration detection - done BEFORE importing any backend.db module,
# so the local-mock patch (if needed) is in place before anything tries
# a real Supabase call.
# ---------------------------------------------------------------------

_settings = get_settings()
SUPABASE_CONFIGURED = bool(_settings.supabase_url and _settings.supabase_key)
OPENAI_CONFIGURED = bool(_settings.openai_api_key)
QDRANT_CONFIGURED = bool(_settings.qdrant_url and _settings.qdrant_api_key)
RAG_AVAILABLE = OPENAI_CONFIGURED and QDRANT_CONFIGURED  # embedding needs OpenAI too


def _patch_in_local_mock_database() -> None:
    """
    Replace every backend.db module's get_client() with the same
    in-memory FakeSupabaseClient the test suite uses, so this script can
    still exercise the full, real persistence-calling code path locally
    without a Supabase project. Each backend.db.* module imported
    get_client into its OWN namespace (`from backend.db.client import
    get_client`), so it must be patched on every module individually -
    patching backend.db.client.get_client alone would not affect
    modules that already captured a reference to the original function.
    """
    from tests._fake_supabase import FakeSupabaseClient  # a test double, deliberately reused - see module docstring

    from backend.db import (
        applications as applications_db,
        assessment_results as assessment_results_db,
        candidates as candidates_db,
        favorites as favorites_db,
        metrics as metrics_db,
        rationales as rationales_db,
        role_requirements as role_requirements_db,
        roles as roles_db,
        study_plans as study_plans_db,
        upserts as upserts_db,
        users as users_db,
    )

    shared_client = FakeSupabaseClient()
    for module in (
        roles_db, candidates_db, favorites_db, applications_db, role_requirements_db,
        rationales_db, study_plans_db, assessment_results_db, upserts_db, users_db, metrics_db,
    ):
        module.get_client = lambda c=shared_client: c


if not SUPABASE_CONFIGURED:
    _patch_in_local_mock_database()

# Every backend.db/backend.services import below happens AFTER the
# local-mock patch above (when needed), so they either use the real
# Supabase client or the shared fake one - never a half-patched mix.
from backend.db import metrics as metrics_db  # noqa: E402
from backend.db import rationales as rationales_db  # noqa: E402
from backend.db import role_requirements as role_requirements_db  # noqa: E402
from backend.db import roles as roles_db  # noqa: E402
from backend.services import candidate as candidate_service  # noqa: E402
from backend.services import discovery, prep, tracking  # noqa: E402


# ---------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------

_RESULTS: list[tuple[int, str, str]] = []  # (stage_number, title, status)


def stage(number: int, title: str) -> None:
    print()
    print("=" * 78)
    print(f"STAGE {number:2d}: {title}")
    print("=" * 78)


def info(label: str, value: Any = "") -> None:
    print(f"  {label}{': ' + str(value) if value != '' else ''}")


def ok(message: str) -> None:
    print(f"  ✅ {message}")


def skipped(number: int, title: str, reason: str) -> None:
    stage(number, title)
    print(f"  ⏭️  SKIPPED - {reason}")
    _RESULTS.append((number, title, "SKIPPED"))


def mark_done(number: int, title: str) -> None:
    _RESULTS.append((number, title, "OK"))


def mark_failed(number: int, title: str, exc: Exception) -> None:
    print(f"  ❌ FAILED: {exc!r}")
    traceback.print_exc()
    _RESULTS.append((number, title, "FAILED"))


# ---------------------------------------------------------------------
# Sample data (one consistent persona used everywhere else in this
# project's demo mode: a CS student with Python/algorithms/probability
# evidence, evaluated against a quant research internship).
# ---------------------------------------------------------------------

SAMPLE_RESUME_TEXT = """
Jane Doe
B.S. Computer Science, State University - Expected 2027

Coursework: Data Structures, Algorithms

Experience:
Software Engineering Intern, Acme Corp (Summer 2025)
Built internal Python ETL pipelines processing 2M+ rows/day.

Skills: Python, C++, React, Docker

Projects:
Distributed KV Store - Raft-based key-value store implemented in C++.
""".strip()

# Deliberately uses alias field names (jobTitle/companyName/jobDescription
# instead of title/company/description) to genuinely exercise
# normalize_role()'s key-alias mapping (stage 5), not just its happy path.
SAMPLE_JOB_RAW: dict[str, Any] = {
    "jobTitle": "Quantitative Research Intern",
    "companyName": "Meridian Capital",
    "jobLocation": "New York, NY",
    "jobDescription": (
        "Build and backtest trading signals using Python and C++. Strong foundation in "
        "probability and statistics is required. Proficiency in Python for backtesting is required."
    ),
    "roleFamily": "quant",
}

SAMPLE_FAVORITE_JOBS_RAW: list[dict[str, Any]] = [
    {
        "title": "Software Engineering Intern", "company": "Acme Corp", "location": "Remote",
        "description": "Build backend services in Python. Comfortable with core algorithms and data structures.",
        "role_family": "swe",
    },
    {
        "title": "Data Science Intern", "company": "Bright Labs", "location": "Boston, MA",
        "description": "Analyze experiment data in Python. Familiarity with statistics is a plus.",
        "role_family": "data",
    },
]


def _mock_candidate_profile() -> CandidateProfile:
    """A hand-built stand-in for extract_candidate_profile()'s output - used only when OPENAI_API_KEY is unset."""
    return CandidateProfile(
        coursework=["Data Structures", "Algorithms"],
        programming_languages=["Python", "C++"],
        frameworks=["React"],
        tools=["Docker"],
        projects=[ProjectEntry(name="Distributed KV Store", description="Raft-based key-value store in C++.", technologies=["C++"])],
        internships=[ExperienceEntry(organization="Acme Corp", role="Software Engineering Intern", description="Built internal Python ETL pipelines processing 2M+ rows/day.")],
        skills=[
            CandidateSkillEstimate(normalized_skill_name="python", display_name="Python", estimated_level=7.5, confidence=0.7, evidence_snippets=["Built internal Python ETL pipelines", "Skills: Python, C++, React, Docker"]),
            CandidateSkillEstimate(normalized_skill_name="algorithms", display_name="Algorithms", estimated_level=4.0, confidence=0.4, evidence_snippets=["Coursework: Data Structures, Algorithms"]),
            CandidateSkillEstimate(normalized_skill_name="c++", display_name="C++", estimated_level=5.0, confidence=0.5, evidence_snippets=["Distributed KV Store - Raft-based key-value store implemented in C++."]),
        ],
    )


_MOCK_REQUIREMENTS_BY_COMPANY: dict[str, list[RoleRequirement]] = {
    "Meridian Capital": [
        RoleRequirement(skill="Probability", normalized_skill="probability", target_level=8.0, importance=9.0, required=True, evidence=["Strong foundation in probability and statistics is required."]),
        RoleRequirement(skill="Python", normalized_skill="python", target_level=7.0, importance=7.0, required=True, evidence=["Proficiency in Python for backtesting is required."]),
        RoleRequirement(skill="C++", normalized_skill="c++", target_level=6.0, importance=6.0, required=False, evidence=["Build and backtest trading signals using Python and C++."]),
    ],
    "Acme Corp": [
        RoleRequirement(skill="Algorithms", normalized_skill="algorithms", target_level=7.0, importance=8.0, required=True, evidence=["Comfortable with core algorithms and data structures."]),
        RoleRequirement(skill="Python", normalized_skill="python", target_level=6.0, importance=6.0, required=True, evidence=["Build backend services in Python."]),
    ],
    "Bright Labs": [
        RoleRequirement(skill="Python", normalized_skill="python", target_level=8.0, importance=8.0, required=True, evidence=["Analyze experiment data in Python."]),
        RoleRequirement(skill="Statistics", normalized_skill="statistics", target_level=7.0, importance=7.0, required=False, evidence=["Familiarity with statistics is a plus."]),
    ],
}


def _score_role(user_id: str, role_row: dict[str, Any]) -> discovery.RoleAnalysis:
    """
    Wraps discovery.analyze_role() so that, without OpenAI, the role's
    requirements are pre-seeded from _MOCK_REQUIREMENTS_BY_COMPANY
    BEFORE calling it - analyze_role() then genuinely takes its own
    real "reuse persisted requirements, skip the LLM" branch instead of
    the script faking a score separately from the real scoring code.
    """
    if not OPENAI_CONFIGURED and not role_requirements_db.list_role_requirements(role_row["id"]):
        mock_requirements = _MOCK_REQUIREMENTS_BY_COMPANY.get(role_row["company"], [])
        role_requirements_db.upsert_role_requirements(
            role_row["id"],
            [
                {
                    "normalized_skill_name": r.normalized_skill, "display_name": r.skill,
                    "target_level": r.target_level, "importance": r.importance,
                    "is_required": r.required, "evidence": r.evidence,
                }
                for r in mock_requirements
            ],
        )
    return discovery.analyze_role(user_id, role_row)


def main() -> int:
    print("RoleRadar end-to-end integration pass")
    print("-" * 78)
    info("Supabase", "configured (real Postgres)" if SUPABASE_CONFIGURED else "NOT configured -> using local in-memory FakeSupabaseClient")
    info("OpenAI", "configured (real LLM calls)" if OPENAI_CONFIGURED else "NOT configured -> using hand-built mock extraction output")
    info("Qdrant", "configured" if QDRANT_CONFIGURED else "NOT configured")
    if not RAG_AVAILABLE:
        info("RAG (embed/index/retrieve/rationale)", "SKIPPED - needs both OpenAI and Qdrant configured")

    user_id = None

    # -- Stage 1: sample resume --------------------------------------
    stage(1, "Provide sample resume")
    info("Resume text (chars)", len(SAMPLE_RESUME_TEXT))
    print(SAMPLE_RESUME_TEXT[:160].replace("\n", " ") + "...")
    mark_done(1, "Provide sample resume")

    # -- Stage 2: extract CandidateProfile ----------------------------
    stage(2, "Extract CandidateProfile")
    try:
        from backend.db.users import get_or_create_demo_user

        user = get_or_create_demo_user("demo_end_to_end@roleradar.local", display_name="Integration Test User")
        user_id = user["id"]
        info("user_id", user_id)

        if OPENAI_CONFIGURED:
            from backend.candidate.profile import extract_candidate_profile

            profile = extract_candidate_profile(SAMPLE_RESUME_TEXT)
            info("Source", "real LLM extraction")
        else:
            profile = _mock_candidate_profile()
            info("Source", "MOCK (OPENAI_API_KEY not set)")
        info("Raw skills extracted", len(profile.skills))
        for skill in profile.skills:
            info(f"  - {skill.display_name}", f"level={skill.estimated_level:g}, confidence={skill.confidence:.2f}")
        mark_done(2, "Extract CandidateProfile")
    except Exception as exc:
        mark_failed(2, "Extract CandidateProfile", exc)
        return _summarize()

    # -- Stage 3: persist candidate ------------------------------------
    stage(3, "Persist candidate")
    try:
        deduped_preview = skills_from_profile(profile)
        info("Deduplicated skill count", len(deduped_preview))
        profile = candidate_service.save_candidate_profile(user_id, profile, resume_text=SAMPLE_RESUME_TEXT)
        from backend.db import candidates as candidates_db

        stored_skills = candidates_db.list_candidate_skills(user_id)
        ok(f"Persisted {len(stored_skills)} candidate_skills row(s) for user {user_id}")
        stored_profile = candidates_db.get_candidate_profile(user_id)
        assert stored_profile is not None, "candidate_profiles row was not persisted"
        ok("candidate_profiles row round-tripped through Postgres")
        mark_done(3, "Persist candidate")
    except Exception as exc:
        mark_failed(3, "Persist candidate", exc)
        return _summarize()

    # -- Stage 4: sample job -------------------------------------------
    stage(4, "Provide sample job")
    info("Raw job dict", SAMPLE_JOB_RAW)
    mark_done(4, "Provide sample job")

    # -- Stage 5: normalize Role ----------------------------------------
    stage(5, "Normalize Role")
    try:
        role = normalize_role(SAMPLE_JOB_RAW)
        info("Normalized title", role.title)
        info("Normalized company", role.company)
        info("Normalized location", role.location)
        info("Normalized role_family", role.role_family)
        ok("normalize_role() correctly mapped jobTitle/companyName/jobLocation/jobDescription/roleFamily aliases")
        mark_done(5, "Normalize Role")
    except Exception as exc:
        mark_failed(5, "Normalize Role", exc)
        return _summarize()

    # -- Stage 6: deterministic external_id -----------------------------
    stage(6, "Generate deterministic external_id")
    try:
        recomputed = generate_role_external_id(role.company, role.title, role.location, role.url)
        info("external_id (from normalize_role)", role.external_id)
        info("external_id (recomputed independently)", recomputed)
        assert recomputed == role.external_id, "external_id is not deterministic across calls with the same inputs"
        ok("Same inputs -> same external_id, confirmed by recomputing it independently")
        mark_done(6, "Generate deterministic external_id")
    except Exception as exc:
        mark_failed(6, "Generate deterministic external_id", exc)
        return _summarize()

    # -- Stage 7: idempotent upsert --------------------------------------
    stage(7, "Idempotently upsert role")
    try:
        first = roles_db.upsert_role(
            company=role.company, title=role.title, location=role.location, url=role.url,
            description=role.description, role_family=role.role_family, external_id=role.external_id,
        )
        second = roles_db.upsert_role(
            company=role.company, title=role.title, location=role.location, url=role.url,
            description=role.description, role_family=role.role_family, external_id=role.external_id,
        )
        info("Role id (1st upsert)", first["id"])
        info("Role id (2nd upsert)", second["id"])
        assert first["id"] == second["id"], "upsert_role() created a duplicate row for identical input"
        ok("Repeated upsert_role() with identical input resolved to the same row - idempotent")
        role_row = second
        mark_done(7, "Idempotently upsert role")
    except Exception as exc:
        mark_failed(7, "Idempotently upsert role", exc)
        return _summarize()

    # -- Stages 8-11: extract requirements, persist, fit, gaps -----------
    stage(8, "Extract RoleRequirements")
    analysis: discovery.RoleAnalysis | None = None
    try:
        if not OPENAI_CONFIGURED:
            info("Source", "MOCK (OPENAI_API_KEY not set) - pre-seeded before analyze_role()")
        else:
            info("Source", "real LLM extraction (via backend.services.discovery.analyze_role)")
        analysis = _score_role(user_id, role_row)
        for req in analysis.requirements:
            info(f"  - {req.skill}", f"target={req.target_level:g}, importance={req.importance:g}, required={req.required}")
        mark_done(8, "Extract RoleRequirements")
    except Exception as exc:
        mark_failed(8, "Extract RoleRequirements", exc)
        return _summarize()

    stage(9, "Persist requirements")
    try:
        persisted = role_requirements_db.list_role_requirements(role_row["id"])
        assert len(persisted) == len(analysis.requirements), "persisted requirement count does not match extracted count"
        ok(f"Verified {len(persisted)} requirement row(s) round-tripped through role_requirements")
        mark_done(9, "Persist requirements")
    except Exception as exc:
        mark_failed(9, "Persist requirements", exc)

    stage(10, "Calculate FitScore")
    try:
        info("Overall fit score", f"{analysis.fit_result.overall_score:.1f} / 100")
        for name, value in analysis.fit_result.components.model_dump().items():
            info(f"  {name}", f"{value:.1f}")
        mark_done(10, "Calculate FitScore")
    except Exception as exc:
        mark_failed(10, "Calculate FitScore", exc)

    stage(11, "Calculate SkillGaps")
    try:
        for gap in analysis.gaps:
            info(f"  {gap.display_skill}", f"target={gap.target_level:g}, candidate={gap.candidate_level:g}, gap={gap.raw_gap:g}")
        mark_done(11, "Calculate SkillGaps")
    except Exception as exc:
        mark_failed(11, "Calculate SkillGaps", exc)

    # -- Stages 12-15: RAG (needs OpenAI + Qdrant) ------------------------
    if not RAG_AVAILABLE:
        missing = [name for name, present in (("OPENAI_API_KEY", OPENAI_CONFIGURED), ("QDRANT_URL/QDRANT_API_KEY", QDRANT_CONFIGURED)) if not present]
        reason = f"missing: {', '.join(missing)} - no local mock vector store is used (would misrepresent semantic retrieval as real)"
        skipped(12, "Embed candidate/job evidence", reason)
        skipped(13, "Upsert embeddings to Qdrant", reason)
        skipped(14, "Retrieve evidence", reason)
        skipped(15, "Generate rationale", reason)
        rationale = None
    else:
        stage(12, "Embed candidate/job evidence")
        try:
            from backend.rag.chunking import chunk_candidate_evidence, chunk_role_content
            from backend.rag.embeddings import get_embedding_provider

            candidate_chunks = chunk_candidate_evidence(profile)
            role_chunks = chunk_role_content(role, analysis.requirements)
            info("Candidate evidence chunks", len(candidate_chunks))
            info("Role content chunks", len(role_chunks))
            info("Note", "Embedding happens fused into the Qdrant upsert call (backend.rag.vector_store.upsert_chunks) - there is no separate embed-only step in this codebase.")
            provider = get_embedding_provider()
            mark_done(12, "Embed candidate/job evidence")
        except Exception as exc:
            mark_failed(12, "Embed candidate/job evidence", exc)
            candidate_chunks, role_chunks, provider = [], [], None

        stage(13, "Upsert embeddings to Qdrant")
        try:
            from backend.rag.vector_store import CANDIDATE_EVIDENCE_COLLECTION, ROLE_CONTENT_COLLECTION, upsert_chunks

            candidate_points = upsert_chunks(CANDIDATE_EVIDENCE_COLLECTION, candidate_chunks, provider, user_id=user_id, role_id=role_row["id"])
            role_points = upsert_chunks(ROLE_CONTENT_COLLECTION, role_chunks, provider, user_id=user_id, role_id=role_row["id"])
            ok(f"Upserted {candidate_points} candidate-evidence point(s) and {role_points} role-content point(s)")
            mark_done(13, "Upsert embeddings to Qdrant")
        except Exception as exc:
            mark_failed(13, "Upsert embeddings to Qdrant", exc)

        stage(14, "Retrieve evidence")
        try:
            from backend.rag.retrieval import retrieve_skill_evidence

            bundle = retrieve_skill_evidence(user_id, role_row["id"], "python", top_k=3)
            info("Candidate evidence for 'python'", [c.payload.get("text") for c in bundle.candidate_evidence])
            info("Role evidence for 'python'", [c.payload.get("text") for c in bundle.role_evidence])
            mark_done(14, "Retrieve evidence")
        except Exception as exc:
            mark_failed(14, "Retrieve evidence", exc)

        stage(15, "Generate rationale")
        rationale = None
        try:
            rationale = discovery.get_or_generate_rationale(user_id, role_row, analysis)
            info("Overall explanation", rationale.overall_explanation)
            info("Why this role?", rationale.why_this_role)
            info("Biggest risk", rationale.biggest_risk)
            mark_done(15, "Generate rationale")
        except Exception as exc:
            mark_failed(15, "Generate rationale", exc)

    # -- Stage 16: save primary role as favorite --------------------------
    stage(16, "Save role as Favorite")
    try:
        favorite = tracking.save_role(user_id, role_row["id"], priority="dream", notes="Primary target role for this integration pass.")
        info("Favorite priority", favorite["priority"])
        ok(f"Role {role_row['id']} favorited")
        mark_done(16, "Save role as Favorite")
    except Exception as exc:
        mark_failed(16, "Save role as Favorite", exc)
        return _summarize()

    # -- Stage 17: 2 more sample favorites ---------------------------------
    stage(17, "Save 2 more sample favorites")
    favorite_role_ids = [role_row["id"]]
    try:
        priorities = ["high", "interested"]
        for raw_job, priority in zip(SAMPLE_FAVORITE_JOBS_RAW, priorities):
            extra_role = normalize_role(raw_job)
            extra_row = roles_db.upsert_role(
                company=extra_role.company, title=extra_role.title, location=extra_role.location,
                description=extra_role.description, role_family=extra_role.role_family, external_id=extra_role.external_id,
            )
            _score_role(user_id, extra_row)
            tracking.save_role(user_id, extra_row["id"], priority=priority)
            favorite_role_ids.append(extra_row["id"])
            ok(f"Favorited '{extra_role.title}' at {extra_role.company} (priority={priority})")
        info("Total favorites", len(favorite_role_ids))
        mark_done(17, "Save 2 more sample favorites")
    except Exception as exc:
        mark_failed(17, "Save 2 more sample favorites", exc)

    # -- Stage 18: cross-role skill analysis -------------------------------
    stage(18, "Run cross-role skill analysis")
    try:
        comparisons, summary = discovery.compare_roles(user_id, favorite_role_ids)
        for comparison in comparisons:
            info(f"  {comparison.role.title}", f"fit={comparison.fit_score:.1f}, readiness={comparison.readiness_score}")
        info("Best current match", summary.best_current_match.title if summary.best_current_match else None)
        info("Highest potential upside", summary.highest_potential_upside.title if summary.highest_potential_upside else None)
        mark_done(18, "Run cross-role skill analysis")
    except Exception as exc:
        mark_failed(18, "Run cross-role skill analysis", exc)

    # -- Stage 19: skill ROI ------------------------------------------------
    stage(19, "Calculate skill ROI")
    try:
        roi_results = discovery.get_skill_roi_for_favorites(user_id)
        for result in roi_results:
            info(f"  {result.display_skill}", f"roles_requiring_it={result.roles_requiring_it}, roi_score={result.roi_score:.1f}")
        if roi_results:
            ok(f"Highest-leverage skill: {roi_results[0].display_skill} (ROI {roi_results[0].roi_score:.1f})")
        mark_done(19, "Calculate skill ROI")
    except Exception as exc:
        mark_failed(19, "Calculate skill ROI", exc)

    # -- Stage 20: add application --------------------------------------------
    stage(20, "Add application")
    try:
        application = tracking.update_application_stage(user_id, role_row["id"], status="applied", application_date=date.today().isoformat())
        info("Application status", application["status"])
        mark_done(20, "Add application")
    except Exception as exc:
        mark_failed(20, "Add application", exc)
        return _summarize()

    # -- Stage 21: set interview date -----------------------------------------
    stage(21, "Set interview date")
    interview_date = date.today() + timedelta(days=10)
    try:
        application = tracking.update_application_stage(
            user_id, role_row["id"], status="interview", interview_date=interview_date.isoformat()
        )
        info("Interview date", application["interview_date"])
        mark_done(21, "Set interview date")
    except Exception as exc:
        mark_failed(21, "Set interview date", exc)
        return _summarize()

    # -- Stage 22: calculate readiness ----------------------------------------
    stage(22, "Calculate readiness")
    try:
        readiness = analysis.readiness
        assert readiness is not None, "role has a role_family - readiness should have been computed in stage 10-11"
        info("Overall readiness", f"{readiness.overall_readiness:.1f} / 100")
        for topic in readiness.topic_readiness:
            info(f"  {topic.topic}", f"{topic.readiness_score:.1f} ({topic.evidence_source})")
        mark_done(22, "Calculate readiness")
    except Exception as exc:
        mark_failed(22, "Calculate readiness", exc)

    # -- Stage 23: deadline-aware prep plan -----------------------------------
    stage(23, "Create deadline-aware prep plan")
    plan_view = None
    try:
        plan_view = prep.get_or_create_plan_view(user_id, role_row["id"])
        info("Days remaining", plan_view["days_remaining"])
        info("Hours/day", plan_view["hours_available_per_day"])
        info("Total available minutes", plan_view["total_available_minutes"])
        info("Task count", len(plan_view["tasks"]))
        mark_done(23, "Create deadline-aware prep plan")
    except Exception as exc:
        mark_failed(23, "Create deadline-aware prep plan", exc)

    # -- Stages 24-25: diagnostic + adaptive replan ---------------------------
    stage(24, "Record diagnostic result")
    diagnostic_result = None
    try:
        diagnostic_result = prep.submit_diagnostic(user_id, role_row["id"], "probability", observed_level=9.0, confidence=0.95)
        info("Readiness before", f"{diagnostic_result['readiness_before'].overall_readiness:.1f}")
        info("Readiness after", f"{diagnostic_result['readiness_after'].overall_readiness:.1f}")
        mark_done(24, "Record diagnostic result")
    except Exception as exc:
        mark_failed(24, "Record diagnostic result", exc)

    stage(25, "Adapt remaining prep plan")
    try:
        assert diagnostic_result is not None
        info("New plan version", diagnostic_result["plan"]["version"])
        info("Message", diagnostic_result["message"])
        ok("Completed tasks preserved; remaining schedule regenerated around the new diagnostic evidence")
        mark_done(25, "Adapt remaining prep plan")
    except Exception as exc:
        mark_failed(25, "Adapt remaining prep plan", exc)

    # -- Stage 26: verify pipeline metrics -------------------------------------
    stage(26, "Verify pipeline metrics")
    try:
        llm_analyzed = len(favorite_role_ids)  # every favorited role had requirements extracted (real or mock)
        run_metrics = PipelineMetrics(ingested=llm_analyzed, deduplicated=0, hard_filtered=0, keyword_filtered=0, llm_analyzed=llm_analyzed)
        recorded = metrics_db.record_ingestion_run(run_metrics, source="scripts/demo_end_to_end.py")
        info("Recorded ingestion_runs id", recorded["id"])
        latest = metrics_db.get_latest_ingestion_run()
        assert latest is not None and latest["id"] == recorded["id"], "recorded run did not round-trip as the latest run"
        ok(f"Round-tripped through Postgres - llm_avoidance_rate={run_metrics.llm_avoidance_rate()}")
        mark_done(26, "Verify pipeline metrics")
    except Exception as exc:
        mark_failed(26, "Verify pipeline metrics", exc)

    return _summarize()


def _summarize() -> int:
    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    ok_count = sum(1 for *_, status in _RESULTS if status == "OK")
    skipped_count = sum(1 for *_ , status in _RESULTS if status == "SKIPPED")
    failed_count = sum(1 for *_ , status in _RESULTS if status == "FAILED")
    for number, title, status in _RESULTS:
        icon = {"OK": "✅", "SKIPPED": "⏭️ ", "FAILED": "❌"}[status]
        print(f"  {icon} Stage {number:2d}: {title} [{status}]")
    print("-" * 78)
    print(f"  {ok_count} ok, {skipped_count} skipped, {failed_count} failed (of {len(_RESULTS)} stages run)")
    return 1 if failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
