"""
Role Analysis page: the full detailed fit/readiness breakdown for one
role. Reached from Discover or Favorites (never browsed to directly -
those pages set st.session_state["selected_role_id"] before switching
here) via "View Analysis" buttons.

This file renders only. Every number on it comes from
backend.services.discovery:
- analyze_role() computes/persists the deterministic fit score and
  skill gaps (the only place requirement-extraction's LLM call happens,
  and only on a role's first analysis - see that function's docstring).
- get_or_generate_rationale() returns the evidence-grounded narrative
  explaining that already-computed fit score. It is the most expensive
  step here (several Qdrant retrievals plus one LLM call) and is cached
  in Postgres (backend.db.rationales): opening this same role again, or
  a plain Streamlit rerun, reuses the cached rationale as long as the
  candidate's skills and the role's requirements haven't changed -
  never a fresh LLM/Qdrant round trip just because the script reran.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from backend.matching.confidence import confidence_label
from backend.matching.gaps import classify_gap_status
from backend.services import discovery, tracking
from ui import components
from ui_common import configure_page, database_not_configured_notice, get_current_user_id, is_demo_mode

configure_page("Role Analysis", icon="🔬")

_ALL_DIMENSIONS = ["technical", "experience", "coursework", "domain", "interest", "constraints"]
_STATUS_TONE = {"met": "positive", "partial": "warning", "uncertain": "warning", "missing": "negative"}


# ---------------------------------------------------------------------
# Demo data - a fully worked example so this page's structure can be
# previewed without OpenAI/Qdrant credentials configured. Mirrors the
# "demo-1" card already shown on Discover/Favorites in demo mode.
# ---------------------------------------------------------------------

_DEMO_HEADER = {
    "title": "Quantitative Research Intern",
    "company": "Meridian Capital",
    "location": "New York, NY",
    "deadline": None,
    "interview_date": "2026-09-13",
    "application_status": "interview",
    "priority": "dream",
    "is_saved": True,
}
_DEMO_FIT_SCORE = 57.4
_DEMO_READINESS = 31.5
_DEMO_COMPONENTS = {
    "technical": 58.3, "experience": 35.0, "coursework": 65.0,
    "domain": 50.0, "interest": 60.0, "constraints": 100.0,
}
_DEMO_SKILL_ROWS = [
    {"skill": "Python", "candidate_level": 8.0, "target_level": 7.0, "gap": 0.0, "importance": 7.0, "confidence": 0.8, "required": True, "status": "met"},
    {"skill": "C++", "candidate_level": 3.0, "target_level": 6.0, "gap": 3.0, "importance": 6.0, "confidence": 0.3, "required": False, "status": "uncertain"},
    {"skill": "Probability", "candidate_level": 0.0, "target_level": 8.0, "gap": 8.0, "importance": 9.0, "confidence": 0.0, "required": True, "status": "missing"},
]
_DEMO_DIMENSIONS: dict[str, dict[str, Any]] = {
    "technical": {
        "rationale": "Strong, well-evidenced Python skills from a real ETL project, but no demonstrated probability experience and only weak C++ exposure - both required for this role.",
        "candidate_evidence": ["Built Python ETL pipelines processing 2M+ rows/day", "Coursework: Data Structures, Algorithms"],
        "role_evidence": ["Strong Python and C++ programming skills required", "Solid foundation in probability and statistics"],
        "confidence": 0.62, "insufficient_evidence": False,
        "strengths": ["Demonstrated, production-grade Python experience."],
        "weaknesses": ["No resume evidence of probability or statistics coursework/projects.", "C++ mentioned only in a class listing, no applied evidence."],
        "risks": ["May struggle with probability-heavy technical interview rounds."],
        "recommended_action": "Complete a probability/statistics refresher and 1-2 applied C++ projects before applying.",
    },
    "experience": {
        "rationale": "One internship and two projects, but neither is in a trading, finance, or quant-adjacent setting.",
        "candidate_evidence": ["Software Engineering Intern, Acme Corp - built internal tooling"],
        "role_evidence": ["Prior internship or project experience with trading systems or market data preferred"],
        "confidence": 0.55, "insufficient_evidence": False,
        "strengths": ["One prior software engineering internship."],
        "weaknesses": ["No finance, trading, or market-data-adjacent experience."],
        "risks": ["Interviewers may probe for domain-relevant experience the resume doesn't show."],
        "recommended_action": "Highlight any quantitative, data-heavy aspects of the Acme internship explicitly in the application.",
    },
    "coursework": {
        "rationale": "Algorithms coursework covers part of the technical bar, but no probability/statistics coursework is listed.",
        "candidate_evidence": ["Coursework: Data Structures, Algorithms"],
        "role_evidence": ["Coursework in probability, statistics, or stochastic processes a plus"],
        "confidence": 0.5, "insufficient_evidence": False,
        "strengths": ["Algorithms coursework directly supports the technical requirements."],
        "weaknesses": ["No probability or statistics coursework listed."],
        "risks": ["A missing prerequisite course may be read as a gap by a reviewer."],
        "recommended_action": "Take or audit a probability/statistics course before the application deadline.",
    },
    "domain": {
        "rationale": "The candidate's domain experience does not mention trading, markets, or finance.",
        "candidate_evidence": [],
        "role_evidence": ["Exposure to trading systems or market microstructure a plus"],
        "confidence": 0.4, "insufficient_evidence": True,
        "strengths": [],
        "weaknesses": ["No stated domain experience relevant to trading or markets."],
        "risks": ["Domain unfamiliarity could surface in behavioral or fit interview questions."],
        "recommended_action": "Read up on market microstructure basics and reference it in the cover letter.",
    },
    "interest": {
        "rationale": "No explicit statement of interest in quantitative research is available in the current profile.",
        "candidate_evidence": [],
        "role_evidence": ["Looking for candidates passionate about markets and quantitative problem solving"],
        "confidence": 0.3, "insufficient_evidence": True,
        "strengths": [],
        "weaknesses": ["No declared interest signal specific to quant research."],
        "risks": ["A generic cover letter may read as low-conviction to this employer."],
        "recommended_action": "Add a specific, concrete reason for interest in quant research to the application.",
    },
    "constraints": {
        "rationale": "No location or work-authorization constraints are recorded that would conflict with this on-site New York role.",
        "candidate_evidence": [],
        "role_evidence": ["On-site, New York, NY"],
        "confidence": 0.5, "insufficient_evidence": False,
        "strengths": ["No known conflict with the role's on-site NYC location."],
        "weaknesses": [],
        "risks": [],
        "recommended_action": "Confirm relocation/housing logistics for a New York on-site internship.",
    },
}
_DEMO_WHY_THIS_ROLE = (
    "The candidate's strongest, best-evidenced skill (Python, backed by a real production ETL project) "
    "lines up directly with this role's requirements, and no location or availability constraint works against it."
)
_DEMO_WHY_NOT_THIS_ROLE = (
    "The role's highest-importance requirement, probability, has zero resume evidence behind it, and the "
    "candidate's domain/interest signals toward quantitative trading are also currently unevidenced."
)
_DEMO_BIGGEST_RISK = (
    "Probability is both required and the most heavily weighted requirement, with zero candidate evidence "
    "behind it - likely to surface directly in a technical interview."
)
_DEMO_HIGHEST_IMPACT_ACTION = "Complete a focused probability/statistics review before the interview on 2026-09-13."


def _render_evidence_list(snippets: list[str]) -> None:
    if not snippets:
        st.caption("None retrieved.")
        return
    for snippet in snippets:
        st.caption(f"“{snippet}”")


def _render_bullet_list(items: list[str]) -> None:
    for item in items or ["—"]:
        st.markdown(f"- {item}")


def _render_dimension_section(dimension: str, score: float | None, data: dict[str, Any] | None) -> None:
    label = dimension.upper() + (f" — {score:.0f}/100" if score is not None else "")
    with st.expander(label):
        if data is None:
            st.caption("No rationale available for this dimension yet.")
            return

        confidence = data["confidence"]
        components.render_status_badge(f"CONFIDENCE: {confidence_label(confidence).upper()} ({confidence:.2f})")
        if data["insufficient_evidence"]:
            st.warning("Insufficient evidence was found for this dimension.")
        st.write(data["rationale"])

        evidence_cols = st.columns(2)
        with evidence_cols[0]:
            st.markdown("**Candidate evidence**")
            _render_evidence_list(data["candidate_evidence"])
        with evidence_cols[1]:
            st.markdown("**Role evidence**")
            _render_evidence_list(data["role_evidence"])

        detail_cols = st.columns(3)
        with detail_cols[0]:
            st.markdown("**Strengths**")
            _render_bullet_list(data["strengths"])
        with detail_cols[1]:
            st.markdown("**Weaknesses**")
            _render_bullet_list(data["weaknesses"])
        with detail_cols[2]:
            st.markdown("**Risks**")
            _render_bullet_list(data["risks"])

        st.markdown(f"**Recommended action:** {data['recommended_action']}")


def _render_analysis(
    header: dict[str, Any],
    fit_score: float | None,
    readiness_score: float | None,
    components_scores: dict[str, float],
    dimension_data: dict[str, dict[str, Any]],
    skill_rows: list[dict[str, Any]],
    why_this_role: str,
    why_not_this_role: str,
    biggest_risk: str,
    highest_impact_action: str,
) -> None:
    # --- ROLE HEADER (equity-research-memo style) ---
    components.render_page_header("Role Analysis", f"{header['company']} — {header['title']}")

    top_cols = st.columns([1, 4, 2])
    with top_cols[0]:
        components.render_score_badge(fit_score, size="lg")
    with top_cols[1]:
        components.render_meta_line(
            [
                header.get("location"),
                f"Deadline {header['deadline']}" if header.get("deadline") else None,
                f"Interview {header['interview_date']}" if header.get("interview_date") else None,
                header["application_status"].upper() if header.get("application_status") else "NOT TRACKED",
            ]
        )
    with top_cols[2]:
        components.render_priority_badge(header.get("priority"))
        st.markdown("<br/>", unsafe_allow_html=True)
        components.render_status_badge("SAVED" if header.get("is_saved") else "NOT SAVED", tone="gold" if header.get("is_saved") else "default")

    components.render_metric_strip(
        [
            {"label": "Fit Score", "value": f"{fit_score:.0f}" if fit_score is not None else "—", "tone": "gold"},
            {"label": "Readiness", "value": f"{readiness_score:.0f}" if readiness_score is not None else "—"},
        ]
    )

    # --- EXECUTIVE SUMMARY ---
    components.render_section_header("Executive Summary")
    summary_cols = st.columns(2)
    with summary_cols[0]:
        with components.panel("match-thesis"):
            st.markdown('<p class="rr-eyebrow">Match Thesis</p>', unsafe_allow_html=True)
            st.write(why_this_role)
        with components.panel("highest-impact"):
            st.markdown('<p class="rr-eyebrow">Highest-Impact Action</p>', unsafe_allow_html=True)
            st.write(highest_impact_action)
    with summary_cols[1]:
        with components.panel("risks-gaps"):
            st.markdown('<p class="rr-eyebrow">Risks / Gaps</p>', unsafe_allow_html=True)
            st.write(why_not_this_role)
        with components.panel("biggest-risk"):
            st.markdown('<p class="rr-eyebrow">Biggest Risk</p>', unsafe_allow_html=True)
            st.write(biggest_risk)

    # --- FIT BREAKDOWN ---
    components.divider()
    components.render_section_header("Fit Breakdown")
    for dimension in _ALL_DIMENSIONS:
        _render_dimension_section(dimension, components_scores.get(dimension), dimension_data.get(dimension))

    # --- REQUIREMENT / SKILL MATCH BREAKDOWN ---
    components.divider()
    components.render_section_header("Requirement Breakdown")
    if skill_rows:
        components.render_table(
            ["Skill", "Candidate Level", "Target Level", "Gap", "Importance", "Confidence", "Status"],
            [
                [
                    row["skill"] + (" *" if row["required"] else ""),
                    f"{row['candidate_level']:g}",
                    f"{row['target_level']:g}",
                    f"{row['gap']:g}",
                    f"{row['importance']:g}",
                    f"{row['confidence']:.2f}",
                    components.status_badge_html(row["status"].upper(), tone=_STATUS_TONE.get(row["status"], "default")),
                ]
                for row in skill_rows
            ],
        )
        st.caption("* required requirement")

        missing = [row["skill"] for row in skill_rows if row["status"] == "missing"]
        uncertain = [row["skill"] for row in skill_rows if row["status"] == "uncertain"]
        callout_cols = st.columns(2)
        with callout_cols[0]:
            if missing:
                st.markdown('<p class="rr-eyebrow">Missing Skills</p>', unsafe_allow_html=True)
                components.render_tags(missing, kind="gap")
            else:
                components.render_status_badge("NO MISSING REQUIRED SKILLS", tone="positive")
        with callout_cols[1]:
            if uncertain:
                st.markdown('<p class="rr-eyebrow">Uncertain Skills (Low Evidence)</p>', unsafe_allow_html=True)
                components.render_tags(uncertain, kind="gap")
            else:
                components.render_status_badge("NO UNCERTAIN SKILLS", tone="positive")
    else:
        st.caption("No skill requirements to compare against yet.")

    # --- RECOMMENDED ACTION ---
    components.divider()
    components.render_section_header("Recommended Action")
    if fit_score is not None and fit_score >= 75:
        action, tone = "APPLY NOW", "positive"
    elif fit_score is not None and fit_score >= 50:
        action, tone = "PREP FIRST", "warning"
    elif fit_score is not None:
        action, tone = "LOW PRIORITY", "negative"
    else:
        action, tone = "NOT YET SCORED", "default"
    components.render_status_badge(action, tone=tone)


# ---------------------------------------------------------------------
# Resolve which role to show and render it
# ---------------------------------------------------------------------

selected_role_id = st.session_state.get("selected_role_id")

if is_demo_mode():
    if selected_role_id == "demo-2":
        components.render_page_header("Role Analysis", "Acme Corp — Software Engineering Intern")
        components.render_meta_line(["Remote", "Deadline 2026-09-14"])
        st.info(
            "This role hasn't been analyzed yet. Generating a full analysis is disabled in Demo mode - "
            "open the **Quantitative Research Intern** card from Discover or Favorites for a complete worked example."
        )
        st.stop()
    if selected_role_id != "demo-1":
        components.render_empty_state("No role selected", detail='Click "View Analysis" on a role from Discover or Favorites first.')
        st.stop()

    _render_analysis(
        header=_DEMO_HEADER,
        fit_score=_DEMO_FIT_SCORE,
        readiness_score=_DEMO_READINESS,
        components_scores=_DEMO_COMPONENTS,
        dimension_data=_DEMO_DIMENSIONS,
        skill_rows=_DEMO_SKILL_ROWS,
        why_this_role=_DEMO_WHY_THIS_ROLE,
        why_not_this_role=_DEMO_WHY_NOT_THIS_ROLE,
        biggest_risk=_DEMO_BIGGEST_RISK,
        highest_impact_action=_DEMO_HIGHEST_IMPACT_ACTION,
    )
else:
    if not selected_role_id:
        components.render_empty_state("No role selected", detail='Click "View Analysis" on a role from Discover or Favorites first.')
        st.stop()

    user_id = get_current_user_id()
    if user_id is None:
        database_not_configured_notice()
        st.stop()

    role_row = discovery.get_role(selected_role_id)
    if role_row is None:
        st.error("This role no longer exists.")
        st.stop()

    with st.spinner("Extracting requirements and scoring fit..."):
        try:
            analysis = discovery.analyze_role(user_id, role_row)
        except ValueError as exc:
            st.warning(str(exc))
            st.stop()

    with st.spinner("Loading evidence-grounded rationale (reused from cache unless your skills or this role's requirements changed)..."):
        rationale = discovery.get_or_generate_rationale(user_id, role_row, analysis)

    favorite = tracking.get_favorite(user_id, selected_role_id)
    application = tracking.get_application_status(user_id, selected_role_id)

    header = {
        "title": role_row["title"],
        "company": role_row["company"],
        "location": role_row.get("location"),
        "deadline": application.get("deadline") if application else None,
        "interview_date": application.get("interview_date") if application else None,
        "application_status": application["status"] if application else None,
        "priority": favorite["priority"] if favorite else None,
        "is_saved": favorite is not None,
    }
    fit_components = analysis.fit_result.components.model_dump()
    dimension_data = {
        d.dimension: {
            "candidate_evidence": d.candidate_evidence,
            "role_evidence": d.role_evidence,
            "confidence": d.confidence,
            "insufficient_evidence": d.insufficient_evidence,
            "rationale": d.rationale,
            "strengths": d.strengths,
            "weaknesses": d.weaknesses,
            "risks": d.risks,
            "recommended_action": d.recommended_action,
        }
        for d in rationale.dimension_rationales
    }
    skill_rows = [
        {
            "skill": gap.display_skill,
            "candidate_level": gap.candidate_level,
            "target_level": gap.target_level,
            "gap": gap.raw_gap,
            "importance": gap.importance,
            "confidence": gap.confidence,
            "required": gap.required,
            "status": classify_gap_status(gap),
        }
        for gap in analysis.gaps
    ]

    _render_analysis(
        header=header,
        fit_score=analysis.fit_result.overall_score,
        readiness_score=analysis.readiness.overall_readiness if analysis.readiness else None,
        components_scores=fit_components,
        dimension_data=dimension_data,
        skill_rows=skill_rows,
        why_this_role=rationale.why_this_role,
        why_not_this_role=rationale.why_not_this_role,
        biggest_risk=rationale.biggest_risk,
        highest_impact_action=rationale.highest_impact_action,
    )
