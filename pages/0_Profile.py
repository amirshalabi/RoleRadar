"""
Profile page: resume upload/replace and the full parsed candidate
profile - the first-class onboarding surface every other page depends
on (Discover's fit scores, Skill Gaps' baseline, Interview Prep's
readiness, Role Analysis's evidence/rationale all read the same
candidate_profiles/candidate_skills rows this page writes).

This file renders and orchestrates only. The PDF-to-text step
(backend.candidate.parser.extract_text_from_pdf) is a stateless utility
safe to call directly; everything with a side effect - the LLM
extraction call, Postgres persistence, Qdrant evidence indexing - goes
through backend.services.candidate.process_resume_upload(), the single
entry point that keeps this page from re-implementing any of that
logic. Preferences (user-STATED target roles/locations/employment
type/interests) persist through backend.services.preferences, a
separate table from resume-INFERRED facts - see sql/schema.sql's
candidate_preferences docstring for why they must never merge.
"""

from __future__ import annotations

import logging

import streamlit as st

from backend.candidate.parser import extract_text_from_pdf
from backend.candidate.profile import CandidateProfile, CandidateSkillEstimate, EducationEntry, ExperienceEntry, ProjectEntry, ResearchEntry
from backend.db.client import SupabaseNotConfiguredError
from backend.llm.client import LLMExtractionError, OpenAINotConfiguredError
from backend.services import candidate as candidate_service
from backend.services import preferences as preferences_service
from backend.services.candidate import ResumeProcessingError
from ui import components
from ui_common import configure_page, database_not_configured_notice, get_current_user_id, is_demo_mode

logger = logging.getLogger(__name__)

configure_page("Profile", icon="🧬")
components.render_page_header(
    "Candidate Intelligence",
    "Profile",
    "Your resume powers every fit score, skill gap, and readiness estimate in RoleRadar.",
)


# ---------------------------------------------------------------------
# Demo data - a fuller worked example than the two-skill fixture reused
# elsewhere, so every section on this page (education/coursework/
# projects/research/domain experience, not just skills) has something
# real-looking to show. Clearly labeled Demo mode by configure_page().
# ---------------------------------------------------------------------

_DEMO_PROFILE = CandidateProfile(
    education=[EducationEntry(institution="State University", degree="B.S.", major="Computer Science", graduation_year=2027)],
    coursework=["Data Structures", "Algorithms", "Operating Systems", "Probability & Statistics"],
    programming_languages=["Python", "C++", "SQL"],
    frameworks=["FastAPI", "PyTorch"],
    tools=["Docker", "Git"],
    projects=[
        ProjectEntry(name="Distributed KV Store", description="Raft-based key-value store in C++ with leader election and log replication.", technologies=["C++", "Networking"]),
    ],
    internships=[
        ExperienceEntry(organization="Acme Corp", role="Software Engineering Intern", description="Built internal Python ETL pipelines processing 2M+ rows/day.", start_date="2025-06", end_date="2025-08"),
    ],
    research=[],
    domain_experience=["Backend systems"],
    skills=[
        CandidateSkillEstimate(normalized_skill_name="python", display_name="Python", estimated_level=7.5, confidence=0.7, evidence_snippets=["Built internal Python ETL pipelines processing 2M+ rows/day", "Skills: Python, C++, SQL"]),
        CandidateSkillEstimate(normalized_skill_name="algorithms", display_name="Algorithms", estimated_level=4.0, confidence=0.4, evidence_snippets=["Coursework: Data Structures, Algorithms"]),
        CandidateSkillEstimate(normalized_skill_name="probability", display_name="Probability", estimated_level=3.0, confidence=0.25, evidence_snippets=[]),
    ],
)
_DEMO_RESUME_META = {"filename": "sample_resume.pdf", "parsed_at": "2026-08-20T00:00:00+00:00"}
_DEMO_PREFERENCES = {"target_role_families": ["SWE"], "preferred_locations": ["Remote", "New York, NY"], "employment_types": ["Internship"], "interests": ["Systems", "Data"]}


# ---------------------------------------------------------------------
# Upload / replace flow
# ---------------------------------------------------------------------


def _friendly_error_message(exc: Exception) -> str:
    if isinstance(exc, ResumeProcessingError):
        return str(exc)
    if isinstance(exc, OpenAINotConfiguredError):
        return "AI extraction isn't configured yet. Set `OPENAI_API_KEY` in your `.env` to enable resume parsing."
    if isinstance(exc, SupabaseNotConfiguredError):
        return "Database not connected. Set `SUPABASE_URL` and `SUPABASE_KEY` in your `.env` to save your profile."
    if isinstance(exc, LLMExtractionError):
        return "RoleRadar's AI couldn't extract a structured profile from that resume. Try again, or paste plain text instead."
    return "Something went wrong while analyzing your resume. Please try again in a moment."


def _process_upload(user_id: str, resume_text: str, filename: str | None) -> None:
    with st.spinner("Analyzing resume..."):
        try:
            result = candidate_service.process_resume_upload(user_id, resume_text, filename=filename)
        except Exception as exc:  # noqa: BLE001 - mapped to a friendly, typed message below; technical detail is already logged in the service layer
            logger.warning("Resume upload failed for user %s: %s", user_id, exc)
            st.session_state["profile_upload_notice"] = {"kind": "error", "message": _friendly_error_message(exc)}
            st.rerun()
            return

    if not result.is_new_resume:
        message = "This is the same resume you already have on file - nothing to re-analyze."
        kind = "info"
    elif not result.evidence_indexed:
        message = (
            f"Resume analyzed successfully - {result.skills_saved} skill(s) identified. "
            f"Evidence indexing failed ({result.evidence_index_error}), so role-analysis rationale may have limited "
            "supporting evidence until it's retried."
        )
        kind = "warning"
    else:
        message = f"Resume analyzed successfully - {result.skills_saved} skill(s) identified."
        kind = "success"

    st.session_state["profile_upload_notice"] = {"kind": kind, "message": message}
    st.rerun()


def _render_upload_controls(user_id: str, *, prominent: bool) -> None:
    uploaded = st.file_uploader("Upload Resume", type=["pdf"], key="resume_pdf_uploader" if prominent else "resume_pdf_uploader_replace")
    if uploaded is not None:
        if st.button("Analyze Resume", type="primary", key="analyze_pdf_btn" if prominent else "analyze_pdf_btn_replace"):
            try:
                text = extract_text_from_pdf(uploaded.getvalue())
            except Exception:
                logger.exception("Failed to extract text from uploaded PDF %r", uploaded.name)
                st.session_state["profile_upload_notice"] = {
                    "kind": "error",
                    "message": "We couldn't read that PDF. Try exporting it again, or paste your resume text instead.",
                }
                st.rerun()
            else:
                _process_upload(user_id, text, uploaded.name)

    st.caption("Supported: PDF")

    with st.expander("Paste Resume Text Instead"):
        pasted = st.text_area("Resume text", height=200, key="resume_paste_area", label_visibility="collapsed")
        if st.button("Analyze Pasted Text", key="analyze_paste_btn" if prominent else "analyze_paste_btn_replace"):
            _process_upload(user_id, pasted, None)


# ---------------------------------------------------------------------
# Results rendering
# ---------------------------------------------------------------------


def _render_notice() -> None:
    notice = st.session_state.pop("profile_upload_notice", None)
    if not notice:
        return
    {"success": st.success, "warning": st.warning, "error": st.error, "info": st.info}[notice["kind"]](notice["message"])


def _render_profile_summary(profile: CandidateProfile, resume_meta: dict | None, *, demo: bool) -> None:
    top_skills = sorted(profile.skills, key=lambda s: s.estimated_level, reverse=True)[:6]

    metrics = [
        {"label": "Skills Identified", "value": str(len(profile.skills)), "tone": "gold"},
        {"label": "Projects", "value": str(len(profile.projects))},
        {"label": "Work Experience", "value": str(len(profile.internships))},
        {"label": "Coursework", "value": str(len(profile.coursework))},
    ]
    components.render_metric_strip(metrics)

    if resume_meta and resume_meta.get("filename"):
        components.render_meta_line([resume_meta["filename"], f"Parsed {resume_meta['parsed_at']}" if resume_meta.get("parsed_at") else None])

    if top_skills:
        st.markdown('<p class="rr-eyebrow">Top Skills</p>', unsafe_allow_html=True)
        components.render_tags([s.display_name for s in top_skills], kind="gold")


def _render_sections(profile: CandidateProfile) -> None:
    components.divider()
    components.render_section_header("Technical Skills", "Level and confidence are shown separately - confidence reflects how much resume evidence RoleRadar found, not the skill level itself.")
    if profile.skills:
        for skill in sorted(profile.skills, key=lambda s: s.estimated_level, reverse=True):
            components.render_skill_evidence(skill.display_name, skill.estimated_level, skill.confidence, skill.evidence_snippets)
    else:
        st.caption("No technical skills extracted yet.")

    components.divider()
    components.render_section_header("Education")
    if profile.education:
        for entry in profile.education:
            with components.panel(f"edu-{entry.institution}"):
                st.markdown(f"**{entry.institution}**")
                components.render_meta_line([entry.degree, entry.major, str(entry.graduation_year) if entry.graduation_year else None])
    else:
        st.caption("No education history extracted yet.")

    components.render_section_header("Experience")
    if profile.internships:
        for entry in profile.internships:
            with components.panel(f"exp-{entry.organization}-{entry.role}"):
                st.markdown(f"**{entry.role}** · {entry.organization}")
                components.render_meta_line([entry.start_date, entry.end_date])
                st.caption(entry.description)
    else:
        st.caption("No work experience extracted yet.")

    components.render_section_header("Projects")
    if profile.projects:
        for entry in profile.projects:
            with components.panel(f"proj-{entry.name}"):
                st.markdown(f"**{entry.name}**")
                st.caption(entry.description)
                if entry.technologies:
                    components.render_tags(entry.technologies)
    else:
        st.caption("No projects extracted yet.")

    components.render_section_header("Coursework")
    if profile.coursework:
        components.render_tags(profile.coursework)
    else:
        st.caption("No coursework extracted yet.")

    components.render_section_header("Research")
    if profile.research:
        for entry in profile.research:
            with components.panel(f"research-{entry.title}"):
                st.markdown(f"**{entry.title}**")
                st.caption(entry.description)
    else:
        st.caption("No research experience extracted yet.")

    components.render_section_header("Domains / Interests (Resume-Inferred)")
    if profile.domain_experience:
        components.render_tags(profile.domain_experience)
    else:
        st.caption("No domain experience extracted yet.")


def _render_completeness(profile: CandidateProfile, preferences: dict | None) -> None:
    components.divider()
    components.render_section_header("Profile Completeness")
    items = [
        {"label": "Resume", "done": True},
        {"label": "Skills", "done": len(profile.skills) > 0},
        {"label": "Coursework", "done": len(profile.coursework) > 0},
        {"label": "Projects", "done": len(profile.projects) > 0},
        {"label": "Target Roles", "done": bool(preferences and preferences.get("target_role_families"))},
        {"label": "Preferences", "done": bool(preferences and any(preferences.get(k) for k in ("preferred_locations", "employment_types", "interests")))},
    ]
    with components.panel("completeness"):
        components.render_checklist(items)
        completed = sum(1 for item in items if item["done"])
        st.caption(f"{completed} of {len(items)} completed — computed from real persisted fields, not estimated.")


def _render_preferences_form(user_id: str | None, preferences: dict | None, *, demo: bool) -> None:
    components.divider()
    components.render_section_header("Candidate Preferences", "User-stated, kept separate from resume-inferred facts - never mixed with your extracted skills.")

    current = preferences or {}
    with st.form("preferences_form"):
        target_role_families = st.multiselect(
            "Target role families", options=preferences_service.TARGET_ROLE_FAMILY_OPTIONS,
            default=current.get("target_role_families", []),
        )
        preferred_locations_text = st.text_input(
            "Preferred locations (comma-separated)", value=", ".join(current.get("preferred_locations", [])),
        )
        employment_types = st.multiselect(
            "Employment type", options=preferences_service.EMPLOYMENT_TYPE_OPTIONS,
            default=current.get("employment_types", []),
        )
        interests = st.multiselect(
            "Optional interests", options=preferences_service.INTEREST_OPTIONS,
            default=current.get("interests", []),
        )
        submitted = st.form_submit_button("Save Preferences", type="primary", disabled=demo)

    if submitted and not demo and user_id:
        preferred_locations = [loc.strip() for loc in preferred_locations_text.split(",") if loc.strip()]
        try:
            preferences_service.save_candidate_preferences(
                user_id, target_role_families, preferred_locations, employment_types, interests
            )
        except Exception:
            logger.exception("Failed to save candidate preferences for user %s", user_id)
            st.error("Couldn't save preferences right now. Your resume and skills are unaffected.")
        else:
            st.success("Preferences saved.")
            st.rerun()


# ---------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------

def _safe_get_preferences(user_id: str) -> dict | None:
    """Preferences are a secondary feature - a failure here (e.g. sql/schema.sql not yet re-applied for the candidate_preferences table) must not take down the rest of the Profile page, which is the primary resume-onboarding flow."""
    try:
        return preferences_service.get_candidate_preferences(user_id)
    except Exception:
        logger.exception("Failed to load candidate preferences for user %s", user_id)
        return None


_render_notice()

demo = is_demo_mode()

if demo:
    profile = _DEMO_PROFILE
    resume_meta = _DEMO_RESUME_META
    preferences = _DEMO_PREFERENCES
    user_id = None
else:
    user_id = get_current_user_id()
    if user_id is None:
        database_not_configured_notice()
        st.stop()
    profile = candidate_service.load_candidate_profile(user_id)
    resume_meta = candidate_service.get_resume_metadata(user_id)
    preferences = _safe_get_preferences(user_id)

if profile is None:
    with components.panel("resume-empty-state"):
        st.markdown("### Your Candidate Profile")
        st.write("Upload your resume to personalize RoleRadar.")
        st.caption(
            "RoleRadar uses your resume to extract technical skills, coursework, projects, work experience, and "
            "domain knowledge. This profile powers job matching, skill-gap analysis, and interview-readiness scoring."
        )
        if demo:
            st.info("Resume upload is disabled in Demo mode. Turn off Demo mode to upload your real resume.", icon="🧪")
        else:
            _render_upload_controls(user_id, prominent=True)
else:
    _render_profile_summary(profile, resume_meta, demo=demo)

    action_cols = st.columns(2)
    with action_cols[0]:
        with st.expander("Replace Resume"):
            if demo:
                st.info("Resume upload is disabled in Demo mode.", icon="🧪")
            else:
                _render_upload_controls(user_id, prominent=False)

    _render_sections(profile)
    _render_completeness(profile, preferences)
    _render_preferences_form(user_id, preferences, demo=demo)
