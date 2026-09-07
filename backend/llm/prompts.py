"""
Prompt templates.

Centralizes prompt wording for LLM extraction and rationale-generation
tasks so it is versioned in one place. Prompts describe the task and
the evidence standard the model must follow; the actual schema is
enforced by Pydantic response models, not by prompt wording alone.
"""

from typing import Any

CANDIDATE_EXTRACTION_SYSTEM_PROMPT = """\
You are extracting structured information from a college student's resume \
for a technical job-matching system. Use ONLY information stated or \
strongly implied by the resume text you are given. Never invent skills, \
projects, employers, coursework, or experience that are not present in \
the text.

For every entry in `skills`:
- `estimated_level` (0-10) is your best estimate of the candidate's \
proficiency, based solely on the evidence in the resume.
- `confidence` (0-1) reflects how strong that evidence is:
  - A skill only listed by name in a skills/tools list (no supporting \
project or experience description) deserves LOW-to-MODERATE confidence, \
roughly 0.2-0.5.
  - A skill backed by a detailed project, internship, or research \
description deserves HIGHER confidence, roughly 0.6-0.9.
  - Never use 1.0 confidence - resume evidence alone is never absolute \
proof of proficiency.
- `evidence_snippets` must be short, near-exact quotes copied from the \
resume text that support the estimate. Do not fabricate a snippet that \
does not appear in the resume. If you cannot find supporting text beyond \
the skill's name itself, quote that listing and keep confidence low.

If a section of the schema (e.g. research, internships) has no \
corresponding content in the resume, return an empty list for it rather \
than inventing an entry.
"""


def build_candidate_extraction_user_prompt(resume_text: str) -> str:
    """Build the user-turn prompt containing the resume text to extract from."""
    return (
        "Extract a structured candidate profile from the following resume "
        "text. Follow the response schema exactly.\n\n"
        "--- RESUME TEXT START ---\n"
        f"{resume_text}\n"
        "--- RESUME TEXT END ---"
    )


ROLE_REQUIREMENT_EXTRACTION_SYSTEM_PROMPT = """\
You are extracting structured skill and knowledge requirements from a job \
description for a technical job-matching system (SWE, quant, AI/ML, and \
research roles). Use ONLY requirements stated or strongly implied by the \
job description text you are given. Never invent a requirement that has \
no textual support - if the description doesn't mention it, leave it out.

Requirements can be concrete technical skills (e.g. "C++", "Python", \
"machine learning") or broader knowledge areas explicitly referenced in \
the text (e.g. "probability", "statistics", "algorithms", "systems", \
"market knowledge"). Only include a knowledge area if the description \
actually references it, directly or through a close paraphrase - do not \
add generic requirements just because they are common for similar roles.

For every requirement:
- `skill`: the requirement's display name, written the way a person would \
read it (e.g. "C++", "Probability & Statistics").
- `normalized_skill`: a lowercase, whitespace-collapsed identity for the \
same requirement (e.g. "c++", "probability and statistics").
- `target_level` (0-10): how much depth/proficiency the posting implies is \
needed. A passing mention ("familiarity with X") implies a lower target \
level than language demanding deep expertise ("expert-level X", "5+ years \
of X").
- `importance` (0-10): how central this requirement is to the role, based \
on emphasis and placement in the text - listed under a "requirements" \
heading or repeated implies higher importance than a single passing \
mention under "nice to have".
- `required` (bool): true if the text frames it as required/must-have, \
false if it is framed as preferred/nice-to-have/a plus.
- `evidence`: one or more short, near-exact quotes copied from the job \
description that support this requirement. Do not fabricate a quote that \
does not appear in the text.

If the job description does not clearly support any requirements, return \
an empty list rather than guessing.
"""


def build_role_requirement_extraction_user_prompt(title: str, company: str, description: str) -> str:
    """Build the user-turn prompt containing the job posting to extract requirements from."""
    return (
        "Extract structured requirements for the following job posting. "
        "Follow the response schema exactly.\n\n"
        f"Company: {company}\n"
        f"Title: {title}\n\n"
        "--- JOB DESCRIPTION START ---\n"
        f"{description}\n"
        "--- JOB DESCRIPTION END ---"
    )


RATIONALE_SYSTEM_PROMPT = """\
You are writing an evidence-grounded explanation of a candidate-job fit \
assessment for a technical job-matching system. The overall and \
component fit scores have ALREADY been computed by deterministic code \
and are given to you only as fixed context - you must not state, imply, \
suggest, or reinterpret a different score, and you must not perform any \
scoring arithmetic yourself. Your only job is to explain in plain \
language what the evidence you are given supports.

For each skill you will be given: the candidate's own resume evidence, \
the role's own requirement evidence (both retrieved from a vector \
database), and already-computed metrics for that skill (target level, \
candidate level, importance, required/preferred, satisfaction ratio, \
confidence). You will also be given general evidence for non-technical \
dimensions (experience, coursework, domain, interest, constraints) and \
a short list of ground-truth candidate profile facts.

Rules:
- Only make claims directly supported by the evidence given to you. \
Never invent a project, skill, employer, course, or requirement that \
does not appear in the evidence or profile facts provided.
- If a skill has no candidate evidence, no role evidence, or both, say \
so explicitly (e.g. "no resume evidence was found for X") and set that \
skill's `insufficient_evidence` to true - never fabricate evidence to \
fill the gap.
- Do not restate or reinterpret any numeric score in your prose - refer \
to fit qualitatively (e.g. "strong alignment", "a significant gap") \
since the numbers are reported separately by the system and are not \
yours to alter.
- Ground each skill's `assessment` and `risk` only in the evidence given \
for that specific skill.
- In each skill entry, set `skill` to EXACTLY the skill identifier given \
in its "Skill identifier" line below (case-sensitive) - it is used to \
programmatically match your narrative back to that skill's computed \
metrics, so it must be copied verbatim.
"""


def build_rationale_user_prompt(
    role_title: str,
    role_company: str,
    overall_score: float,
    component_scores: dict[str, float],
    profile_facts: dict[str, list[str]],
    skill_contexts: list[dict[str, Any]],
    dimension_evidence: dict[str, list[str]],
) -> str:
    """
    Build the user-turn prompt for rationale generation from
    already-retrieved evidence and already-computed metrics.

    `skill_contexts`: one dict per skill, with keys skill_display,
    normalized_skill, target_level, candidate_level, importance,
    required, satisfaction_ratio, confidence, candidate_evidence
    (list[str]), role_evidence (list[str]).
    `dimension_evidence`: dimension name -> list of general evidence
    snippets for that dimension.
    """
    lines = [
        f"Role: {role_title} at {role_company}",
        f"Overall fit score (already computed - do not restate as a number): {overall_score:.1f}/100",
        "Component scores (already computed): "
        + ", ".join(f"{name}={value:.1f}" for name, value in component_scores.items()),
        "",
        "=== Candidate profile facts (ground truth, not retrieved) ===",
    ]
    for label, values in profile_facts.items():
        lines.append(f"{label}: {', '.join(values) if values else 'none listed'}")

    lines.append("\n=== Per-skill evidence and computed metrics ===")
    for context in skill_contexts:
        lines.append(f"\nSkill: {context['skill_display']}")
        lines.append(f"  Skill identifier (echo back exactly): {context['normalized_skill']}")
        lines.append(
            f"  Computed metrics: target_level={context['target_level']}, "
            f"candidate_level={context['candidate_level']}, importance={context['importance']}, "
            f"required={context['required']}, satisfaction_ratio={context['satisfaction_ratio']:.2f}, "
            f"confidence={context['confidence']:.2f}"
        )
        if context["candidate_evidence"]:
            lines.append("  Candidate evidence:")
            lines.extend(f'    - "{snippet}"' for snippet in context["candidate_evidence"])
        else:
            lines.append("  Candidate evidence: none retrieved.")
        if context["role_evidence"]:
            lines.append("  Role requirement evidence:")
            lines.extend(f'    - "{snippet}"' for snippet in context["role_evidence"])
        else:
            lines.append("  Role requirement evidence: none retrieved.")

    lines.append("\n=== General evidence by score dimension ===")
    for dimension, snippets in dimension_evidence.items():
        lines.append(f"\n{dimension}:")
        if snippets:
            lines.extend(f'  - "{snippet}"' for snippet in snippets)
        else:
            lines.append("  none retrieved.")

    lines.append(
        "\nWrite the overall explanation, strengths, weaknesses, risks, uncertain "
        "areas, recommended actions, and one narrative entry per skill listed "
        "above, following the response schema exactly. Do not produce a list of "
        "missing requirements yourself - that list is computed separately, "
        "directly from the metrics already given to you above."
    )
    return "\n".join(lines)
