"""
Tests for backend.llm.study_tasks (LLM-generated activity breakdowns
inside already-fixed study blocks). No real OpenAI calls are made:
parse_structured is monkeypatched throughout.
"""

from __future__ import annotations

from datetime import date

import pytest

from backend.ingestion.normalize import normalize_role
from backend.llm import study_tasks as study_tasks_module
from backend.llm.client import LLMExtractionError, OpenAINotConfiguredError
from backend.llm.study_tasks import (
    StudyActivity,
    _TaskBreakdown,
    enrich_study_plan,
    generate_task_activities,
)
from backend.llm.extract_requirements import RoleRequirement
from backend.matching.gaps import calculate_skill_gaps
from backend.planning.scheduler import StudyTask, generate_study_plan

ROLE = normalize_role({"title": "Quant Research Intern", "company": "Meridian Capital"})


def _task(skill="probability", display="Probability", minutes=60.0) -> StudyTask:
    return StudyTask(
        normalized_skill=skill,
        display_name=display,
        day_index=0,
        scheduled_date=date(2026, 9, 7),
        allocated_minutes=minutes,
        priority_score=5.0,
        description=f"{display} practice block",
    )


def _requirement(skill="Probability", normalized="probability", target=8.0, importance=9.0, required=True):
    return RoleRequirement(
        skill=skill, normalized_skill=normalized, target_level=target, importance=importance,
        required=required, evidence=["x"],
    )


# ---------------------------------------------------------------------
# generate_task_activities: happy path + rescaling
# ---------------------------------------------------------------------


def test_llm_activities_used_when_minutes_already_sum_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    task = _task(minutes=45.0)
    breakdown = _TaskBreakdown(
        activities=[
            StudyActivity(description="Conditional probability review", minutes=15.0),
            StudyActivity(description="Expected value practice problems", minutes=20.0),
            StudyActivity(description="Review past mistakes", minutes=10.0),
        ]
    )
    monkeypatch.setattr(study_tasks_module, "parse_structured", lambda **kwargs: breakdown)

    activities, source = generate_task_activities(task, ROLE, [])

    assert source == "llm"
    assert [a.minutes for a in activities] == [15.0, 20.0, 10.0]
    assert sum(a.minutes for a in activities) == 45.0


def test_llm_activities_rescaled_when_minutes_sum_incorrectly(monkeypatch: pytest.MonkeyPatch) -> None:
    task = _task(minutes=60.0)
    breakdown = _TaskBreakdown(
        activities=[
            StudyActivity(description="A", minutes=15.0),
            StudyActivity(description="B", minutes=20.0),
            StudyActivity(description="C", minutes=5.0),
        ]  # sums to 40, not 60
    )
    monkeypatch.setattr(study_tasks_module, "parse_structured", lambda **kwargs: breakdown)

    activities, source = generate_task_activities(task, ROLE, [])

    assert source == "llm"
    assert sum(a.minutes for a in activities) == pytest.approx(60.0, abs=0.1)
    # relative proportions preserved: A:B:C was 15:20:5 = 3:4:1
    assert activities[0].minutes == pytest.approx(22.5, abs=0.1)
    assert activities[1].minutes == pytest.approx(30.0, abs=0.1)


def test_llm_activity_descriptions_preserved_through_rescale(monkeypatch: pytest.MonkeyPatch) -> None:
    task = _task(minutes=90.0)
    breakdown = _TaskBreakdown(
        activities=[StudyActivity(description="Custom activity wording", minutes=10.0)]
    )
    monkeypatch.setattr(study_tasks_module, "parse_structured", lambda **kwargs: breakdown)

    [activity], source = generate_task_activities(task, ROLE, [])

    assert activity.description == "Custom activity wording"
    assert activity.minutes == 90.0
    assert source == "llm"


def test_total_never_exceeds_allocated_minutes_regardless_of_llm_proposal(monkeypatch: pytest.MonkeyPatch) -> None:
    task = _task(minutes=30.0)
    # LLM wildly overshoots (proposes 500 minutes total for a 30 minute block)
    breakdown = _TaskBreakdown(
        activities=[StudyActivity(description="A", minutes=300.0), StudyActivity(description="B", minutes=200.0)]
    )
    monkeypatch.setattr(study_tasks_module, "parse_structured", lambda **kwargs: breakdown)

    activities, _ = generate_task_activities(task, ROLE, [])

    assert sum(a.minutes for a in activities) == pytest.approx(30.0, abs=0.1)


# ---------------------------------------------------------------------
# generate_task_activities: grounding
# ---------------------------------------------------------------------


def test_skill_gap_context_included_in_prompt_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    task = _task(skill="probability", minutes=30.0)
    gaps = calculate_skill_gaps([], [_requirement(target=8.0, importance=9.0)])
    captured = {}

    def fake_parse_structured(system_prompt, user_prompt, response_model, model=None):
        captured["user_prompt"] = user_prompt
        return _TaskBreakdown(activities=[StudyActivity(description="x", minutes=30.0)])

    monkeypatch.setattr(study_tasks_module, "parse_structured", fake_parse_structured)

    generate_task_activities(task, ROLE, gaps)

    assert "target proficiency level 8" in captured["user_prompt"]
    assert "Meridian Capital" in captured["user_prompt"]
    assert "Quant Research Intern" in captured["user_prompt"]


def test_no_skill_gap_context_when_no_matching_gap(monkeypatch: pytest.MonkeyPatch) -> None:
    task = _task(skill="behavioral", display="Behavioral", minutes=15.0)
    captured = {}

    def fake_parse_structured(system_prompt, user_prompt, response_model, model=None):
        captured["user_prompt"] = user_prompt
        return _TaskBreakdown(activities=[StudyActivity(description="x", minutes=15.0)])

    monkeypatch.setattr(study_tasks_module, "parse_structured", fake_parse_structured)

    generate_task_activities(task, ROLE, [])  # no skill_gaps at all

    assert "No specific skill-gap data is available" in captured["user_prompt"]


# ---------------------------------------------------------------------
# generate_task_activities: graceful fallback
# ---------------------------------------------------------------------


def test_falls_back_when_llm_call_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    task = _task(minutes=45.0)

    def raise_not_configured(**kwargs):
        raise OpenAINotConfiguredError("no key")

    monkeypatch.setattr(study_tasks_module, "parse_structured", raise_not_configured)

    activities, source = generate_task_activities(task, ROLE, [])

    assert source == "fallback_generic"
    assert len(activities) == 1
    assert activities[0].minutes == 45.0
    assert activities[0].description == task.description


def test_falls_back_when_llm_returns_no_activities(monkeypatch: pytest.MonkeyPatch) -> None:
    task = _task(minutes=45.0)
    monkeypatch.setattr(study_tasks_module, "parse_structured", lambda **kwargs: _TaskBreakdown(activities=[]))

    activities, source = generate_task_activities(task, ROLE, [])

    assert source == "fallback_generic"
    assert sum(a.minutes for a in activities) == 45.0


def test_falls_back_when_llm_extraction_error_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    task = _task(minutes=20.0)

    def raise_extraction_error(**kwargs):
        raise LLMExtractionError("model refused")

    monkeypatch.setattr(study_tasks_module, "parse_structured", raise_extraction_error)

    activities, source = generate_task_activities(task, ROLE, [])

    assert source == "fallback_generic"
    assert activities[0].minutes == 20.0


def test_falls_back_on_unexpected_exception_type(monkeypatch: pytest.MonkeyPatch) -> None:
    """The broad except is deliberate - ANY failure mode must degrade gracefully, not just the expected ones."""
    task = _task(minutes=20.0)

    def raise_weird_error(**kwargs):
        raise ConnectionError("network blip")

    monkeypatch.setattr(study_tasks_module, "parse_structured", raise_weird_error)

    activities, source = generate_task_activities(task, ROLE, [])

    assert source == "fallback_generic"
    assert sum(a.minutes for a in activities) == 20.0


def test_zero_minute_task_short_circuits_without_calling_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    task = _task(minutes=0.0)
    called = False

    def fake_parse_structured(**kwargs):
        nonlocal called
        called = True
        return _TaskBreakdown(activities=[])

    monkeypatch.setattr(study_tasks_module, "parse_structured", fake_parse_structured)

    activities, source = generate_task_activities(task, ROLE, [])

    assert activities == []
    assert source == "fallback_generic"
    assert called is False


# ---------------------------------------------------------------------
# enrich_study_plan: whole-plan orchestration
# ---------------------------------------------------------------------


def test_enrich_study_plan_preserves_task_topic_date_minutes(monkeypatch: pytest.MonkeyPatch) -> None:
    gaps = calculate_skill_gaps([], [_requirement()])
    plan = generate_study_plan(gaps, interview_date=date(2026, 9, 10), current_date=date(2026, 9, 7), hours_available_per_day=1.0)
    monkeypatch.setattr(
        study_tasks_module,
        "parse_structured",
        lambda **kwargs: _TaskBreakdown(activities=[StudyActivity(description="x", minutes=1.0)]),
    )

    enriched = enrich_study_plan(plan, ROLE, gaps)

    assert len(enriched) == len(plan.tasks)
    for enriched_task, original_task in zip(enriched, plan.tasks):
        assert enriched_task.task == original_task  # scheduler's decisions are untouched
        assert sum(a.minutes for a in enriched_task.activities) == pytest.approx(original_task.allocated_minutes, abs=0.1)


def test_enrich_study_plan_handles_empty_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = generate_study_plan([], interview_date=None, current_date=date(2026, 9, 7), hours_available_per_day=2.0)

    enriched = enrich_study_plan(plan, ROLE, [])

    assert enriched == []


def test_enrich_study_plan_mixed_success_and_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    gaps = calculate_skill_gaps(
        [], [_requirement(skill="Python", normalized="python"), _requirement(skill="C++", normalized="c++")]
    )
    plan = generate_study_plan(gaps, interview_date=date(2026, 9, 14), current_date=date(2026, 9, 7), hours_available_per_day=2.0)
    assert len(plan.tasks) >= 1

    call_count = 0

    def flaky_parse_structured(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _TaskBreakdown(activities=[StudyActivity(description="ok", minutes=1.0)])
        raise LLMExtractionError("simulated failure on later calls")

    monkeypatch.setattr(study_tasks_module, "parse_structured", flaky_parse_structured)

    enriched = enrich_study_plan(plan, ROLE, gaps)

    sources = {e.generation_source for e in enriched}
    assert "llm" in sources or "fallback_generic" in sources  # at minimum, no crash and every task handled
    total_allocated = sum(t.allocated_minutes for t in plan.tasks)
    total_activity_minutes = sum(a.minutes for e in enriched for a in e.activities)
    assert total_activity_minutes == pytest.approx(total_allocated, abs=0.5)
