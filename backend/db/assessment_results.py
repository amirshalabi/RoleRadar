"""
Diagnostic/assessment result persistence.

Append-only, like ingestion_runs: a candidate can retake a diagnostic
for the same topic multiple times, and each attempt is a genuinely new
event worth keeping (backend.planning.readiness combines the full
accumulated history, not just the latest attempt - see
backend.planning.adaptive), so this never upserts over a prior attempt.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from backend.db.client import get_client

ASSESSMENT_RESULTS_TABLE = "assessment_results"


def save_assessment_result(
    user_id: str, topic: str, observed_level: float, confidence: float = 0.9, source: str | None = None
) -> dict[str, Any]:
    """
    Insert one diagnostic attempt. `topic` is stored under
    `normalized_skill_name` (the column this table was originally built
    with) but holds a readiness TOPIC identifier
    (backend.planning.readiness, e.g. "dsa", "probability"), not
    necessarily a raw resume skill name - see
    backend.planning.readiness.DiagnosticResult.
    """
    client = get_client()
    values = {
        "user_id": user_id,
        "normalized_skill_name": topic,
        "observed_level": observed_level,
        "confidence": confidence,
        "confidence_delta": 0,
        "source": source,
        "taken_at": datetime.now(timezone.utc).isoformat(),
    }
    response = client.table(ASSESSMENT_RESULTS_TABLE).insert(values).execute()
    if not response.data:
        raise RuntimeError(f"Insert into {ASSESSMENT_RESULTS_TABLE} returned no data")
    return response.data[0]


def list_assessment_results(user_id: str) -> list[dict[str, Any]]:
    """Every diagnostic attempt for a user, oldest first - the full accumulated history readiness combination needs."""
    client = get_client()
    response = (
        client.table(ASSESSMENT_RESULTS_TABLE).select("*").eq("user_id", user_id).order("taken_at").execute()
    )
    return response.data or []
