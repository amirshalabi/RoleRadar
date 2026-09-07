"""
Job ingestion: source adapters and per-source fetch+normalize.

SourceAdapter is the generic interface every job provider must
implement, so a new provider (a real API, an RSS feed, a partner
integration) can be added later by writing one class - nothing in
backend.ingestion.concurrent, backend.ingestion.deduplicate, or
downstream normalization/persistence needs to change.

The three adapters here are deliberately mock/demo sources rather than
scrapers of any real site (fragile, and outside this project's scope).
Each simulates I/O with a fixed time.sleep() - not random - so local
concurrency benchmarking and tests are fully reproducible while still
genuinely spending wall-clock time the way a real HTTP call would.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from backend.ingestion.normalize import Role, normalize_role

logger = logging.getLogger(__name__)


class SourceAdapter(ABC):
    """
    Common interface every job source must implement. fetch_raw_jobs()
    returns RAW, provider-shaped records (inconsistent keys expected,
    e.g. jobTitle vs title) - normalization into a canonical Role
    happens downstream via backend.ingestion.normalize.normalize_role(),
    so an adapter's only job is "get the data," never "clean the data."

    May raise on failure (network error, non-2xx response, etc.) -
    callers (backend.ingestion.concurrent) are responsible for catching
    that and recording it, not this class.
    """

    #: Human-readable identifier used in metrics/logging - set per adapter.
    name: str = "unnamed_source"

    @abstractmethod
    def fetch_raw_jobs(self) -> list[dict[str, Any]]:
        """Return this source's raw job-like records."""
        raise NotImplementedError


class DemoAPISourceAdapter(SourceAdapter):
    """Simulates a fast, reliable first-party JSON API (camelCase fields)."""

    name = "demo_api"

    def __init__(self, simulated_latency_seconds: float = 0.3):
        self._simulated_latency_seconds = simulated_latency_seconds

    def fetch_raw_jobs(self) -> list[dict[str, Any]]:
        time.sleep(self._simulated_latency_seconds)
        return [
            {
                "jobTitle": "Software Engineering Intern",
                "companyName": "Acme Corp",
                "jobUrl": "https://acme.example/jobs/1",
                "location": "Remote",
                "jobDescription": "Build backend services in Python.",
            },
            {
                "jobTitle": "Data Science Intern",
                "companyName": "Acme Corp",
                "jobUrl": "https://acme.example/jobs/2",
                "location": "New York, NY",
                "jobDescription": "Work on internal analytics pipelines.",
            },
        ]


class DemoBoardSourceAdapter(SourceAdapter):
    """
    Simulates a slower third-party job board (snake_case fields), with
    one deliberately malformed record (no title or company) to exercise
    per-record malformed-data handling.
    """

    name = "demo_board"

    def __init__(self, simulated_latency_seconds: float = 0.6):
        self._simulated_latency_seconds = simulated_latency_seconds

    def fetch_raw_jobs(self) -> list[dict[str, Any]]:
        time.sleep(self._simulated_latency_seconds)
        return [
            {
                "title": "Quantitative Research Intern",
                "company": "Meridian Capital",
                "url": "https://meridian.example/jobs/3",
                "location": "Chicago, IL",
                "description": "Build and backtest trading signals.",
            },
            {"description": "This record is missing a title and company."},  # malformed on purpose
        ]


class DemoUnreliableSourceAdapter(SourceAdapter):
    """
    Simulates a source that is either down (raises) or slow enough to
    exceed a caller's timeout - deterministic and configurable rather
    than randomized, so both failure modes are reproducible for tests
    and local benchmarking. Set `fail=True` for a hard failure, or leave
    `fail=False` with a `simulated_latency_seconds` above your ingestion
    call's `per_source_timeout` to exercise timeout handling instead.
    """

    name = "demo_unreliable"

    def __init__(self, simulated_latency_seconds: float = 0.1, fail: bool = True):
        self._simulated_latency_seconds = simulated_latency_seconds
        self._fail = fail

    def fetch_raw_jobs(self) -> list[dict[str, Any]]:
        time.sleep(self._simulated_latency_seconds)
        if self._fail:
            raise RuntimeError("Simulated upstream failure from demo_unreliable source.")
        return [
            {
                "title": "Backend Engineer Intern",
                "company": "QuietCo",
                "location": "Remote",
                "description": "General backend work.",
            }
        ]


class MalformedRecordError(BaseModel):
    """One raw record from a source that failed to normalize into a Role."""

    source_name: str
    raw_record: dict[str, Any]
    error: str


def fetch_and_normalize_source(adapter: SourceAdapter) -> tuple[list[Role], list[MalformedRecordError]]:
    """
    Fetch raw jobs from `adapter` and normalize each into a Role. A
    single malformed record (missing title/company - see
    backend.ingestion.normalize.normalize_role) is recorded and skipped
    rather than failing the whole source's other valid records.

    Does NOT catch a whole-source failure (the fetch call itself
    raising) - that is handled one layer up, in
    backend.ingestion.concurrent, so partial per-record failure and
    total source failure are handled at the layer that actually needs
    to react to each.
    """
    raw_jobs = adapter.fetch_raw_jobs()
    roles: list[Role] = []
    malformed: list[MalformedRecordError] = []
    for raw_job in raw_jobs:
        try:
            roles.append(normalize_role(raw_job))
        except ValueError as exc:
            malformed.append(MalformedRecordError(source_name=adapter.name, raw_record=raw_job, error=str(exc)))
    return roles, malformed
