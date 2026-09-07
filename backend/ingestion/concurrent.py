"""
Concurrent ingestion.

WHY concurrent.futures.ThreadPoolExecutor (not multiprocessing):
fetching job postings is I/O-bound - each call spends nearly all its
wall-clock time waiting on a network response (or, for the demo
adapters in backend.ingestion.jobs, a simulated time.sleep()), not
doing CPU work. Python's GIL is released during I/O waits, so multiple
threads can be blocked-on-I/O at the same time and the OS/network make
real concurrent progress across them - a thread pool gets genuine
wall-clock speedup here at low overhead (thread creation is cheap).
Multiprocessing would add process-startup cost and IPC/pickling
overhead for zero benefit, since there is no CPU-bound work (parsing a
few small JSON records is negligible) to actually parallelize across
cores - it would very likely be SLOWER than threads for this exact
workload, not faster.

WHY serial_ingest() IS ALSO A ThreadPoolExecutor (with max_workers=1)
rather than a plain for-loop: so the ONLY variable between the serial
and concurrent code paths is the worker count. Both go through the
exact same per-source execution, error handling, and timeout machinery
(_run_sources() below) - this makes the measured speedup a clean,
apples-to-apples comparison of concurrency itself, not a comparison
confounded by two different implementations. Every runtime and speedup
number this module returns comes from an actual time.perf_counter()
measurement of real execution - never invented or hardcoded (see
IngestionRunResult.speedup).
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Literal

from pydantic import BaseModel, Field

from backend.db.client import SupabaseNotConfiguredError
from backend.db.roles import upsert_role
from backend.ingestion.deduplicate import deduplicate_roles
from backend.ingestion.jobs import MalformedRecordError, SourceAdapter, fetch_and_normalize_source
from backend.ingestion.normalize import Role

logger = logging.getLogger(__name__)

DEFAULT_PER_SOURCE_TIMEOUT_SECONDS = 5.0

SourceStatus = Literal["success", "failed", "timeout"]


class SourceRunOutcome(BaseModel):
    """The outcome of running exactly one source through fetch + normalize."""

    source_name: str
    status: SourceStatus
    roles: list[Role] = Field(default_factory=list)
    malformed_records: list[MalformedRecordError] = Field(default_factory=list)
    error: str | None = None
    elapsed_seconds: float


class IngestionFetchResult(BaseModel):
    """Result of fetching + normalizing every source (before dedup/upsert)."""

    source_outcomes: list[SourceRunOutcome]
    roles: list[Role] = Field(description="Merged across every successful source, not yet deduplicated.")
    elapsed_seconds: float = Field(description="Real measured wall-clock time for this run.")
    total_records_fetched: int
    malformed_record_count: int
    failed_source_count: int
    timed_out_source_count: int


def _run_one_source(adapter: SourceAdapter) -> SourceRunOutcome:
    """
    Fetch + normalize one source, capturing a whole-source failure (the
    adapter's fetch_raw_jobs() raising) as part of the returned outcome
    rather than letting it propagate - used by both serial and
    concurrent execution so failure handling is identical in both paths.
    """
    start = time.perf_counter()
    try:
        roles, malformed = fetch_and_normalize_source(adapter)
        elapsed = time.perf_counter() - start
        return SourceRunOutcome(
            source_name=adapter.name, status="success", roles=roles, malformed_records=malformed,
            elapsed_seconds=round(elapsed, 4),
        )
    except Exception as exc:
        elapsed = time.perf_counter() - start
        logger.warning("Source %r failed during ingestion", adapter.name, exc_info=True)
        return SourceRunOutcome(source_name=adapter.name, status="failed", error=str(exc), elapsed_seconds=round(elapsed, 4))


def _run_sources(
    adapters: list[SourceAdapter], max_workers: int, per_source_timeout: float | None
) -> tuple[list[SourceRunOutcome], float]:
    """
    Run every adapter through a ThreadPoolExecutor with `max_workers`
    workers, returning each source's outcome plus the real measured
    total wall-clock time. See this module's docstring for why
    serial_ingest()/concurrent_ingest() are just this function called
    with different worker counts.

    Honesty note on timeouts: a source that exceeds `per_source_timeout`
    is correctly REPORTED as status="timeout" (see the per-future
    result() loop below), but Python cannot forcibly kill a running
    thread - the `with ThreadPoolExecutor(...)` block still waits for
    that thread to actually finish before this function can return (its
    `__exit__` calls shutdown(wait=True)). So a hung/very slow source
    can make the overall returned `elapsed_seconds` longer than
    `per_source_timeout`, even though that source is marked as timed
    out. This is reported as-measured rather than clipped to look
    better - see this module's "do not fake speedup" policy.
    """
    start = time.perf_counter()
    outcomes: list[SourceRunOutcome] = []

    # Iterate in SUBMISSION order (not as_completed()) and deliberately:
    # as_completed() already blocks until a future is done before
    # yielding it, so by the time you'd call future.result(timeout=X)
    # on what it hands you, the future has already finished and the
    # timeout is a no-op. Calling .result(timeout=X) directly on each
    # future - in any fixed order - correctly waits up to X seconds on
    # a future that ISN'T done yet and raises on schedule; a future
    # that already finished (common with multiple workers, since they
    # all run concurrently in the background regardless of check order)
    # just returns immediately. Total loop time is still bounded by
    # each source's own timeout, not their sum.
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_adapter = {executor.submit(_run_one_source, adapter): adapter for adapter in adapters}
        for future, adapter in future_to_adapter.items():
            try:
                outcomes.append(future.result(timeout=per_source_timeout))
            except FuturesTimeoutError:
                outcomes.append(
                    SourceRunOutcome(
                        source_name=adapter.name,
                        status="timeout",
                        error=f"Source exceeded the {per_source_timeout}s per-source timeout.",
                        elapsed_seconds=per_source_timeout or 0.0,
                    )
                )

    elapsed_seconds = round(time.perf_counter() - start, 4)
    return outcomes, elapsed_seconds


def _build_fetch_result(outcomes: list[SourceRunOutcome], elapsed_seconds: float) -> IngestionFetchResult:
    roles: list[Role] = []
    malformed_count = 0
    failed_count = 0
    timeout_count = 0
    for outcome in outcomes:
        roles.extend(outcome.roles)
        malformed_count += len(outcome.malformed_records)
        if outcome.status == "failed":
            failed_count += 1
        elif outcome.status == "timeout":
            timeout_count += 1

    return IngestionFetchResult(
        source_outcomes=outcomes,
        roles=roles,
        elapsed_seconds=elapsed_seconds,
        total_records_fetched=len(roles),
        malformed_record_count=malformed_count,
        failed_source_count=failed_count,
        timed_out_source_count=timeout_count,
    )


def serial_ingest(
    adapters: list[SourceAdapter], per_source_timeout: float | None = DEFAULT_PER_SOURCE_TIMEOUT_SECONDS
) -> IngestionFetchResult:
    """Fetch and normalize every adapter's jobs one at a time (ThreadPoolExecutor(max_workers=1) - see module docstring)."""
    outcomes, elapsed_seconds = _run_sources(adapters, max_workers=1, per_source_timeout=per_source_timeout)
    return _build_fetch_result(outcomes, elapsed_seconds)


def concurrent_ingest(
    adapters: list[SourceAdapter],
    max_workers: int | None = None,
    per_source_timeout: float | None = DEFAULT_PER_SOURCE_TIMEOUT_SECONDS,
) -> IngestionFetchResult:
    """Fetch and normalize every adapter's jobs concurrently (ThreadPoolExecutor - see module docstring for why)."""
    workers = max_workers or max(len(adapters), 1)
    outcomes, elapsed_seconds = _run_sources(adapters, max_workers=workers, per_source_timeout=per_source_timeout)
    return _build_fetch_result(outcomes, elapsed_seconds)


class IngestionRunResult(BaseModel):
    """Everything recorded for one full ingestion run: per-source counts, totals, errors, and a real benchmark."""

    source_counts: dict[str, int] = Field(description="source_name -> roles successfully fetched from it, pre-dedup.")
    total_records_fetched: int
    duplicates_removed: int
    malformed_record_count: int
    failed_sources: list[str]
    timed_out_sources: list[str]
    unique_roles: list[Role]
    upserted_count: int
    upsert_skipped_reason: str | None = None
    serial_elapsed_seconds: float
    concurrent_elapsed_seconds: float
    speedup: float = Field(description="serial_elapsed_seconds / concurrent_elapsed_seconds, measured - never invented.")


def run_ingestion_pipeline(
    adapters: list[SourceAdapter],
    *,
    max_workers: int | None = None,
    per_source_timeout: float | None = DEFAULT_PER_SOURCE_TIMEOUT_SECONDS,
    upsert: bool = True,
) -> IngestionRunResult:
    """
    The full pipeline: benchmark serial vs. concurrent fetch (both
    genuinely executed), merge + deduplicate the CONCURRENT run's roles
    (the one actually used for persistence - serial_ingest() exists only
    for the speedup comparison), and idempotently upsert into Postgres.

    Every source is fetched TWICE (once per benchmark run) - fine for
    these cheap demo/simulated sources, but a real deployment
    benchmarking real APIs would want to benchmark on a schedule, not
    on every pipeline run.

    Gracefully skips persistence (upserted_count=0,
    upsert_skipped_reason set) if Postgres isn't configured, rather than
    raising - matches how the rest of this project treats missing
    credentials (see backend.db.client.SupabaseNotConfiguredError).
    """
    serial_result = serial_ingest(adapters, per_source_timeout=per_source_timeout)
    concurrent_result = concurrent_ingest(adapters, max_workers=max_workers, per_source_timeout=per_source_timeout)

    dedup_result = deduplicate_roles(concurrent_result.roles)

    source_counts = {outcome.source_name: len(outcome.roles) for outcome in concurrent_result.source_outcomes}
    failed_sources = [o.source_name for o in concurrent_result.source_outcomes if o.status == "failed"]
    timed_out_sources = [o.source_name for o in concurrent_result.source_outcomes if o.status == "timeout"]

    upserted_count = 0
    upsert_skipped_reason: str | None = None
    if upsert:
        try:
            for role in dedup_result.unique_roles:
                upsert_role(
                    company=role.company,
                    title=role.title,
                    location=role.location,
                    url=role.url,
                    description=role.description,
                    role_family=role.role_family,
                    external_id=role.external_id,
                    raw_source=role.raw_source,
                )
                upserted_count += 1
        except SupabaseNotConfiguredError as exc:
            upsert_skipped_reason = str(exc)
    else:
        upsert_skipped_reason = "upsert=False; persistence skipped."

    speedup = (
        round(serial_result.elapsed_seconds / concurrent_result.elapsed_seconds, 3)
        if concurrent_result.elapsed_seconds > 0
        else 0.0
    )

    return IngestionRunResult(
        source_counts=source_counts,
        total_records_fetched=concurrent_result.total_records_fetched,
        duplicates_removed=dedup_result.duplicate_count,
        malformed_record_count=concurrent_result.malformed_record_count,
        failed_sources=failed_sources,
        timed_out_sources=timed_out_sources,
        unique_roles=dedup_result.unique_roles,
        upserted_count=upserted_count,
        upsert_skipped_reason=upsert_skipped_reason,
        serial_elapsed_seconds=serial_result.elapsed_seconds,
        concurrent_elapsed_seconds=concurrent_result.elapsed_seconds,
        speedup=speedup,
    )
