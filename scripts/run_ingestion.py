#!/usr/bin/env python3
"""
Fetch real job postings from Adzuna and upsert them into the roles
table via the existing ingestion pipeline
(backend.ingestion.concurrent.run_ingestion_pipeline) - the same
fetch -> dedupe -> upsert machinery already covered by
tests/test_concurrent_ingestion.py, just pointed at a real source
instead of the Demo* adapters.

Note: run_ingestion_pipeline benchmarks serial vs. concurrent fetch by
running every adapter TWICE (see that function's docstring), so each
invocation of this script makes two real calls to the Adzuna API.

Usage:
    python scripts/run_ingestion.py
    python scripts/run_ingestion.py --what "backend engineer intern" --country gb
    python scripts/run_ingestion.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.ingestion.concurrent import run_ingestion_pipeline  # noqa: E402
from backend.ingestion.jobs import AdzunaNotConfiguredError, AdzunaSourceAdapter  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest real job postings from Adzuna into RoleRadar.")
    parser.add_argument("--what", default="software engineer intern", help="Adzuna search query (default: %(default)s)")
    parser.add_argument("--country", default="us", help="Adzuna country code, e.g. us, gb (default: %(default)s)")
    parser.add_argument("--page", type=int, default=1, help="Adzuna result page to fetch (default: %(default)s)")
    parser.add_argument("--results-per-page", type=int, default=20, help="Results per page (default: %(default)s)")
    parser.add_argument("--dry-run", action="store_true", help="Fetch and dedupe but do not write to the database.")
    args = parser.parse_args()

    adapter = AdzunaSourceAdapter(
        country=args.country,
        what=args.what,
        results_per_page=args.results_per_page,
        page=args.page,
    )

    try:
        result = run_ingestion_pipeline([adapter], upsert=not args.dry_run)
    except AdzunaNotConfiguredError as exc:
        print(f"❌ {exc}")
        return 1

    print(f"Fetched {result.total_records_fetched} raw posting(s) from Adzuna.")
    print(f"Duplicates removed: {result.duplicates_removed}")
    print(f"Malformed records skipped: {result.malformed_record_count}")
    if result.failed_sources:
        print(f"Failed sources: {result.failed_sources}")
    if result.timed_out_sources:
        print(f"Timed-out sources: {result.timed_out_sources}")
    print(f"Unique roles after dedup: {len(result.unique_roles)}")

    if args.dry_run:
        print("Dry run - nothing written to the database.")
    elif result.upsert_skipped_reason:
        print(f"Upsert skipped: {result.upsert_skipped_reason}")
    else:
        print(f"Upserted {result.upserted_count} role(s) into the database.")

    print(
        f"serial={result.serial_elapsed_seconds}s "
        f"concurrent={result.concurrent_elapsed_seconds}s "
        f"speedup={result.speedup}x"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
