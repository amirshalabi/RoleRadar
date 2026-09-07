"""
Pipeline metrics tracking.

Will record measured (never invented) pipeline statistics:
jobs ingested, jobs deduplicated, jobs eliminated by hard filters,
jobs eliminated by semantic retrieval, jobs reaching the LLM,
estimated LLM calls avoided, serial ingestion time, concurrent
ingestion time, and concurrency speedup.
"""
