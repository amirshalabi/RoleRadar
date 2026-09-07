"""
Concurrent ingestion.

Will run I/O-heavy ingestion tasks (fetching/parsing job postings) using
concurrent.futures.ThreadPoolExecutor, and record serial vs. concurrent
timing so concurrency speedup can be measured, not estimated.
"""
