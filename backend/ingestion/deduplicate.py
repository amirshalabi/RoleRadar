"""
Job deduplication.

Will deduplicate ingested roles using a stable external_id when
available, or a deterministic hash of normalized company + title +
location + URL otherwise (see backend/utils/hashing.py), so repeated
ingestion never creates duplicate rows.
"""
