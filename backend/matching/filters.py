"""
Deterministic hard filters.

Will eliminate clearly irrelevant roles using rule-based Python logic
(e.g. location/work-authorization constraints, role family mismatch)
BEFORE any embedding or LLM calls are made, to avoid unnecessary cost.
"""
