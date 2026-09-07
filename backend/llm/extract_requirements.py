"""
Structured job requirement extraction.

Will use the LLM to extract structured requirements (skills, importance,
seniority signals) from unstructured job description text, with output
validated against Pydantic schemas before it enters the deterministic
matching pipeline.
"""
