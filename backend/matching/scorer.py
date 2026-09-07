"""
Deterministic fit scoring.

Will compute the overall candidate-job fit score as a weighted sum of
component scores: technical, experience, coursework, domain, interest,
and constraints. Default weights (technical 0.35, experience 0.20,
coursework 0.10, domain 0.15, interest 0.10, constraints 0.10) will be
configurable per role family. This module performs no LLM calls - the
LLM only explains a score already computed here.
"""
