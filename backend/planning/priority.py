"""
Prep priority calculation.

Will compute a deterministic prep priority per skill, initially
resembling:

    priority = skill_gap x requirement_importance
               x interview_topic_weight x confidence_adjustment

Priorities are normalized over total available prep hours. No LLM
involvement in this arithmetic.
"""
