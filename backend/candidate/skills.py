"""
Candidate skill tracking.

Will maintain per-user normalized skill estimates, each with an
estimated proficiency score and a confidence value. Resume evidence
provides moderate confidence; diagnostic/assessment results can later
increase it. Backed by the candidate_skills table
(UNIQUE(user_id, normalized_skill_name)).
"""
