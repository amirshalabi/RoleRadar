"""
Cross-role skill-gap analysis and skill ROI.

Will analyze skill gaps across a candidate's saved/favorite roles and
compute a deterministic, explainable skill_roi:

    skill_roi = average_gap x frequency_across_saved_roles
                x average_requirement_importance x user_role_priority
                / estimated_learning_cost

used to identify which skill would most improve the candidate's fit
across their saved opportunities.
"""
