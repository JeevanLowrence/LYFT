# inference/action_recommender.py

def recommend_actions(threat_level, confidence):
    """
    Returns recommended actions based on threat level and confidence.
    """

    actions = {
        "immediate": [],
        "short_term": [],
        "contingency": []
    }

    # --- Baseline monitoring ---
    actions["immediate"].append("Maintain continuous tracking")
    actions["immediate"].append("Cross-check with radar and IFF")

    if threat_level in ["Moderate", "High", "Severe"]:
        actions["short_term"].append("Increase sensor fusion priority")
        actions["short_term"].append("Assign interceptor-ready assets")

    if threat_level in ["High", "Severe"]:
        actions["short_term"].append("Elevate local alert posture")
        actions["contingency"].append("Prepare escalation protocols")

    if threat_level == "Severe":
        actions["contingency"].append("Notify command authority")
        actions["contingency"].append("Review rules of engagement")

    # --- Confidence modifier ---
    if confidence < 0.5:
        actions["immediate"].append("Revalidate classification with additional sensors")

    return actions
