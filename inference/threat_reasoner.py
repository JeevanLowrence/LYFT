# inference/threat_reasoner.py

def assess_threat(aircraft_name, confidence, specs):
    """
    Returns structured threat assessment based on aircraft specs and confidence.
    """

    capabilities = specs.get("capabilities", {})
    performance = specs.get("performance", {})
    armament = specs.get("armament", {})

    score = 0
    factors = []

    # --- Role & capability scoring ---
    if capabilities.get("stealth"):
        score += 3
        factors.append("Low observability / stealth capability")

    if capabilities.get("bvr_combat"):
        score += 3
        factors.append("Beyond-visual-range combat capability")

    if capabilities.get("multirole"):
        score += 2
        factors.append("Multirole mission flexibility")

    if capabilities.get("strike"):
        score += 2
        factors.append("Precision strike capability")

    if capabilities.get("electronic_warfare"):
        score += 2
        factors.append("Electronic warfare capability")

    # --- Performance ---
    max_speed = performance.get("max_speed", {}).get("kph", 0)
    if max_speed and max_speed > 1800:
        score += 1
        factors.append("High-speed platform")

    # --- Armament ---
    if armament.get("air_to_air"):
        score += 1
        factors.append("Air-to-air armament")

    if armament.get("air_to_ground"):
        score += 1
        factors.append("Air-to-ground armament")

    # --- Confidence adjustment ---
    confidence_adjusted = confidence >= 0.6
    if not confidence_adjusted:
        factors.append("Reduced confidence in identification")

    # --- Threat level mapping ---
    if score >= 9:
        level = "Severe"
    elif score >= 6:
        level = "High"
    elif score >= 4:
        level = "Moderate"
    elif score >= 2:
        level = "Low"
    else:
        level = "Minimal"

    # --- Human-readable synthesis ---
    summary = (
        f"{aircraft_name} is assessed as a {level.lower()} threat platform. "
        f"It possesses capabilities consistent with {', '.join(factors[:3])}. "
        "Actual threat depends on mission intent, loadout, and proximity."
    )

    return {
        "level": level,
        "confidence_adjusted": confidence_adjusted,
        "score": score,
        "contributing_factors": factors,
        "uncertainty_notes": [
            "Weapon loadout unknown",
            "Mission intent unknown",
            "Proximity and support assets unknown"
        ],
        "summary": summary
    }
