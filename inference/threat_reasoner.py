def assess_threat(aircraft_name, confidence, specs):
    """
    Assess threat level based on aircraft specs and classification confidence.
    Returns a stable, fully-populated threat object.
    """

    uncertainty_notes = []

    # ---------------------------
    # HARD FALLBACK
    # ---------------------------
    if not specs or not isinstance(specs, dict):
        return {
            "level": "Unknown",
            "confidence_adjusted_level": "Unknown",
            "rationale": "No technical specifications available.",
            "summary": (
                f"{aircraft_name} detected, but no reliable specifications are "
                "available to assess its threat level."
            ),
            "uncertainty_notes": [
                "Aircraft specifications missing or unavailable",
                "Threat assessment based solely on visual classification",
            ],
        }

    performance = specs.get("performance") or {}
    capabilities = specs.get("capabilities") or {}
    armament = specs.get("armament") or {}

    max_speed = (performance.get("max_speed") or {}).get("kph", 0)
    combat_range = (performance.get("combat_range") or {}).get("km", 0)

    stealth = bool(capabilities.get("stealth", False))
    bvr = bool(capabilities.get("bvr_combat", False))
    strike = bool(capabilities.get("strike", False))
    ew = bool(capabilities.get("electronic_warfare", False))

    air_to_air = armament.get("air_to_air") or []
    air_to_ground = armament.get("air_to_ground") or []

    # ---------------------------
    # UNCERTAINTY FLAGS
    # ---------------------------
    if confidence < 0.30:
        uncertainty_notes.append(
            f"Low classification confidence ({confidence:.1%})"
        )

    if not max_speed:
        uncertainty_notes.append("Maximum speed data unavailable")

    if not armament:
        uncertainty_notes.append("Armament configuration unknown")

    # ---------------------------
    # THREAT SCORING
    # ---------------------------
    threat_score = 0

    if max_speed > 2000:
        threat_score += 2
    elif max_speed > 1000:
        threat_score += 1

    if bvr:
        threat_score += 2
    if stealth:
        threat_score += 2
    if strike:
        threat_score += 1
    if ew:
        threat_score += 1
    if air_to_air:
        threat_score += 1

    if threat_score >= 6:
        level = "High"
    elif threat_score >= 3:
        level = "Medium"
    elif threat_score >= 1:
        level = "Low"
    else:
        level = "Minimal"

    if confidence < 0.15:
        adjusted = "Uncertain"
    elif confidence < 0.30:
        adjusted = f"Possibly {level}"
    else:
        adjusted = level

    rationale_parts = []

    if max_speed:
        rationale_parts.append(f"high speed (~{max_speed} km/h)")
    if bvr:
        rationale_parts.append("beyond-visual-range engagement")
    if stealth:
        rationale_parts.append("reduced observability")
    if strike:
        rationale_parts.append("strike capability")
    if ew:
        rationale_parts.append("electronic warfare systems")

    rationale = (
        ", ".join(rationale_parts)
        if rationale_parts
        else "limited available performance and armament data"
    )

    summary = (
        f"{aircraft_name} is assessed as a {adjusted.lower()} threat. "
        f"It demonstrates {rationale}. "
        "This aircraft should be treated as a credible combat platform."
        if level in {"High", "Medium"}
        else
        f"{aircraft_name} poses a limited direct threat based on available data."
    )

    if not uncertainty_notes:
        uncertainty_notes.append("Assessment confidence is high")

    return {
        "level": level,
        "confidence_adjusted_level": adjusted,
        "rationale": rationale,
        "summary": summary,
        "uncertainty_notes": uncertainty_notes,
    }

def assess_threat(aircraft_name, confidence, specs):
    """
    Assess threat level based on aircraft specs and classification confidence.
    Returns a stable, fully-populated threat object.
    """

    uncertainty_notes = []

    # ---------------------------
    # HARD FALLBACK
    # ---------------------------
    if not specs or not isinstance(specs, dict):
        return {
            "level": "Unknown",
            "confidence_adjusted_level": "Unknown",
            "rationale": "No technical specifications available.",
            "summary": (
                f"{aircraft_name} detected, but no reliable specifications are "
                "available to assess its threat level."
            ),
            "uncertainty_notes": [
                "Aircraft specifications missing or unavailable",
                "Threat assessment based solely on visual classification",
            ],
        }

    performance = specs.get("performance") or {}
    capabilities = specs.get("capabilities") or {}
    armament = specs.get("armament") or {}

    max_speed = (performance.get("max_speed") or {}).get("kph", 0)
    combat_range = (performance.get("combat_range") or {}).get("km", 0)

    stealth = bool(capabilities.get("stealth", False))
    bvr = bool(capabilities.get("bvr_combat", False))
    strike = bool(capabilities.get("strike", False))
    ew = bool(capabilities.get("electronic_warfare", False))

    air_to_air = armament.get("air_to_air") or []
    air_to_ground = armament.get("air_to_ground") or []

    # ---------------------------
    # UNCERTAINTY FLAGS
    # ---------------------------
    if confidence < 0.30:
        uncertainty_notes.append(
            f"Low classification confidence ({confidence:.1%})"
        )

    if not max_speed:
        uncertainty_notes.append("Maximum speed data unavailable")

    if not armament:
        uncertainty_notes.append("Armament configuration unknown")

    # ---------------------------
    # THREAT SCORING
    # ---------------------------
    threat_score = 0

    if max_speed > 2000:
        threat_score += 2
    elif max_speed > 1000:
        threat_score += 1

    if bvr:
        threat_score += 2
    if stealth:
        threat_score += 2
    if strike:
        threat_score += 1
    if ew:
        threat_score += 1
    if air_to_air:
        threat_score += 1

    if threat_score >= 6:
        level = "High"
    elif threat_score >= 3:
        level = "Medium"
    elif threat_score >= 1:
        level = "Low"
    else:
        level = "Minimal"

    if confidence < 0.15:
        adjusted = "Uncertain"
    elif confidence < 0.30:
        adjusted = f"Possibly {level}"
    else:
        adjusted = level

    rationale_parts = []

    if max_speed:
        rationale_parts.append(f"high speed (~{max_speed} km/h)")
    if bvr:
        rationale_parts.append("beyond-visual-range engagement")
    if stealth:
        rationale_parts.append("reduced observability")
    if strike:
        rationale_parts.append("strike capability")
    if ew:
        rationale_parts.append("electronic warfare systems")

    rationale = (
        ", ".join(rationale_parts)
        if rationale_parts
        else "limited available performance and armament data"
    )

    summary = (
        f"{aircraft_name} is assessed as a {adjusted.lower()} threat. "
        f"It demonstrates {rationale}. "
        "This aircraft should be treated as a credible combat platform."
        if level in {"High", "Medium"}
        else
        f"{aircraft_name} poses a limited direct threat based on available data."
    )

    if not uncertainty_notes:
        uncertainty_notes.append("Assessment confidence is high")

    return {
        "level": level,
        "confidence_adjusted_level": adjusted,
        "rationale": rationale,
        "summary": summary,
        "uncertainty_notes": uncertainty_notes,
    }
