from .aircraft_db import get_aircraft_specs


def assess_threat(topk_results):
    """
    topk_results: list of dicts from EfficientNetTopK
    """
    enriched = []

    for item in topk_results:
        specs = get_aircraft_specs(item["aircraft"])

        enriched.append({
            "aircraft": item["aircraft"],
            "confidence": item["confidence"],
            "specs": specs
        })

    return enriched
