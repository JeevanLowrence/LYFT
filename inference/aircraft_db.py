import os
import yaml

SPEC_ROOT = "aircraft_specs"


def get_aircraft_specs(aircraft_name: str):
    """
    Loads specs.yaml for a given aircraft.
    """
    spec_path = os.path.join(
        SPEC_ROOT,
        aircraft_name,
        "specs.yaml"
    )

    if not os.path.exists(spec_path):
        return {
            "error": f"No specs found for {aircraft_name}"
        }

    with open(spec_path, "r", encoding="utf-8") as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            return {
                "error": f"YAML parse error for {aircraft_name}",
                "details": str(e)
            }
