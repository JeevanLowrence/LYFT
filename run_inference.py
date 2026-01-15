# run_inference.py

import sys
from inference.efficientnet_top_k import EfficientNetTopK
from inference.aircraft_db import get_aircraft_specs
from inference.threat_reasoner import assess_threat
from inference.action_recommender import recommend_actions


def main(image_path):
    model = EfficientNetTopK(
        model_path="efficientnet_aircraft_BEST.pth",
        class_list_path="class_names.txt"
    )

    topk = model.predict_topk(image_path, k=3)

    for rank, result in enumerate(topk, 1):
        # --- Robust key handling ---
        aircraft = (
            result.get("aircraft")      # ← PRIMARY (your model)
            or result.get("class")
            or result.get("label")
            or result.get("class_name")
        )

        confidence = (
            result.get("confidence")    # ← PRIMARY
            or result.get("score")
            or result.get("probability")
        )

        if aircraft is None or confidence is None:
            print(f"[WARNING] Invalid Top-K result format: {result}")
            continue


        print("\n" + "=" * 60)
        print(f"TOP-{rank} IDENTIFICATION")
        print(f"Aircraft: {aircraft}")
        print(f"Confidence: {confidence:.2%}")

        specs = get_aircraft_specs(aircraft)
        if not specs:
            print("No specifications available.")
            continue

        threat = assess_threat(aircraft, confidence, specs)
        actions = recommend_actions(threat["level"], confidence)

        print("\nThreat Assessment")
        print(f"Level: {threat['level']}")
        print(threat["summary"])

        print("\nRecommended Actions")
        for phase, acts in actions.items():
            print(f"\n{phase.upper()}:")
            for a in acts:
                print(f"- {a}")

        print("\nUncertainty Notes:")
        for note in threat["uncertainty_notes"]:
            print(f"- {note}")


    print("\n" + "=" * 60)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_inference.py <image_path>")
        sys.exit(1)

    main(sys.argv[1])
