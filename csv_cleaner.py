import csv
import requests
import re
import json

INPUT_CSV = "flights_list.csv"
OUTPUT_CSV = "cleaned_flights.csv"

OLLAMA_MODEL = "llama3.2"

# ---------------------------------------------------------
# Known aircraft manufacturers for auto-recognition
# ---------------------------------------------------------
KNOWN_MANUFACTURERS = [
    "Agusta", "AgustaWestland", "Airbus", "Boeing", "Embraer", "Bombardier",
    "Cessna", "Piper", "Beechcraft", "Antonov", "Ilyushin", "Tupolev",
    "Sukhoi", "Mikoyan", "MiG", "Eurocopter", "Bell", "Sikorsky", "Lockheed",
    "Northrop", "Dassault", "Saab", "Hawker", "Gulfstream", "Pilatus",
    "Robinson", "Kawasaki", "HondaJet", "Harbin", "HAL", "KAI"
]

# Aircraft model pattern: letters+digits (A320, F-35, C-130, AW101)
AIRCRAFT_MODEL_PATTERN = re.compile(r'\b[A-Za-z]{1,4}[- ]?\d{2,4}\b')


# ---------------------------------------------------------
# FIRST CHECK: HARD RULES (no AI)
# ---------------------------------------------------------
def is_definitely_aircraft(name: str):
    name_lower = name.lower()

    # Manufacturer test
    for maker in KNOWN_MANUFACTURERS:
        if maker.lower() in name_lower:
            return True

    # Model pattern test
    if AIRCRAFT_MODEL_PATTERN.search(name):
        return True

    # "Flight ###" should typically be real
    if re.search(r'flight\s*\d+', name_lower):
        return True

    return False


# ---------------------------------------------------------
# Ask Ollama only if needed
# ---------------------------------------------------------
def ollama_check(name: str):
    prompt = f"""
        Classify the following name. Return ONLY a JSON dictionary.

        {{
          "is_flight": true/false,
          "confidence": 0.0-1.0,
          "reason": "short explanation"
        }}

        Consider TRUE if this is:
        - a real aircraft model (military, civilian, helicopter, UAV)
        - an official flight number (e.g., Air France Flight 447)

        Consider FALSE if:
        - it's an accident event
        - a crash description
        - a news story
        - a location
        - anything not an aircraft or flight

        Name: "{name}"
    """

    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=25
        )

        data = r.json()
        text = data.get("response", "")

        # Extract JSON safely
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            return None

        return json.loads(text[start:end+1])

    except Exception as e:
        print(f"[ERROR] Ollama failed: {e}")
        return None


# ---------------------------------------------------------
# Manual user fallback
# ---------------------------------------------------------
def manual_review(name):
    while True:
        user = input(
            f"\nIs '{name}' a real aircraft or flight? (Y=yes / N=no / skip=remove): "
        ).strip().lower()

        if user == "y":
            return name

        elif user == "n":
            new_name = input("Enter the correct aircraft/flight name (or skip): ").strip()
            if new_name.lower() == "skip":
                return None
            if new_name:
                return new_name

        elif user == "skip":
            return None

        else:
            print("Invalid input. Use Y, N, or skip.")


# ---------------------------------------------------------
# Clean each row
# ---------------------------------------------------------
def clean_name(name: str):
    name = name.strip()
    print(f"\n=== Checking: {name} ===")

    # 1) HARD RULES → auto accept
    if is_definitely_aircraft(name):
        print("→ Auto-accepted (matches aircraft patterns).")
        return name

    # 2) Use Ollama only for ambiguous names
    result = ollama_check(name)

    if result:
        is_flight = result.get("is_flight", False)
        conf = float(result.get("confidence", 0))

        print(f"Ollama: is_flight={is_flight}, conf={conf}")

        if is_flight and conf >= 0.70:
            print("→ Accepted automatically by AI.")
            return name

        if not is_flight and conf >= 0.70:
            print("→ Rejected automatically by AI.")
            return manual_review(name)

    print("Ollama uncertain → Asking user.")
    return manual_review(name)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    print("=== CSV Aircraft Name Cleaner ===")

    cleaned = []
    total = 0
    kept = 0
    skipped = 0

    with open(INPUT_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        total += 1
        title = row.get("Title", "").strip()
        if not title:
            continue

        new_name = clean_name(title)

        if new_name:
            row["Title"] = new_name
            cleaned.append(row)
            kept += 1
        else:
            skipped += 1

    if cleaned:
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cleaned[0].keys())
            w.writeheader()
            w.writerows(cleaned)

    print("\n=== Done ===")
    print(f"Total: {total}")
    print(f"Kept: {kept}")
    print(f"Skipped: {skipped}")
    print(f"Saved cleaned CSV → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
