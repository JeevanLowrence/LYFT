import os
import re
import yaml
import time
import logging
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
from subprocess import run, PIPE

# =========================
# CONFIG
# =========================
IMAGE_DATASET_DIR = "aircraft_dataset_final/train"
SPECS_DIR = "aircraft_specs"
OLLAMA_MODEL = "llama3"
WIKI_BASE = "https://en.wikipedia.org/wiki/"

HEADERS = {
    "User-Agent": "AircraftSpecExtractor/1.0"
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# =========================
# HELPERS
# =========================
def safe_name(name: str) -> str:
    return name.replace("/", "_").strip()

def extract_yaml_block(text: str) -> str:
    """
    Extracts the first valid YAML block from model output.
    """
    # Remove common LLM junk
    text = re.sub(r"^.*?(\n|$)", "", text, count=1)

    # Keep only from first valid key
    match = re.search(r"(aircraft:\n.*)", text, re.DOTALL)
    if not match:
        return ""

    return match.group(1)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def is_valid_yaml(text: str) -> bool:
    try:
        obj = yaml.safe_load(text)
        return isinstance(obj, dict)
    except Exception:
        return False

def normalize_speed(text):
    if not text:
        return None
    mach = re.search(r"Mach\s*([\d.]+)", text)
    if mach:
        return {
            "raw": text,
            "kph": round(float(mach.group(1)) * 1225)
        }
    kph = re.search(r"([\d,]+)\s*km/h", text)
    if kph:
        return {
            "raw": text,
            "kph": int(kph.group(1).replace(",", ""))
        }
    return {"raw": text}

def normalize_range(text):
    if not text:
        return None
    km = re.search(r"([\d,]+)\s*km", text)
    if km:
        return {
            "raw": text,
            "km": int(km.group(1).replace(",", ""))
        }
    return {"raw": text}

# =========================
# WIKIPEDIA SCRAPER
# =========================
def scrape_wikipedia(aircraft):
    url = WIKI_BASE + aircraft.replace(" ", "_")
    r = requests.get(url, headers=HEADERS, timeout=10)

    if r.status_code != 200:
        logging.warning(f"Wikipedia page not found for {aircraft}")
        return None

    soup = BeautifulSoup(r.text, "html.parser")

    data = {}

    # Summary paragraph
    p = soup.select_one("p")
    data["summary"] = p.get_text(strip=True) if p else ""

    # Infobox
    infobox = soup.select_one(".infobox")
    info = {}
    if infobox:
        for row in infobox.select("tr"):
            th = row.select_one("th")
            td = row.select_one("td")
            if th and td:
                key = th.get_text(strip=True).lower()
                val = td.get_text(" ", strip=True)
                info[key] = val
    data["infobox"] = info

    # Selected sections
    sections = {}
    for header in soup.select("h2, h3"):
        title = header.get_text().replace("[edit]", "").strip().lower()
        if any(k in title for k in ["design", "development", "armament", "avionics", "variants"]):
            content = []
            for sib in header.find_next_siblings():
                if sib.name and sib.name.startswith("h"):
                    break
                if sib.name == "p":
                    content.append(sib.get_text(strip=True))
            sections[title] = " ".join(content)

    data["sections"] = sections
    return data

# =========================
# OLLAMA PROMPT
# =========================
def ask_ollama(spec_text):
    prompt = f"""
You are a military aviation analyst.

From the provided information, produce STRICT YAML ONLY with this schema:

aircraft:
  name:
  role:
  country:
  introduction_year:

performance:
  max_speed:
    raw:
    kph:
  combat_range:
    raw:
    km:
  service_ceiling_m:

capabilities:
  stealth: true/false
  bvr_combat: true/false
  multirole: true/false
  strike: true/false
  electronic_warfare: true/false

sensors:
  radar:
    present: true/false
    type:
  eo_ir: true/false
  datalink: true/false

armament:
  air_to_air:
    - name
  air_to_ground:
    - name
  internal_bay: true/false

threat_assessment:
  level: Low/Medium/High
  rationale:

summary:
  short_capability_summary:

Use null if unknown. Do NOT include explanations.
SOURCE:
{spec_text}
"""
    proc = run(
        ["ollama", "run", OLLAMA_MODEL],
        input=prompt.encode(),
        stdout=PIPE,
        stderr=PIPE
    )
    return proc.stdout.decode()

# =========================
# MAIN EXTRACTION
# =========================
def extract_all():
    ensure_dir(SPECS_DIR)

    aircrafts = sorted([
        d for d in os.listdir(IMAGE_DATASET_DIR)
        if os.path.isdir(os.path.join(IMAGE_DATASET_DIR, d))
    ])

    logging.info(f"Found {len(aircrafts)} aircraft")

    for aircraft in tqdm(aircrafts, desc="Extracting specs"):
        safe = safe_name(aircraft)
        out_dir = os.path.join(SPECS_DIR, safe)
        ensure_dir(out_dir)
        out_file = os.path.join(out_dir, "specs.yaml")

        if os.path.exists(out_file):
            continue

        wiki = scrape_wikipedia(aircraft)
        if not wiki:
            continue

        spec_text = f"""
SUMMARY:
{wiki['summary']}

INFOBOX:
{wiki['infobox']}

SECTIONS:
{wiki['sections']}
"""

        raw = ask_ollama(spec_text)
        yaml_text = extract_yaml_block(raw)

        if not is_valid_yaml(yaml_text):
            logging.warning(f"Invalid YAML for {aircraft}, skipping")
            continue

        spec = yaml.safe_load(yaml_text)

        # Numeric normalization
        if "performance" in spec:
            perf = spec["performance"]
            perf["max_speed"] = normalize_speed(perf.get("max_speed", {}).get("raw"))
            perf["combat_range"] = normalize_range(perf.get("combat_range", {}).get("raw"))

        with open(out_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(spec, f, sort_keys=False, allow_unicode=True)

        time.sleep(1)

    logging.info("✔ Spec extraction complete")

# =========================
# ENTRYPOINT
# =========================
if __name__ == "__main__":
    extract_all()
