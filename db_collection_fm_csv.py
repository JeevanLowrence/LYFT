import os
import re
import csv
import yaml
import json
import time
import requests
import subprocess
from tqdm import tqdm
from bs4 import BeautifulSoup

# -------------------------------------------------------------------
# 🧹 TEXT CLEANING UTILITIES
# -------------------------------------------------------------------
def clean_wiki_text(text: str) -> str:
    junk_patterns = [
        r"Article", r"Talk", r"Read", r"Edit", r"View history", r"What links here",
        r"Related changes", r"Upload file", r"Permanent link", r"Page information",
        r"Cite this page", r"Get shortened URL", r"Download QR code",
        r"Download as PDF", r"Printable version", r"العربية", r"Français", r"中文",
        r"日本語", r"한국어", r"Bahasa", r"Polski", r"Türkçe", r"Español", r"فارسی",
        r"Italiano", r"עברית", r"ไทย", r"Русский"
    ]
    text = re.sub("|".join(junk_patterns), "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[[^\]]*\]", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9.,;:/()%\-\s]", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def sanitize_dict(d):
    if isinstance(d, dict):
        return {k: sanitize_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [sanitize_dict(i) for i in d]
    elif isinstance(d, str):
        v = clean_wiki_text(d)
        if len(v) > 300:
            v = v[:300] + "..."
        return v
    return d

# -------------------------------------------------------------------
# ⚙️ OLLAMA HELPER
# -------------------------------------------------------------------
def query_ollama(prompt, model="mistral:latest", retries=3):
    for attempt in range(retries):
        try:
            result = subprocess.run(
                ["ollama", "run", model],
                input=prompt.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=120
            )
            response = result.stdout.decode("utf-8", errors="ignore")

            match = re.search(r"\{.*\}", response, re.DOTALL)
            if not match:
                continue
            json_text = match.group(0)
            json_text = re.sub(r",\s*}", "}", json_text)
            json_text = re.sub(r",\s*]", "]", json_text)
            data = json.loads(json_text)
            return sanitize_dict(data)
        except Exception as e:
            print(f"⚠️ Ollama query failed (attempt {attempt+1}): {e}")
            time.sleep(2)
    return None

# -------------------------------------------------------------------
# 🛩️ SPEC EXTRACTION
# -------------------------------------------------------------------
def extract_specs_with_ollama(name: str, wiki_url: str, save_dir="airborne_dataset"):
    os.makedirs(save_dir, exist_ok=True)
    aircraft_dir = os.path.join(save_dir, name.replace("/", "-"))
    os.makedirs(aircraft_dir, exist_ok=True)
    specs_path = os.path.join(aircraft_dir, "specs.yaml")

    if os.path.exists(specs_path):
        print(f"⏩ Skipping {name} — specs already exist.")
        return

    print(f"🔎 Processing entry: {name} | {wiki_url}")

    try:
        html = requests.get(wiki_url, timeout=15).text
    except Exception as e:
        print(f"⚠️ Failed to load {wiki_url}: {e}")
        return

    soup = BeautifulSoup(html, "html.parser")
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    text_block = clean_wiki_text(" ".join(paragraphs))

    prompt = f"""
You are an aerospace analyst. Provide concise, factual technical data for the aircraft '{name}'.
Ignore Wikipedia UI elements or unrelated text.

TEXT:
{text_block[:4000]}

Respond ONLY in JSON with fields:
{{
  "type": "",
  "manufacturer": "",
  "role": "",
  "country_of_origin": "",
  "first_flight": "",
  "introduction": "",
  "number_built": "",
  "status": "",
  "crew": "",
  "capacity": "",
  "dimensions": {{
      "length": "", "wingspan": "", "height": "", "rotor_diameter": "", "wing_area": ""
  }},
  "engine_specs": {{
      "type": "", "count": "", "power_or_thrust": ""
  }},
  "performance": {{
      "max_speed": "", "cruise_speed": "", "range": "", "service_ceiling": "", "rate_of_climb": ""
  }},
  "armament": "",
  "military_capabilities": ["", "", ""]
}}
Keep it short and valid JSON only.
"""
    specs = query_ollama(prompt)

    if specs:
        specs = sanitize_dict(specs)
        with open(specs_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(specs, f, sort_keys=False, allow_unicode=True)
        print(f"✅ Saved specs to {specs_path}")
    else:
        print(f"⚠️ Could not extract specs for {name}")

# -------------------------------------------------------------------
# 🖼️ IMAGE COLLECTION
# -------------------------------------------------------------------
def download_images(aircraft_name: str, save_dir="airborne_dataset"):
    folder = os.path.join(save_dir, aircraft_name.replace("/", "-"), "images")
    os.makedirs(folder, exist_ok=True)

    # Skip if already has images
    if len(os.listdir(folder)) >= 5:
        print(f"⏩ Skipping {aircraft_name} — already has sufficient images.")
        return

    search_queries = [
        f"{aircraft_name} aircraft",
        f"{aircraft_name} helicopter",
        f"{aircraft_name} airplane",
        f"{aircraft_name} aviation photo",
        f"{aircraft_name} site:jetphotos.com",
        f"{aircraft_name} site:airliners.net"
    ]

    headers = {"User-Agent": "Mozilla/5.0"}
    downloaded = 0

    for query in search_queries:
        print(f"DEBUG: Searching for images with term: {query}")
        try:
            url = f"https://www.bing.com/images/search?q={requests.utils.quote(query)}"
            html = requests.get(url, headers=headers, timeout=10).text
            soup = BeautifulSoup(html, "html.parser")
            imgs = [img.get("src") or img.get("data-src") for img in soup.find_all("img")]

            for img_url in imgs:
                if not img_url or "logo" in img_url.lower() or "icon" in img_url.lower():
                    continue
                if not img_url.startswith("http"):
                    continue
                if any(x in img_url.lower() for x in ["wikidata", "svg", "commons"]):
                    continue

                try:
                    img_data = requests.get(img_url, timeout=10).content
                    img_path = os.path.join(folder, f"{aircraft_name}_{downloaded+1}.jpg")
                    with open(img_path, "wb") as f:
                        f.write(img_data)
                    downloaded += 1
                    print(f"✅ Downloaded image: {img_path}")
                    if downloaded >= 5:
                        return
                except Exception:
                    continue
        except Exception as e:
            print(f"⚠️ Image search failed for {aircraft_name}: {e}")

    if downloaded == 0:
        print(f"⚠️ Only 0 images found for {aircraft_name} after all stages.")

# -------------------------------------------------------------------
# 🧭 MAIN LOOP (CSV)
# -------------------------------------------------------------------
def main():
    csv_file = "flights_list.csv"
    save_dir = "airborne_dataset"

    with open(csv_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        entries = [(row["Title"], row["URL"]) for row in reader]

    for i, (name, link) in enumerate(entries, start=1):
        folder = os.path.join(save_dir, name.replace("/", "-"))
        if os.path.exists(folder):
            print(f"⏩ [{i}/{len(entries)}] Skipping {name} — folder already exists.")
            continue

        print(f"\n🔎 [{i}/{len(entries)}] Processing aircraft: {name}")
        extract_specs_with_ollama(name, link, save_dir)
        download_images(name, save_dir)
        time.sleep(2)

if __name__ == "__main__":
    main()
