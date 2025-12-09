#!/usr/bin/env python3
"""
collector_no_api.py
No API keys, no rate limits, no nonsense.
Pulls aircraft images from public sources that don't block scraping (yet).
Works well enough for 20–50 high-quality photos per type without getting banned.
"""

import os
import re
import csv
import json
import time
import random
import logging
import requests
from tqdm import tqdm
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optional perceptual hash deduplication – nice when it works
try:
    from PIL import Image
    import imagehash
    HASH_AVAILABLE = True
except ImportError:  # people forget to pip install imagehash all the time
    HASH_AVAILABLE = False

# ------------------------------------------------------------------
# Config – tweak these if you start getting blocked or want more images
# ------------------------------------------------------------------
SAVE_DIR = "aircraft_dataset"           # where everything lands
CSV_FILE = "flights_list.csv"           # one aircraft per row, "Title" column or just plain list
TARGET_PER_CLASS = 20                   # stop once we hit this per folder
MAX_THREADS = 20                        # more than this usually gets you temp-blocked
DOWNLOAD_MIN_BYTES = 12_000             # ignore obvious thumbnails
DOWNLOAD_RETRIES = 3                    # some images 404 on first try, retry a couple times
DEDUP_DB = "dedup.json"                 # simple pHash database across runs
MISSING_LOG = "missing.log"             # aircraft that returned zero results

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"
}

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/run.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)


# ------------------------------------------------------------------
# Tiny helpers – nothing fancy
# ------------------------------------------------------------------
def ensure_dir(path):
    """Just os.makedirs(exist_ok=True) but I got tired of typing it."""
    os.makedirs(path, exist_ok=True)


def safe_name(name):
    """Turn anything into a valid folder name. Replaces slashes, parentheses, etc."""
    return "".join(c if c.isalnum() or c in " _-" else "_" for c in name).strip()


def count_images_recursive(path):
    """Walk the class folder and count actual image files. Used to skip already-done classes."""
    exts = (".jpg", ".jpeg", ".png", ".webp", ".gif")
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            if f.lower().endswith(exts):
                total += 1
    return total


def log_missing(name):
    """Keep track of aircraft that gave us literally nothing."""
    with open(MISSING_LOG, "a", encoding="utf-8") as f:
        f.write(name + "\n")


# ------------------------------------------------------------------
# Alias generation – because nobody searches the exact same string
# ------------------------------------------------------------------
def alias_queries(name):
    """
    People search "F-16", "F16 Fighting Falcon", "F-16 aircraft", etc.
    Throw a bunch of variations at the sources so we actually find something.
    """
    clean = re.sub(r"\(.*?\)", "", name).strip()  # drop (whatever) at the end
    variants = {
        name,
        clean,
        clean.replace("-", " "),
        clean.replace(" ", "-"),
        f"{clean} aircraft",
        f"{clean} helicopter",
        f"{clean} airplane",
        f"{clean} drone",
        f"{clean} uav",
        f"{clean} photo",
    }
    tokens = clean.split()
    if len(tokens) >= 2:
        variants.add(" ".join(tokens[:2]))   # e.g. "F-16 Fighting"
        variants.add(tokens[-1])             # just the model code sometimes works
    return list(variants)


# ------------------------------------------------------------------
# Wikimedia Commons – the real gold mine
# ------------------------------------------------------------------
def wikimedia_search(query, limit=200):
    """
    Uses the public MediaWiki API. No key needed.
    Returns direct URLs (often full-res) instead of thumbnails.
    """
    results = []
    seen = set()
    api = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "imageinfo",
        "generator": "search",
        "gsrnamespace": 6,           # File: namespace
        "gsrsearch": query,
        "gsrlimit": 50,
        "iiprop": "url|size|mime",
        "iiurlwidth": 4096,          # ask for biggest thumb – often gives full URL
    }
    cont = {}
    while len(results) < limit:
        p = params.copy()
        p.update(cont)
        try:
            r = requests.get(api, params=p, headers=HEADERS, timeout=15)
            data = r.json()
            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                ii = page.get("imageinfo", [{}])[0]
                url = ii.get("thumburl") or ii.get("url")
                if url and url not in seen:
                    seen.add(url)
                    results.append(url)
            cont = data.get("continue", {})
            if not cont:
                break
        except Exception as e:
            logging.debug(f"Wikimedia search broke: {e}")
            break
    return results[:limit]


def wikimedia_category(category_title, limit=300):
    """
    Some aircraft have their own Category: page with hundreds of clean photos.
    Example: Category:F-16 Fighting Falcon
    """
    results = []
    seen = set()
    api = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "categorymembers",
        "cmtitle": category_title,
        "cmnamespace": 6,      # File namespace only
        "cmlimit": 50
    }
    cont = {}
    while len(results) < limit:
        p = params.copy()
        p.update(cont)
        try:
            r = requests.get(api, params=p, headers=HEADERS, timeout=15)
            data = r.json()
            for member in data.get("query", {}).get("categorymembers", []):
                title = member["title"]
                # Quick second query to get the direct URL
                info_url = (
                    f"{api}?action=query&format=json&titles={quote(title)}"
                    "&prop=imageinfo&iiprop=url"
                )
                try:
                    r2 = requests.get(info_url, headers=HEADERS, timeout=10)
                    for page in r2.json().get("query", {}).get("pages", {}).values():
                        ii = page.get("imageinfo", [{}])[0]
                        url = ii.get("url")
                        if url and url not in seen:
                            seen.add(url)
                            results.append(url)
                except:
                    continue
            cont = data.get("continue", {})
            if not cont:
                break
        except:
            break
    return results[:limit]


def wikimedia_possible_categories(name):
    """Guess plausible Wikimedia category names."""
    base = safe_name(name).replace(" ", "_")
    guesses = [
        f"Category:{base}",
        f"Category:{base}_aircraft",
        f"Category:{base}_helicopter",
        f"Category:{base}_(aircraft)",
    ]
    return guesses


# ------------------------------------------------------------------
# DuckDuckGo image search – works surprisingly well and no key needed
# ------------------------------------------------------------------
def ddg_images(query, max_results=100):
    """
    Classic vqd token trick. Fragile, but has been stable for years.
    Returns direct image links most of the time.
    """
    try:
        # Grab the vqd token from the main page
        init = requests.get("https://duckduckgo.com/", headers=HEADERS, timeout=10)
        token_match = re.search(r"vqd='([0-9-]+)'", init.text)
        if not token_match:
            return []
        token = token_match.group(1)

        urls = []
        s = 0
        while len(urls) < max_results:
            params = {
                "l": "us-en",
                "o": "json",
                "q": query,
                "vqd": token,
                "s": s,
            }
            r = requests.get("https://duckduckgo.com/i.js", params=params, headers=HEADERS, timeout=10)
            data = r.json()
            for item in data.get("results", []):
                img_url = item.get("image")
                if img_url:
                    urls.append(img_url)
            if "next" not in data:
                break
            s += len(data.get("results", []))
            time.sleep(0.2)
        return urls[:max_results]
    except Exception:
        return []


# ------------------------------------------------------------------
# Reddit – last resort, lots of low-res stuff but sometimes unique shots
# ------------------------------------------------------------------
def reddit_images(query, max_results=100):
    """
    Public search.json endpoint. No auth needed.
    Grabs preview images and direct links when available.
    """
    try:
        url = f"https://www.reddit.com/search.json?q={quote(query)}&sort=relevance&t=all&limit=100"
        r = requests.get(url, headers=HEADERS, timeout=12)
        items = r.json().get("data", {}).get("children", [])
        urls = []
        for item in items:
            data = item.get("data", {})
            # High-res preview if it exists
            if "preview" in data and "images" in data["preview"]:
                highres = data["preview"]["images"][0]["source"]["url"]
                urls.append(highres.replace("&amp;", "&"))
            # Direct jpg/png links
            if data.get("url", "").lower().endswith((".jpg", ".jpeg", ".png")):
                urls.append(data["url"])
            if len(urls) >= max_results:
                break
        return urls[:max_results]
    except Exception:
        return []


# ------------------------------------------------------------------
# Collect everything into one big list, stop early if we have enough
# ------------------------------------------------------------------
def collect_candidates(name, target):
    """
    Try the best sources first (Wikimedia categories → search → DDG → Reddit).
    Deduplicate on the fly.
    """
    queries = alias_queries(name)
    results = []
    seen = set()

    # 1. Wikimedia categories – usually the cleanest photos
    for cat in wikimedia_possible_categories(name):
        urls = wikimedia_category(cat, limit=300)
        for u in urls:
            if u not in seen:
                seen.add(u)
                results.append(u)
        if len(results) >= target:
            return results[:target]

    # 2. Wikimedia full-text search
    for q in queries:
        urls = wikimedia_search(q, limit=200)
        for u in urls:
            if u not in seen:
                seen.add(u)
                results.append(u)
        if len(results) >= target:
            return results[:target]

    # 3. DuckDuckGo
    for q in queries:
        urls = ddg_images(q, max_results=150)
        for u in urls:
            if u not in seen:
                seen.add(u)
                results.append(u)
        if len(results) >= target:
            return results[:target]

    # 4. Reddit – noisy but sometimes catches rare types
    for q in queries:
        urls = reddit_images(q, max_results=150)
        for u in urls:
            if u not in seen:
                seen.add(u)
                results.append(u)
        if len(results) >= target:
            return results[:target]

    return results[:target]


# ------------------------------------------------------------------
# Actual downloading – parallel, with retries and optional pHash dedup
# ------------------------------------------------------------------
def download_file(url, path):
    """Download with a few retries. Returns True on success."""
    for _ in range(DOWNLOAD_RETRIES):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code != 200:
                continue
            data = r.content
            if len(data) < DOWNLOAD_MIN_BYTES:
                return False
            with open(path, "wb") as f:
                f.write(data)
            return True
        except Exception:
            time.sleep(0.3 + random.random())
    return False


def compute_phash(path):
    """Return perceptual hash if Pillow+imagehash are installed."""
    if not HASH_AVAILABLE:
        return None
    try:
        with Image.open(path) as im:
            return str(imagehash.phash(im))
    except Exception:
        return None


def download_candidates(urls, outdir, prefix, existing_count, target_total):
    """
    Fire off parallel downloads.
    Skip duplicates using pHash if available.
    Stop once we hit the target.
    """
    ensure_dir(outdir)

    # Load previous hashes so we don't redownload near-identical images
    try:
        seen_hashes = set(json.load(open(DEDUP_DB)))
    except Exception:
        seen_hashes = set()

    saved = 0
    futures_to_path = {}

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        for idx, url in enumerate(urls, start=1):
            filepath = os.path.join(outdir, f"{prefix}_{existing_count + idx}.jpg")
            futures_to_path[executor.submit(download_file, url, filepath)] = filepath

        for future in tqdm(as_completed(futures_to_path),
                           total=len(futures_to_path),
                           desc="Downloading",
                           leave=False):
            path = futures_to_path[future]
            success = False
            try:
                success = future.result()
            except Exception:
                pass

            if not success:
                if os.path.exists(path):
                    os.remove(path)
                continue

            # Deduplication step
            h = compute_phash(path)
            if h and h in seen_hashes:
                os.remove(path)   # near-duplicate
                continue
            if h:
                seen_hashes.add(h)

            saved += 1
            if existing_count + saved >= target_total:
                break

    # Persist hash database for next run
    if HASH_AVAILABLE and seen_hashes:
        try:
            with open(DEDUP_DB, "w") as f:
                json.dump(list(seen_hashes), f)
        except Exception:
            pass

    return saved


# ------------------------------------------------------------------
# Process a single aircraft type
# ------------------------------------------------------------------
def process_aircraft(name):
    """Main worker function for one row in the CSV."""
    safe = safe_name(name)
    class_dir = os.path.join(SAVE_DIR, safe)
    img_dir = os.path.join(class_dir, "images")
    ensure_dir(img_dir)

    current_count = count_images_recursive(class_dir)
    if current_count >= TARGET_PER_CLASS:
        logging.info(f"Skipping {name} – already has {current_count} images")
        return

    logging.info(f"Processing {name} – have {current_count}, targeting {TARGET_PER_CLASS}")

    candidates = collect_candidates(name, target=TARGET_PER_CLASS * 5)
    logging.info(f"{name} → found {len(candidates)} candidate URLs")

    if not candidates:
        logging.warning(f"Nothing found for {name}")
        log_missing(name)
        return

    needed = TARGET_PER_CLASS - current_count
    # Grab more than needed because many will be tiny or duplicates
    selected = candidates[:needed * 4]

    added = download_candidates(selected, img_dir, safe, current_count, TARGET_PER_CLASS)
    total_now = current_count + added
    logging.info(f"{name} → +{added} images (now {total_now})")


# ------------------------------------------------------------------
# CSV loading + main loop
# ------------------------------------------------------------------
def load_aircraft_list(csv_path):
    """Accepts either a proper CSV with 'Title' column or just a plain list."""
    if not os.path.exists(csv_path):
        logging.error(f"{csv_path} not found")
        return []

    items = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        try:
            reader = csv.DictReader(f)
            if "Title" in reader.fieldnames:
                for row in reader:
                    title = row["Title"].strip()
                    if title:
                        items.append(title)
            else:
                # Fallback: treat as plain list
                f.seek(0)
                for line in f:
                    title = line.strip()
                    if title and not title.startswith("#"):
                        items.append(title)
        except Exception as e:
            logging.error(f"Failed to read CSV: {e}")
    return items


def main():
    ensure_dir(SAVE_DIR)
    aircraft_list = load_aircraft_list(CSV_FILE)

    if not aircraft_list:
        logging.error("No aircraft loaded – check your CSV")
        return

    logging.info(f"Starting collection of {len(aircraft_list)} aircraft types")

    for idx, name in enumerate(aircraft_list, start=1):
        logging.info(f"[{idx}/{len(aircraft_list)}] {name}")
        try:
            process_aircraft(name)
        except KeyboardInterrupt:
            logging.info("Stopped by user")
            break
        except Exception as e:
            logging.exception(f"Crashed on {name}: {e}")

        time.sleep(0.3)  # be nice to the servers

    logging.info("All done.")


if __name__ == "__main__":
    main()