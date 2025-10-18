import subprocess
import re
import csv
import json
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
import time
import psutil
import os
from selenium.common.exceptions import WebDriverException
from datetime import datetime

base_url = "https://en.wikipedia.org"
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")

def ask_ollama(page_title, page_text, url):
    """
    Send Wikipedia page text to Ollama llama3:8b and ask if it's aircraft names or a subcategory.
    """
    prompt = f"""
Return ONLY valid JSON (double quotes, no trailing commas, enclosed in {{}}). No extra text or comments.
Analyze the Wikipedia page titled "{page_title}" with text:
---
{page_text[:1000]}
---
1. If the page describes a specific aircraft model (e.g., Boeing 747, MiG-21, F-16, ANF Les Mureaux 140T), return its name and URL.
2. If the page title starts with "Category:", return as subcategory with its URL.
3. Otherwise, return SKIP.

Examples:
{{
  "type": "aircraft",
  "items": [
    {{"title": "Boeing 747", "url": "https://en.wikipedia.org/wiki/Boeing_747"}},
    {{"title": "ANF Les Mureaux 140T", "url": "https://en.wikipedia.org/wiki/ANF_Les_Mureaux_140T"}}
  ]
}}
{{
  "type": "subcategory",
  "url": "{url}"
}}
{{
  "type": "skip"
}}
"""
    try:
        process = psutil.Process()
        mem_info = process.memory_info()
        start_time = time.time()
        print(f"DEBUG: [{current_time}] Memory usage before Ollama: {mem_info.rss / 1024 / 1024:.2f} MB")
        
        result = subprocess.run(
            ["ollama", "run", "llama3:8b"],
            input=prompt,
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=20
        )
        response = result.stdout.strip()
        elapsed_time = time.time() - start_time
        print(f"DEBUG: [{current_time}] Ollama processing took {elapsed_time:.2f} seconds")
        print(f"DEBUG: [{current_time}] Raw Ollama output type: {type(response)}, content: {response[:100]}...")
        try:
            json.loads(response)
            return response
        except json.JSONDecodeError:
            print(f"⚠️ [{current_time}] Ollama returned invalid JSON: {response}. Attempting to fix...")
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                response = json_match.group(0)
                try:
                    json.loads(response)
                    print(f"DEBUG: [{current_time}] Fixed JSON: {response[:100]}...")
                    return response
                except json.JSONDecodeError:
                    print(f"⚠️ [{current_time}] Still invalid JSON. Treating as skip.")
            return '{"type": "skip"}'
    except UnicodeEncodeError as e:
        print(f"⚠️ [{current_time}] UnicodeEncodeError in Ollama: {e}. Skipping.")
        return '{"type": "skip"}'
    except subprocess.TimeoutExpired:
        print(f"⚠️ [{current_time}] Ollama timeout for {url}. Skipping.")
        return '{"type": "skip"}'
    except subprocess.CalledProcessError as e:
        print(f"⚠️ [{current_time}] Subprocess error: {e}. Skipping.")
        return '{"type": "skip"}'
    except Exception as e:
        print(f"⚠️ [{current_time}] Unexpected error in ask_ollama: {e}. Skipping.")
        return '{"type": "skip"}'

def is_potential_aircraft_title(title):
    """
    Check if a page title likely represents a specific aircraft model.
    """
    aircraft_patterns = [
        r'\d',  # Contains numbers (e.g., Boeing 747, F-16)
        r'(Boeing|Airbus|Lockheed|McDonnell|Douglas|Cessna|MiG|Sukhoi|Northrop|Grumman|Dassault|Embraer|Bombardier|'
        r'Tupolev|Yakovlev|Antonov|Ilyushin|Saab|Fokker|Hawker|de Havilland|Bell|Eurofighter|Eurocopter|Sikorsky|Piper|Beechcraft)',
    ]
    non_aircraft = [
        'balloon', 'airship', 'parachute', 'model', 'method', 'expedition', 'list', 'avionics', 'industry',
        'system', 'configuration', 'accident', 'incident', 'noise', 'marking', 'maintenance', 'recycling',
        'fnrs', 'acs', 'avic', 'kj'
    ]
    if any(keyword in title.lower() for keyword in non_aircraft):
        return False
    return any(re.search(pattern, title, re.IGNORECASE) for pattern in aircraft_patterns)

def process_page(url, visited, writer, driver, existing_titles, depth=0, csv_file="flights_list.csv"):
    """
    Process a Wikipedia page and handle Ollama response.
    Only stop on user interrupt or error.
    """
    if url in visited:
        print(f"DEBUG: [{current_time}] Already visited {url}. Stopping branch.")
        return
    visited.add(url)
    print(f"DEBUG: [{current_time}] Total pages processed: {len(visited)}")

    print(f"🔎 [{current_time}] Processing: {url} (Depth: {depth})")
    start_time = time.time()
    for attempt in range(3):
        try:
            driver.get(url)
            time.sleep(2)  # Adjustable delay
            soup = BeautifulSoup(driver.page_source, "html.parser")
            elapsed_time = time.time() - start_time
            print(f"DEBUG: [{current_time}] Page fetch took {elapsed_time:.2f} seconds")
            title = soup.find("h1").get_text(strip=True)

            text = soup.get_text(" ", strip=True)
            print(f"DEBUG: [{current_time}] Sending to Ollama - Title: {title}, Text snippet: {text[:100]}...")
            ollama_response = ask_ollama(title, text, url)
            elapsed_time = time.time() - start_time
            print(f"DEBUG: [{current_time}] Total processing time for {url}: {elapsed_time:.2f} seconds")

            print(f"DEBUG: [{current_time}] Ollama response type: {type(ollama_response)}, content: {ollama_response[:200]}...")
            if not isinstance(ollama_response, str):
                raise TypeError(f"Ollama response is {type(ollama_response)}, expected str")

            # Process subcategories and pages
            subcat_div = soup.find("div", id="mw-subcategories")
            if subcat_div:
                subcat_links = subcat_div.find_all("a", href=True)
                print(f"DEBUG: [{current_time}] Found {len(subcat_links)} subcategory links")
                for link in subcat_links:
                    sub_url = base_url + link['href']
                    if sub_url in visited:
                        print(f"DEBUG: [{current_time}] Already visited subcategory {sub_url}. Skipping.")
                        continue
                    print(f"DEBUG: [{current_time}] Processing subcategory: {sub_url}")
                    process_page(sub_url, visited, writer, driver, existing_titles, depth + 1, csv_file)

            page_div = soup.find("div", id="mw-pages")
            if page_div:
                page_links = page_div.find_all("a", href=True)
                print(f"DEBUG: [{current_time}] Found {len(page_links)} page links")
                for link in page_links:
                    page_url = base_url + link['href']
                    page_title = link.get_text(strip=True)
                    if page_url in visited:
                        print(f"DEBUG: [{current_time}] Already visited page {page_url}. Skipping.")
                        continue
                    if any(keyword in page_url.lower() or keyword in page_title.lower() for keyword in [
                        "wikipedia:", "expedition", "method", "list_of", "avionics", "industry", "balloon", "parachute", "airship",
                        "system", "configuration", "accident", "incident", "noise", "marking", "maintenance", "recycling"
                    ]):
                        print(f"DEBUG: [{current_time}] Skipping non-aircraft page: {page_url}")
                        continue
                    if is_potential_aircraft_title(page_title) and page_title not in existing_titles:
                        print(f"✅ [{current_time}] Aircraft (fallback): {page_title}")
                        writer.writerow([page_title, page_url])
                        existing_titles.add(page_title)
                        writer.flush()
                        csv_count = sum(1 for _ in open(csv_file)) - 1
                        print(f"DEBUG: [{current_time}] CSV now has {csv_count} aircraft entries")
                    print(f"DEBUG: [{current_time}] Processing potential aircraft page: {page_url}")
                    process_page(page_url, visited, writer, driver, existing_titles, depth + 1, csv_file)

            # Handle Ollama response
            if '"type": "aircraft"' in ollama_response:
                items = re.findall(r'\"title\":\s*\"(.*?)\".*?\"url\":\s*\"(.*?)\"', ollama_response)
                for name, link in items:
                    if link in visited:
                        print(f"DEBUG: [{current_time}] Already visited aircraft {link}. Skipping.")
                        continue
                    if any(keyword in link.lower() or keyword in name.lower() for keyword in [
                        "wikipedia:", "expedition", "method", "list_of", "avionics", "industry", "balloon", "parachute", "airship",
                        "system", "configuration", "accident", "incident", "noise", "marking", "maintenance", "recycling"
                    ]):
                        print(f"DEBUG: [{current_time}] Skipping non-aircraft page: {link}")
                        continue
                    if name not in existing_titles:
                        print(f"✅ [{current_time}] Aircraft (Ollama): {name}")
                        writer.writerow([name, link])
                        existing_titles.add(name)
                        writer.flush()
                        csv_count = sum(1 for _ in open(csv_file)) - 1
                        print(f"DEBUG: [{current_time}] CSV now has {csv_count} aircraft entries")
            elif '"type": "subcategory"' in ollama_response:
                print(f"DEBUG: [{current_time}] Ollama identified as subcategory, already processing subcategories and pages")
            else:
                print(f"❌ [{current_time}] Skipping: {title} (Ollama response: {ollama_response})")
                if not subcat_div and not page_div:
                    print(f"DEBUG: [{current_time}] No subcategories or pages found in {url}")
            break
        except WebDriverException as e:
            print(f"⚠️ [{current_time}] Network error processing {url}: {e}. Retrying ({attempt + 1}/3)...")
            time.sleep(5 * (attempt + 1))  # Exponential backoff
            if attempt == 2:
                print(f"⚠️ [{current_time}] Failed to process {url} after 3 attempts. Skipping.")
                break
        except Exception as e:
            print(f"⚠️ [{current_time}] Error processing {url}: {e}. Skipping.")
            break

def main():
    """
    Main function to set up Selenium and start processing.
    """
    start_url = "https://en.wikipedia.org/wiki/Category:Aircraft"
    csv_file = "flights_list.csv"
    visited = set()
    existing_titles = set()

    # Load existing CSV for visited URLs and titles
    try:
        with open(csv_file, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 2:
                    title, url = row[0], row[1]
                    visited.add(url)
                    existing_titles.add(title)
            print(f"DEBUG: [{current_time}] Loaded {len(visited)} visited URLs and {len(existing_titles)} existing titles from CSV.")
    except FileNotFoundError:
        print(f"DEBUG: [{current_time}] No existing CSV found. Starting fresh.")
        with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Title", "URL"])

    options = Options()
    options.add_argument("--headless")
    service = Service()  # Specify executable_path if geckodriver not in PATH
    driver = webdriver.Firefox(service=service, options=options)

    try:
        with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            process_page(start_url, visited, writer, driver, existing_titles, depth=0, csv_file=csv_file)
    except KeyboardInterrupt:
        print(f"⚠️ [{current_time}] Script stopped by user (Ctrl+C).")
    except Exception as e:
        print(f"⚠️ [{current_time}] Error in main: {e}")
    finally:
        driver.quit()

    print(f"🎯 [{current_time}] Finished. Results appended to {csv_file}")

if __name__ == "__main__":
    main()