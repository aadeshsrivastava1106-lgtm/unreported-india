"""
scraper.py — zero-cost ingestion pipeline for Unreported India.

Logic:
  1. Attempts to pull live transcripts/metadata from YouTube via yt-dlp.
  2. If blocked by YouTube (Cloud IP protection) or URL fails, it triggers 
     the GROUND_DATA_FALLBACK to ensure the prototype is always populated.
  3. Sends the text (Live or Fallback) to Gemini 1.5 Flash to compute 
     discrepancies against the OFFICIAL_PRESS_RELEASES.
  4. Saves a perfectly structured news_feed.json for the frontend.
"""

import glob
import json
import os
import re
import sys
import tempfile
import time
from datetime import datetime

import yt_dlp
from google import genai
from google.genai import types

# ----------------------------------------------------------------------------
# Hardcoded Inputs: Official Context
# ----------------------------------------------------------------------------

OFFICIAL_PRESS_RELEASES = """
[Official Report - Balaghat Admin] The District Health Department confirms 
8 child deaths in the Birsa block due to a seasonal malaria and measles 
outbreak. Health camps have been established in Kundekasa. Vaccination 
drives are reportedly at 95% coverage. CMHO Dr. Paresh Uplap states that 
medical teams are present and monitoring the situation. No shortage of 
nutritional supplements or rations has been officially recorded.
""".strip()

# ----------------------------------------------------------------------------
# Hardcoded Inputs: Real Ground Data (Used if yt-dlp is blocked)
# ----------------------------------------------------------------------------

GROUND_DATA_FALLBACK = [
    {
        "title": "Mass Child Deaths in Kundekasa Village",
        "content": "Local tribal leaders in Kundekasa report at least 25 child deaths, tripling the official count. Families claim children died with high fever and rashes after waiting days for medical help that never arrived.",
        "reporter": "Field Report - Tribal Pulse",
        "location": "Kundekasa Village",
        "date": "20240820"
    },
    {
        "title": "Ration Crisis in Birsa Block",
        "content": "Adivasi families in Bondari and Korka villages report that the 'Take Home Ration' (THR) for pregnant women and malnourished children has not been delivered for 6 months. Empty Anganwadi centers are being reported across the block.",
        "reporter": "Ground Briefing",
        "location": "Bondari/Korka Villages",
        "date": "20240822"
    },
    {
        "title": "20km Journey for Basic Healthcare",
        "content": "Tribal residents of the Baiga community are forced to carry sick children on makeshift stretchers for over 20km to reach the nearest Primary Health Centre. Poor road connectivity is cited as a major factor in the rising death toll.",
        "reporter": "News18 Field Interview",
        "location": "Birsa Block Interior",
        "date": "20240824"
    }
]

INDEPENDENT_SOURCES = [
    {"url": "https://www.youtube.com/watch?v=D-ZpXn3fH50", "reporter": "TV9 MP"},
    {"url": "https://www.youtube.com/watch?v=7uK_Z_Eivm4", "reporter": "PTI News"},
    {"url": "https://www.youtube.com/watch?v=kY67l0F-D40", "reporter": "News18 MP"}
]

GEMINI_MODEL = "gemini-1.5-flash"
OUTPUT_PATH = "news_feed.json"
SECONDS_BETWEEN_CALLS = 2

# ----------------------------------------------------------------------------
# Extraction & Cleaning Logic
# ----------------------------------------------------------------------------

def clean_json_response(raw_text):
    """Strips Markdown code blocks and extracts raw JSON."""
    # Find the first '{' and the last '}' to ignore Markdown formatting or extra text
    match = re.search(r"(\{.*\}|\[.*\])", raw_text, re.DOTALL)
    if match:
        return match.group(1)
    return raw_text

def extract_video_data(url):
    """Attempt to fetch live data; return None if blocked or error."""
    try:
        ydl_opts = {"skip_download": True, "quiet": True, "no_warnings": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if not info or not info.get('title'):
                return None
            return {
                "id": str(info.get("id", "")),
                "title": info.get("title", ""),
                "transcript": info.get("description", "")[:2000],
                "upload_date": info.get("upload_date")
            }
    except Exception:
        return None

# ----------------------------------------------------------------------------
# Gemini Analysis Logic
# ----------------------------------------------------------------------------

def analyze_with_gemini(client, report_text, official_text):
    prompt = f"""
    Compare this Independent Field Report to the Official Government Statement.
    
    Independent Report: {report_text}
    Official Statement: {official_text}
    
    Task:
    1. headline: Summarize independent report (max 10 words).
    2. body: 2 sentence summary of report.
    3. official_quote: The specific line from the official statement that is being challenged.
    4. official_source: Name of the official department.
    5. missing: List 3 specific things the official report is ignoring or undercounting.
    6. severity: Rate urgency from 1 (low) to 5 (critical).
    
    Return valid JSON ONLY. Do not use Markdown tags.
    """
    
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1,
        ),
    )
    
    cleaned_json = clean_json_response(response.text)
    return json.loads(cleaned_json)

# ----------------------------------------------------------------------------
# Formatting
# ----------------------------------------------------------------------------

def format_date(date_str):
    if not date_str: return datetime.utcnow().strftime("%d %b")
    try:
        return datetime.strptime(date_str, "%Y%m%d").strftime("%d %b")
    except:
        return datetime.utcnow().strftime("%d %b")

# ----------------------------------------------------------------------------
# Main Orchestrator
# ----------------------------------------------------------------------------

def build_news_feed():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("GEMINI_API_KEY not found in environment.")

    client = genai.Client(api_key=api_key)
    entries = []

    print("Starting ingestion...")

    for i, source in enumerate(INDEPENDENT_SOURCES):
        # 1. Attempt Live Fetch
        print(f"[{i+1}] Fetching: {source['url']}")
        data = extract_video_data(source["url"])
        
        # 2. Fallback if Blocked
        if not data:
            print(f"    ! YouTube blocked fetch. Injecting Fallback Ground Data [{i}].")
            fallback = GROUND_DATA_FALLBACK[i % len(GROUND_DATA_FALLBACK)]
            data = {
                "id": f"fallback-{i}",
                "title": fallback["title"],
                "transcript": fallback["content"],
                "upload_date": fallback["date"],
                "location": fallback["location"],
                "reporter": fallback["reporter"]
            }
        else:
            data["location"] = "Balaghat District"
            data["reporter"] = source["reporter"]

        # 3. Analyze with AI
        try:
            print(f"    > AI Analysis in progress...")
            analysis = analyze_with_gemini(client, data["transcript"], OFFICIAL_PRESS_RELEASES)
            
            entries.append({
                "id": data["id"],
                "date": format_date(data["upload_date"]),
                "reporter": data["reporter"],
                "location": data["location"],
                "severity": analysis.get("severity", 3),
                "headline": analysis.get("headline", data["title"]),
                "body": analysis.get("body", ""),
                "official": analysis.get("official_quote") or "No direct matching official statement found.",
                "officialSource": analysis.get("official_source") or "District Administration",
                "missing": analysis.get("missing", [])
            })
            time.sleep(SECONDS_BETWEEN_CALLS)
        except Exception as e:
            print(f"    ! Error parsing AI response for entry {i}: {e}")

    # 4. Save Final JSON
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    
    print(f"\nCOMPLETED: {len(entries)} entries saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    build_news_feed()
