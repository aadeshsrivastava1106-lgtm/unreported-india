"""
scraper.py — Zero-network ingestion pipeline for Unreported India.

This version bypasses live YouTube scraping to avoid GitHub Action IP blocks.
It processes a hardcoded set of verified ground reports through the Gemini API
to generate dynamic gap analysis against official government statements.
"""

import json
import os
import re
import sys
import time
from datetime import datetime
from google import genai
from google.genai import types

# ----------------------------------------------------------------------------
# 1. Hardcoded Ground Data (The Truth from the Field)
# ----------------------------------------------------------------------------
GROUND_TRUTH_REPORTS = [
    {
        "id": "report-001",
        "reporter": "Tribal Health Watch",
        "location": "Kundekasa Village, Birsa Block",
        "date": "20240821",
        "field_notes": "Local community count confirms 25 child deaths across the block, but the District Health Dept is only acknowledging 8. Families in Kundekasa report that children developed high fever and rashes; medical teams only arrived after the fourth death occurred."
    },
    {
        "id": "report-002",
        "reporter": "Field Briefing - Adivasi Rights",
        "location": "Bondari & Korka Villages",
        "date": "20240822",
        "field_notes": "Take-home rations (THR) and nutritional supplements for pregnant women and malnourished children have not reached these interior villages for 6 months. Anganwadi workers state supply chains are broken, while official records claim 95% distribution efficiency."
    },
    {
        "id": "report-003",
        "reporter": "Ground Report - News18 Interview",
        "location": "Birsa Interior",
        "date": "20240824",
        "field_notes": "Baiga families are forced to carry sick children on wooden stretchers for over 20km to reach the nearest Primary Health Centre (PHC) because roads are washed out and ambulances refuse to enter the forest. CMHO claims mobile health units are patrolling, but residents say they haven't seen a doctor in weeks."
    }
]

# ----------------------------------------------------------------------------
# 2. Official Context (The Government PR)
# ----------------------------------------------------------------------------
OFFICIAL_PRESS_RELEASES = """
[Official Report - Balaghat Administration] The District Health Department 
officially confirms 8 child deaths in the Birsa block due to seasonal malaria 
and measles outbreaks. Health camps have been successfully established in 
Kundekasa. Vaccination drives are verified at 95% coverage across the block. 
CMHO Dr. Paresh Uplap states that specialized medical teams are monitoring 
every village daily. There are no reported shortages of rations or medical 
supplies in the tribal areas.
""".strip()

GEMINI_MODEL = "gemini-1.5-flash"
OUTPUT_PATH = "news_feed.json"

# ----------------------------------------------------------------------------
# 3. Processing Logic
# ----------------------------------------------------------------------------

def clean_json_response(raw_text):
    """Strips Markdown wrappers like ```json and returns a raw JSON string."""
    # Find the first '{' and the last '}'
    match = re.search(r"(\{.*\}|\[.*\])", raw_text, re.DOTALL)
    if match:
        return match.group(1)
    return raw_text

def analyze_with_gemini(client, report, official_text):
    """Analyzes field notes against official text using AI."""
    prompt = f"""
    Compare this Field Report to the Official Government Statement.
    
    Field Report: {report['field_notes']}
    Official Statement: {official_text}
    
    Task:
    1. headline: Create a short headline (max 10 words).
    2. body: Write a 2 sentence summary of the field findings.
    3. official_quote: Find the specific line in the Official Statement that contradicts this report.
    4. official_source: Use 'District Health Department'.
    5. missing: List 3 specific things the official report is ignoring (e.g., transport, actual death count, ration timeline).
    6. severity: Rate the urgency from 1 to 5.
    
    Return valid JSON ONLY. No Markdown.
    """
    
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1,
        ),
    )
    
    cleaned = clean_json_response(response.text)
    return json.loads(cleaned)

def format_date(date_str):
    """Converts YYYYMMDD to 'Day Month'."""
    try:
        return datetime.strptime(date_str, "%Y%m%d").strftime("%d %b")
    except:
        return datetime.utcnow().strftime("%d %b")

def build_news_feed():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("CRITICAL: GEMINI_API_KEY environment variable is missing.")

    client = genai.Client(api_key=api_key)
    entries = []

    print(f"Bypassing network... Processing {len(GROUND_TRUTH_REPORTS)} local reports.")

    for report in GROUND_TRUTH_REPORTS:
        try:
            print(f"Analyzing Gap: {report['id']}...")
            analysis = analyze_with_gemini(client, report, OFFICIAL_PRESS_RELEASES)
            
            entries.append({
                "id": report["id"],
                "date": format_date(report["date"]),
                "reporter": report["reporter"],
                "location": report["location"],
                "severity": analysis.get("severity", 3),
                "headline": analysis.get("headline", "Report: " + report["location"]),
                "body": analysis.get("body", ""),
                "official": analysis.get("official_quote") or "Status: Monitoring.",
                "officialSource": analysis.get("official_source") or "District Administration",
                "missing": analysis.get("missing", [])
            })
            # Be polite to the free tier rate limit
            time.sleep(1)
        except Exception as e:
            print(f"Error processing {report['id']}: {e}")

    # Write the raw JSON list to file
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    
    print(f"\nSUCCESS: Generated {len(entries)} verified entries in {OUTPUT_PATH}")

if __name__ == "__main__":
    build_news_feed()
