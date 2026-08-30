"""
scraper.py — Zero-network ingestion pipeline for Unreported India.
Bypasses live scraping to avoid IP blocks and cleans AI Markdown wrappers.
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
# 1. Hardcoded Ground Data (Bypasses scraper blocks)
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

def clean_ai_json(text):
    """Strips Markdown wrappers (```json ... ```) and preamble from AI response."""
    # Remove markdown code blocks
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```", "", text)
    # Find the start of the actual JSON object
    start_index = text.find('{')
    end_index = text.rfind('}')
    if start_index != -1 and end_index != -1:
        return text[start_index:end_index+1]
    return text.strip()

def analyze_with_gemini(client, report, official_text):
    prompt = f"""
    Compare this Field Report to the Official Government Statement.
    Independent Report: {report['field_notes']}
    Official Statement: {official_text}
    
    Task:
    1. headline: Create a short headline (max 10 words).
    2. body: Write a 2 sentence summary of the field findings.
    3. official_quote: Specific contradictory line from Official Statement.
    4. official_source: Use 'District Health Department'.
    5. missing: List 3 specific things ignored (e.g., actual death count).
    6. severity: Rate urgency 1 to 5.
    
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
    # Ensure response is treated as string and cleaned
    raw_response = response.text if hasattr(response, 'text') else str(response)
    cleaned_json = clean_ai_json(raw_response)
    return json.loads(cleaned_json)

def build_news_feed():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("CRITICAL: GEMINI_API_KEY missing.")
        return

    client = genai.Client(api_key=api_key)
    entries = []

    for report in GROUND_TRUTH_REPORTS:
        try:
            analysis = analyze_with_gemini(client, report, OFFICIAL_PRESS_RELEASES)
            entries.append({
                "id": report["id"],
                "date": datetime.strptime(report["date"], "%Y%m%d").strftime("%d %b"),
                "reporter": report["reporter"],
                "location": report["location"],
                "severity": int(analysis.get("severity", 3)),
                "headline": analysis.get("headline", "Crisis Report"),
                "body": analysis.get("body", "Field notes pending."),
                "official": analysis.get("official_quote", "Statement pending."),
                "officialSource": analysis.get("official_source", "District Administration"),
                "missing": analysis.get("missing", [])
            })
            time.sleep(1) # Rate limit protection
        except Exception as e:
            print(f"Error processing {report['id']}: {e}")

    # Write cleaned raw JSON array
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    print(f"SUCCESS: {len(entries)} entries written to {OUTPUT_PATH}")

if __name__ == "__main__":
    build_news_feed()
