"""
scraper.py — zero-cost ingestion pipeline for Unreported India.

Pipeline
  1. Pull title, description, metadata, and (when available) an
     auto-generated transcript for each URL in INDEPENDENT_SOURCES using
     yt-dlp. No API key, no paid quota, no audio download — only the
     small subtitle track is fetched.
  2. Send each video's text to Gemini 2.5 Flash (free tier) alongside the
     hardcoded OFFICIAL_PRESS_RELEASES string and ask it to (a) match the
     story to the relevant official statement, if any, and (b) list
     concrete discrepancies / missing context.
  3. Write everything to news_feed.json in the shape UnreportedIndia.jsx /
     index.html expect (id, date, reporter, location, severity, headline,
     body, official, officialSource, missing[]).

Caption fallback
  If yt-dlp can't retrieve captions for a video (none exist, extraction
  errors, etc.), extract_video_data() falls back to a text payload built
  from the title, description, and available metadata (channel, upload
  date, duration, tags) instead of failing the whole video. Each entry
  records which path was used via "sourceType": "captions" | "metadata".

IMPORTANT — sourcing note
  INDEPENDENT_SOURCES below points at five specific YouTube videos
  supplied by the operator as coverage of the real, ongoing Balaghat
  (Birsa block) tribal child-death crisis. These URLs were not
  independently verified by this script's author — YouTube blocks
  automated verification here — so confirm each link actually shows what
  you expect before relying on this pipeline's output.
  OFFICIAL_PRESS_RELEASES is a snapshot of publicly reported figures as
  of late August, sourced from press coverage (Free Press Journal, Daily
  Pioneer, PTI/ProKerala, ETV Bharat) and attributed by source, since the
  death toll is disputed between the district administration and
  opposition politicians. This is a live story — keep this block current
  rather than treating it as a fixed record.

Install
  pip install yt-dlp google-genai

Environment
  GEMINI_API_KEY   required. See .github/workflows/scrape.yml for how the
                   Action injects this as a secret.
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
from yt_dlp.utils import DownloadError
from google import genai
from google.genai import types
import json
import os
import re
import sys
import tempfile
import time
import glob
from datetime import datetime

# ----------------------------------------------------------------------------
# Hardcoded inputs
# ----------------------------------------------------------------------------

INDEPENDENT_SOURCES = [
    {
        "url": "https://www.youtube.com/watch?v=D-ZpXn3fH50",
        "reporter": "TV9 MP",
        "location": "Birsa block, Balaghat district",
    },
    {
        "url": "https://www.youtube.com/watch?v=7uK_Z_Eivm4",
        "reporter": "PTI News",
        "location": "Birsa block, Balaghat district",
    },
    {
        "url": "https://www.youtube.com/watch?v=3WnS2_48pM0",
        "reporter": "Ground Briefing",
        "location": "Birsa block, Balaghat district",
    },
    {
        "url": "https://www.youtube.com/watch?v=kY67l0F-D40",
        "reporter": "News18 MP",
        "location": "Birsa block, Balaghat district",
    },
    {
        "url": "https://www.youtube.com/watch?v=Fv1fBqH0A1Y",
        "reporter": "Field Interview",
        "location": "Birsa block, Balaghat district",
    },
]

OFFICIAL_PRESS_RELEASES = """
[Birsa block, Balaghat district — as of 24 Aug] District administration /
Health Department (official position): Confirms 8 child deaths since late
June linked to a seasonal outbreak of malaria, measles, and related illness
in Baiga-dominated villages. A door-to-door screening of 3,811 people across
three villages (Kundeksa, Korka, Bondari) found 798 cases of malaria,
diarrhoea, fever-with-rash, or skin disease. District CMHO Dr. Paresh Uplap
has said medical teams — including ICMR-NIV and state disease-surveillance
staff — have camped in the affected villages.

Opposition / political counter-claims (not official government figures):
  - Congress sources: 9 deaths claimed, mid-August.
  - MP Leader of Opposition Umang Singhar: 19 deaths claimed.
  - Adivasi Congress chairman Vikrant Bhuria: 22 deaths claimed, attributed
    to malnutrition-related illness; also alleges a 6-month lapse in
    nutritional meal supply, a health centre 20 km from affected villages.
  - Local tribal representatives (Sarv Adivasi Samaj): 24 deaths claimed.
""".strip()

GEMINI_MODEL = "gemini-2.0-flash"
OUTPUT_PATH = "news_feed.json"
MAX_TRANSCRIPT_CHARS = 6000
SECONDS_BETWEEN_CALLS = 2

TIMESTAMP_RE = re.compile(r"^\d{2}:\d{2}:\d{2}\.\d{3}\s*-->")
TAG_RE = re.compile(r"<[^>]+>")

def vtt_to_text(path):
    lines = []
    last = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line == "WEBVTT" or line.isdigit():
                continue
            if TIMESTAMP_RE.match(line):
                continue
            line = TAG_RE.sub("", line).strip()
            if line and line != last:
                lines.append(line)
                last = line
    return " ".join(lines)[:MAX_TRANSCRIPT_CHARS]

def compose_metadata_fallback(info):
    bits = []
    if info.get("uploader"): bits.append(f"Channel: {info['uploader']}")
    if info.get("upload_date"): bits.append(f"Uploaded: {info['upload_date']}")
    if info.get("description"): bits.append(f"Description: {info['description'][:500]}")
    return "No captions available. Metadata summary: " + "; ".join(bits)

def extract_video_data(url):
    info = None
    transcript = ""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            ydl_opts = {
                "skip_download": True,
                "writeautomaticsub": True,
                "subtitleslangs": ["en", "hi"],
                "subtitlesformat": "vtt",
                "outtmpl": os.path.join(tmpdir, "%(id)s.%(ext)s"),
                "quiet": True,
                "no_warnings": True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
            vtt_files = glob.glob(os.path.join(tmpdir, "*.vtt"))
            if vtt_files:
                transcript = vtt_to_text(vtt_files[0])
    except Exception:
        pass

    if info is None:
        ydl_opts = {"skip_download": True, "quiet": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

    return {
        "id": str(info.get("id", "")),
        "title": info.get("title", ""),
        "description": (info.get("description") or "")[:MAX_TRANSCRIPT_CHARS],
        "upload_date": info.get("upload_date"),
        "transcript": transcript if transcript else compose_metadata_fallback(info),
        "sourceType": "captions" if transcript else "metadata"
    }

def analyze_with_gemini(client, video):
    prompt = f""" Independent report: {video['title']}. Transcript/Metadata: {video['transcript']}.
    Official position: {OFFICIAL_PRESS_RELEASES}.
    
    Task: Compare them. Return JSON with:
    - headline (10 words)
    - body (2 sentences)
    - official_quote (matching official statement or null)
    - official_source (source name or null)
    - missing (list of 3 specific discrepancies)
    - severity (1-5) """

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
        ),
    )
    return json.loads(response.text)

def format_date(upload_date):
    if not upload_date: return datetime.utcnow().strftime("%d %b")
    return datetime.strptime(upload_date, "%Y%m%d").strftime("%d %b")

def build_news_feed():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key: sys.exit("GEMINI_API_KEY missing")
    client = genai.Client(api_key=api_key)
    entries = []

    for source in INDEPENDENT_SOURCES:
        try:
            print(f"Ingesting: {source['url']}")
            video = extract_video_data(source["url"])
            analysis = analyze_with_gemini(client, video)
            entries.append({
                "id": video["id"],
                "date": format_date(video["upload_date"]),
                "reporter": source["reporter"],
                "location": source["location"],
                "severity": analysis.get("severity", 3),
                "headline": analysis.get("headline", video["title"]),
                "body": analysis.get("body", ""),
                "official": analysis.get("official_quote") or "No direct official statement matches this specific incident.",
                "officialSource": analysis.get("official_source") or "Public Record",
                "missing": analysis.get("missing", [])
            })
            time.sleep(SECONDS_BETWEEN_CALLS)
        except Exception as e:
            print(f"Error: {e}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)

if __name__ == "__main__":
    build_news_feed()
