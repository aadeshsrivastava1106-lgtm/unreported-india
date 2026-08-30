"""
scraper.py — zero-cost ingestion pipeline for Unreported India.

Pipeline
  1. Pull title, description, and an auto-generated transcript for each URL
     in INDEPENDENT_SOURCES using yt-dlp. No API key, no paid quota, no
     audio download — only the small subtitle track is fetched.
  2. Send each transcript to Gemini 2.5 Flash (free tier) alongside the
     hardcoded OFFICIAL_PRESS_RELEASES string and ask it to (a) match the
     story to the relevant official statement, if any, and (b) list
     concrete discrepancies / missing context.
  3. Write everything to news_feed.json in the shape UnreportedIndia.jsx
     expects (id, date, reporter, location, severity, headline, body,
     official, officialSource, missing[]).

IMPORTANT — placeholder data
  INDEPENDENT_SOURCES below contains placeholder video URLs and
  OFFICIAL_PRESS_RELEASES contains the same fabricated demo statements used
  in the front-end mockup. Replace both with real, sourced material before
  using this pipeline's output as anything other than a prototype.

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
from google import genai
from google.genai import types

# ----------------------------------------------------------------------------
# Hardcoded inputs — replace with real data for real use
# ----------------------------------------------------------------------------

# TODO: replace with real independent-journalism video/Shorts URLs.
# yt-dlp can't infer `reporter` or `location` — attach that metadata
# yourself, since it's how each story gets matched to an official release.
INDEPENDENT_SOURCES = [
    {
        "url": "https://www.youtube.com/watch?v=REPLACE_WITH_VIDEO_ID_1",
        "reporter": "R. Uikey",
        "location": "Baihar block",
    },
    {
        "url": "https://www.youtube.com/watch?v=REPLACE_WITH_VIDEO_ID_2",
        "reporter": "S. Netam",
        "location": "Lanji block",
    },
    {
        "url": "https://www.youtube.com/shorts/REPLACE_WITH_VIDEO_ID_3",
        "reporter": "A. Mehra",
        "location": "Paraswada block",
    },
    {
        "url": "https://www.youtube.com/watch?v=REPLACE_WITH_VIDEO_ID_4",
        "reporter": "S. Netam",
        "location": "Kirnapur block",
    },
]

# One hardcoded block of official statements. The whole string is handed to
# Gemini for every story; the model is responsible for finding the entry (if
# any) that applies, the same way a human fact-checker would scan a folder
# of press notes for the relevant one. This is fabricated demo text — see
# module docstring.
OFFICIAL_PRESS_RELEASES = """
[Baihar block — 14 Sep] District Health Dept.: "An isolated case of seasonal
fever was reported and has been immediately contained. There is no cause for
public concern."

[Lanji block — 18 Sep] District Health Dept.: "The vector control drive has
been completed across all blocks of the district as per schedule."

[Baihar block — 24 Sep] District Education Office: "Seasonal absenteeism is
being monitored; attendance remains within normal range for the quarter."

[Kirnapur block — 27 Sep] District Health Dept.: "Adequate stock of essential
medicines is being maintained at all health facilities."

[Balaghat district — 3 Oct] State Health Dept.: "A routine seasonal health
campaign has been rolled out across the district as per the annual calendar."

No public release in this period addresses language access for non-Hindi-
speaking patients at treatment camps.
""".strip()

GEMINI_MODEL = "gemini-2.5-flash"
OUTPUT_PATH = "news_feed.json"
MAX_TRANSCRIPT_CHARS = 6000
SECONDS_BETWEEN_CALLS = 2  # be polite to the free-tier rate limit

TIMESTAMP_RE = re.compile(r"^\d{2}:\d{2}:\d{2}\.\d{3}\s*-->")
TAG_RE = re.compile(r"<[^>]+>")


# ----------------------------------------------------------------------------
# yt-dlp extraction
# ----------------------------------------------------------------------------

def vtt_to_text(path):
    """Turn a .vtt caption file into plain, deduplicated text."""
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


def extract_video_data(url):
    """Return {id, title, description, upload_date, transcript} for a URL."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ydl_opts = {
            "skip_download": True,       # never fetch the actual video
            "writeautomaticsub": True,   # only the small caption track
            "subtitleslangs": ["en"],
            "subtitlesformat": "vtt",
            "outtmpl": os.path.join(tmpdir, "%(id)s.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

        vtt_files = glob.glob(os.path.join(tmpdir, f"{info['id']}*.vtt"))
        transcript = vtt_to_text(vtt_files[0]) if vtt_files else ""

        return {
            "id": info.get("id", ""),
            "title": info.get("title", ""),
            "description": (info.get("description") or "")[:MAX_TRANSCRIPT_CHARS],
            "upload_date": info.get("upload_date"),  # "YYYYMMDD" or None
            "transcript": transcript,
        }


# ----------------------------------------------------------------------------
# Gemini analysis
# ----------------------------------------------------------------------------

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "body": {"type": "string"},
        "official_quote": {"type": ["string", "null"]},
        "official_source": {"type": ["string", "null"]},
        "missing": {"type": "array", "items": {"type": "string"}},
        "severity": {"type": "integer"},
    },
    "required": ["headline", "body", "missing", "severity"],
}

ANALYSIS_PROMPT = """You are a fact-checking assistant for an accountability
journalism dashboard. You are given (1) an independent video report and (2)
a block of official government press statements covering the same period.

Independent report
  Title: {title}
  Description: {description}
  Transcript: {transcript}

Official statements on record
  {official_context}

Task
  1. Find the one official statement above (if any) that addresses the same
     event as the independent report. If none does, say so explicitly.
  2. Write a one-line "headline" (<=15 words) and a two-sentence "body"
     summarizing what the independent report actually says.
  3. List 2-4 concrete, specific points in "missing" describing what the
     official statement leaves out, understates, or contradicts, compared
     to the independent report. Do not invent facts not present in the
     transcript/description or the official text — if you're not sure
     about a point, leave it out.
  4. Rate "severity" 1-5: how serious is the gap between the official
     account and the independent report (1 = trivial wording difference,
     5 = official account appears to substantially misrepresent events).

Respond with JSON only, matching this shape:
{{
  "headline": "...",
  "body": "...",
  "official_quote": "... or null if no matching statement exists",
  "official_source": "... or null",
  "missing": ["...", "..."],
  "severity": 1
}}
"""


def analyze_with_gemini(client, video):
    prompt = ANALYSIS_PROMPT.format(
        title=video["title"] or "(no title)",
        description=video["description"] or "(no description)",
        transcript=video["transcript"] or "(no transcript available)",
        official_context=OFFICIAL_PRESS_RELEASES,
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
            temperature=0.2,
        ),
    )

    text = (response.text or "").strip()
    text = re.sub(r"^```json|```$", "", text, flags=re.MULTILINE).strip()
    return json.loads(text)


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------

def format_date(upload_date):
    if not upload_date:
        return datetime.utcnow().strftime("%-d %b")
    try:
        return datetime.strptime(upload_date, "%Y%m%d").strftime("%-d %b")
    except ValueError:
        return datetime.utcnow().strftime("%-d %b")


def build_news_feed():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("GEMINI_API_KEY is not set. Export it or add it as a repo secret.")

    client = genai.Client(api_key=api_key)

    entries = []
    failures = 0

    for i, source in enumerate(INDEPENDENT_SOURCES, start=1):
        print(f"[{i}/{len(INDEPENDENT_SOURCES)}] fetching {source['url']}", file=sys.stderr)
        try:
            video = extract_video_data(source["url"])
            analysis = analyze_with_gemini(client, video)

            entries.append({
                "id": f"yt-{video['id']}" if video["id"] else f"story-{i}",
                "date": format_date(video["upload_date"]),
                "reporter": source["reporter"],
                "location": source["location"],
                "severity": max(1, min(5, int(analysis.get("severity", 3)))),
                "headline": analysis.get("headline", video["title"]),
                "body": analysis.get("body", ""),
                "official": analysis.get("official_quote") or "No matching official statement found on record.",
                "officialSource": analysis.get("official_source") or "N/A",
                "missing": analysis.get("missing", []),
            })
        except Exception as exc:  # keep one bad source from killing the run
            failures += 1
            print(f"  skipped ({exc})", file=sys.stderr)

        time.sleep(SECONDS_BETWEEN_CALLS)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(entries)} entries to {OUTPUT_PATH} ({failures} skipped).", file=sys.stderr)


if __name__ == "__main__":
    build_news_feed()
