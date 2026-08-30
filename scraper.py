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

# ----------------------------------------------------------------------------
# Hardcoded inputs
# ----------------------------------------------------------------------------

# Real video URLs covering the Balaghat / Birsa block child-death crisis, as
# supplied by the operator. `reporter` holds the outlet/channel name (not a
# byline, since yt-dlp can't infer that); `location` reflects the block most
# consistently named in press coverage of this story. Sanity-check each link
# before treating this as production data — see module docstring.
INDEPENDENT_SOURCES = [
    {
        "url": "https://www.youtube.com/watch?v=VwtqS6LuVcc",
        "reporter": "TV9 MP",
        "location": "Birsa block, Balaghat district",
    },
    {
        "url": "https://www.youtube.com/watch?v=rBoB_k_Ur38",
        "reporter": "PTI News",
        "location": "Birsa block, Balaghat district",
    },
    {
        "url": "https://www.youtube.com/watch?v=d24xylXkqe4",
        "reporter": "Ground Briefing",
        "location": "Birsa block, Balaghat district",
    },
    {
        "url": "https://www.youtube.com/watch?v=WLPdtPy-G20",
        "reporter": "News18 MP",
        "location": "Birsa block, Balaghat district",
    },
    {
        "url": "https://www.youtube.com/watch?v=2sbg56kP-Tk",
        "reporter": "Field Interview",
        "location": "Birsa block, Balaghat district",
    },
]

# A snapshot of the public record, attributed by source and dated, because
# the toll is actively disputed rather than settled. The whole string is
# handed to Gemini for every story; the model finds the entry (if any) that
# applies, the same way a human fact-checker would scan a folder of press
# notes and public statements for the relevant one.
OFFICIAL_PRESS_RELEASES = """
[Birsa block, Balaghat district — as of 24 Aug] District administration /
Health Department (official position): Confirms 8 child deaths since late
June linked to a seasonal outbreak of malaria, measles, and related illness
in Baiga-dominated villages. A door-to-door screening of 3,811 people across
three villages (Kundeksa, Korka, Bondari) found 798 cases of malaria,
diarrhoea, fever-with-rash, or skin disease. District CMHO Dr. Paresh Uplap
has said medical teams — including ICMR-NIV and state disease-surveillance
staff — have camped in the affected villages.

Opposition / political counter-claims (not official government figures,
listed separately because they conflict with the administration's count):
  - Congress sources: 9 deaths claimed, mid-August.
  - MP Leader of Opposition Umang Singhar: 19 deaths claimed.
  - Adivasi Congress chairman Vikrant Bhuria: 22 deaths claimed, attributed
    to malnutrition-related illness; also alleges a 6-month lapse in
    nutritional meal supply, a health centre 20 km from affected villages,
    Forest Rights Act land-title cancellations, and vaccination coverage
    below 80% in the area.
  - Local tribal representatives (Sarv Adivasi Samaj): 24 deaths claimed.

NOTE: This toll is actively disputed and still developing. Treat every
figure above as a time-stamped claim from a named source, not settled fact
— update this block as the story develops rather than relying on this
snapshot indefinitely.
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


def compose_metadata_fallback(info):
    """Build a text payload from title/description/metadata when no
    transcript is available, so the video still contributes a signal to
    Gemini instead of being dropped."""
    bits = []
    if info.get("uploader"):
        bits.append(f"Channel: {info['uploader']}")
    if info.get("upload_date"):
        bits.append(f"Uploaded: {info['upload_date']}")
    if info.get("duration"):
        bits.append(f"Duration: {info['duration']}s")
    if info.get("view_count") is not None:
        bits.append(f"Views: {info['view_count']}")
    tags = info.get("tags") or []
    if tags:
        bits.append("Tags: " + ", ".join(tags[:12]))

    if not bits:
        return "No captions or extended metadata were available for this video."
    return "No captions were available for this video. Falling back to metadata — " + "; ".join(bits) + "."


def extract_video_data(url):
    """Return {id, title, description, upload_date, transcript, sourceType}
    for a URL.

    Two-tier fallback:
      1. Try a full extract_info() call that also requests auto captions.
      2. If that call itself raises (network hiccup, restricted video,
         captions endpoint error, etc.), retry with a lighter metadata-only
         call so the video still contributes title/description/metadata
         instead of being lost entirely. A per-source try/except one level
         up in build_news_feed() is the final backstop if even that fails.
    """
    info = None
    transcript = ""

    try:
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
            if vtt_files:
                transcript = vtt_to_text(vtt_files[0])
    except (DownloadError, Exception) as exc:
        print(f"  caption fetch failed ({exc}); retrying metadata-only", file=sys.stderr)
        info = None

    # Fallback call: metadata only, no subtitle attempt. Runs whenever the
    # caption-including call above failed outright.
    if info is None:
        ydl_opts = {"skip_download": True, "quiet": True, "no_warnings": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

    if transcript:
        source_type = "captions"
        transcript_payload = transcript
    else:
        source_type = "metadata"
        transcript_payload = compose_metadata_fallback(info)

    return {
        "id": info.get("id", ""),
        "title": info.get("title", ""),
        "description": (info.get("description") or "")[:MAX_TRANSCRIPT_CHARS],
        "upload_date": info.get("upload_date"),  # "YYYYMMDD" or None
        "transcript": transcript_payload,
        "sourceType": source_type,
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
a block of official and political statements covering the same period.

Independent report
  Title: {title}
  Description: {description}
  Transcript: {transcript}

Official / political statements on record
  {official_context}

Task
  1. Find the one statement above (if any) that addresses the same event as
     the independent report. If none does, say so explicitly. If the
     transcript field says captions were unavailable and you're working
     from title/description/metadata only, be conservative — don't infer
     detail beyond what's actually stated.
  2. Write a one-line "headline" (<=15 words) and a two-sentence "body"
     summarizing what the independent report actually says.
  3. List 2-4 concrete, specific points in "missing" describing what the
     official position leaves out, understates, or contradicts, compared
     to the independent report or to other named sources above. Do not
     invent facts not present in the transcript/description or the
     official text — if you're not sure about a point, leave it out.
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
            print(f"  sourceType={video['sourceType']}", file=sys.stderr)
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
                "sourceType": video["sourceType"],
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
