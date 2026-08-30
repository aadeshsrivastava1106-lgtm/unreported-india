#!/usr/bin/env python3
"""
UNREPORTED INDIA - scraper.py

Pulls the latest uploads from a curated list of independent, on-ground
Indian video journalism channels using YouTube's public RSS feeds. This
deliberately avoids yt-dlp and watch-page scraping: those require
rendering a real browser session and are the first thing YouTube's
anti-bot systems flag on a cloud IP (like a GitHub Actions runner). The
RSS feed below is a plain, unauthenticated XML endpoint meant to be
polled by machines, so it runs cleanly in CI with zero blocks.

Each new video's title and description are sent to Gemini 2.5 Flash in
a single batched call, which scores field-level public interest
urgency, tags a category, and writes a short summary. Results are
merged into ./news_feed.json, which index.html reads directly.

Environment variable required:
    GEMINI_API_KEY   free key from https://aistudio.google.com/apikey

Run locally:
    pip install requests feedparser google-genai
    GEMINI_API_KEY=your_key python scraper.py
"""

import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone

import feedparser
import requests
from google import genai
from google.genai import types

# --------------------------------------------------------------------------
# 1. SEED LIST - independent / on-ground channels to track.
#
#    These three are real, verified channel IDs:
#      - Peek TV: India's short-form ground-report news network
#      - Khabar Lahariya: all-women rural news team, Uttar Pradesh & Madhya
#        Pradesh (central India)
#      - Gaon Connection: pan-India rural affairs and ground reporting
#
#    To add more, open the channel's page, view page source, and search
#    for "channel_id" (or use any channel-ID lookup tool). The @handle
#    alone will not work in the RSS URL below - it needs the UC... ID.
# --------------------------------------------------------------------------
SEED_CHANNELS = [
    {"id": "UCF_kWxafJ5Nr0OLd9aoCHsQ", "name": "Peek TV"},
    {"id": "UCbvNC1RcIdlM2Kzn-QnjFng", "name": "Khabar Lahariya"},
    {"id": "UCy4rsUzwespkNvjMI1coX8A", "name": "Gaon Connection"},
    # {"id": "UCxxxxxxxxxxxxxxxxxxxxxx", "name": "Add another channel here"},
]

RSS_URL_TEMPLATE = "https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; UnreportedIndiaBot/1.0)"}

FEED_PATH = "./news_feed.json"
GEMINI_MODEL = "gemini-2.5-flash"
LOOKBACK_HOURS = 96   # only score videos uploaded in the last 4 days
PRUNE_DAYS = 10       # drop alerts from the saved feed once they're this old
MAX_ITEMS = 80        # cap on how many alerts the feed file holds
BATCH_SIZE = 15       # videos per Gemini call - keeps prompts small and free-tier friendly


def fetch_channel_entries(channel):
    """Fetch and parse one channel's public RSS/Atom feed."""
    url = RSS_URL_TEMPLATE.format(channel_id=channel["id"])
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"  [skip] {channel['name']}: fetch failed ({exc})")
        return []

    parsed = feedparser.parse(resp.content)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)
    entries = []

    for entry in parsed.entries:
        video_id = entry.get("yt_videoid") or entry.get("id", "").split(":")[-1]
        published_struct = entry.get("published_parsed")
        if not video_id or not published_struct:
            continue

        published = datetime(*published_struct[:6], tzinfo=timezone.utc)
        if published < cutoff:
            continue

        description = entry.get("summary", "") or entry.get("media_description", "")
        thumbnail_url = ""
        if entry.get("media_thumbnail"):
            thumbnail_url = entry["media_thumbnail"][0].get("url", "")

        entries.append({
            "video_id": video_id,
            "title": entry.get("title", "").strip(),
            "description": description.strip(),
            "channel_name": channel["name"],
            "channel_id": channel["id"],
            "published": published.isoformat(),
            "video_url": entry.get("link", f"https://www.youtube.com/watch?v={video_id}"),
            "thumbnail_url": thumbnail_url,
        })

    return entries


def load_existing_feed():
    if not os.path.exists(FEED_PATH):
        return {}
    try:
        with open(FEED_PATH, "r", encoding="utf-8") as f:
            items = json.load(f)
        return {item["video_id"]: item for item in items}
    except (json.JSONDecodeError, OSError, KeyError):
        return {}


def strip_markdown_fences(raw_text):
    """Sanitize Gemini output: strip ```json / ``` wrappers before parsing."""
    cleaned = (raw_text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def build_prompt(batch):
    payload = [
        {"video_id": e["video_id"], "title": e["title"], "description": e["description"][:600]}
        for e in batch
    ]
    return f"""You are a field-intelligence editor for an underreported-news alert desk covering rural India.

For each video below, judge ONLY from its title and description whether it
reports a real, field-level public interest issue: rural distress, human
rights violations, caste or gender violence, land or forest rights
conflict, police or administrative overreach, health or infrastructure
crises, farmer or livelihood issues, environmental disasters, or similar
on-ground stories. Studio debates, promos, entertainment, and routine
political coverage with no ground angle should be marked not relevant.

Videos:
{json.dumps(payload, ensure_ascii=False)}

Return ONLY a raw JSON array, one object per video, in exactly this shape,
with no markdown fences and no text outside the array:
[
  {{
    "video_id": "same id as input",
    "is_relevant": true or false,
    "urgency_score": integer 1-10 (10 = life-threatening or rights-violating and time-critical, 1 = minor or routine),
    "category": one of ["Land & Forest Rights", "Caste & Gender Violence", "Police & State Action", "Health Crisis", "Environmental Disaster", "Farmer & Livelihood", "Infrastructure Neglect", "Other"],
    "summary": "punchy one-sentence summary in English, 20 words or fewer",
    "locals_say": "short paraphrased line capturing what affected locals or the reporter are conveying, 18 words or fewer"
  }}
]"""


def score_with_gemini(client, entries):
    """Send entries to Gemini in batches, return {video_id: assessment}."""
    assessments = {}
    for i in range(0, len(entries), BATCH_SIZE):
        batch = entries[i:i + BATCH_SIZE]
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=build_prompt(batch),
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    response_mime_type="application/json",
                ),
            )
            cleaned = strip_markdown_fences(response.text)
            for result in json.loads(cleaned or "[]"):
                if "video_id" in result:
                    assessments[result["video_id"]] = result
        except Exception as exc:
            print(f"  [warn] Gemini batch {i // BATCH_SIZE + 1} failed: {exc}")
        time.sleep(1)  # stay comfortably inside the free-tier rate limit
    return assessments


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY is not set. Add it as a repo secret.", file=sys.stderr)
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    existing = load_existing_feed()
    print(f"Loaded {len(existing)} existing alerts from {FEED_PATH}")

    fresh_entries = []
    for channel in SEED_CHANNELS:
        print(f"Checking {channel['name']}...")
        found = fetch_channel_entries(channel)
        new = [e for e in found if e["video_id"] not in existing]
        print(f"  {len(found)} recent uploads, {len(new)} new")
        fresh_entries.extend(new)

    if fresh_entries:
        print(f"Scoring {len(fresh_entries)} new videos with {GEMINI_MODEL}...")
        assessments = score_with_gemini(client, fresh_entries)

        kept = 0
        for entry in fresh_entries:
            verdict = assessments.get(entry["video_id"])
            if not verdict or not verdict.get("is_relevant", False):
                continue
            entry["urgency_score"] = max(1, min(10, int(verdict.get("urgency_score", 5))))
            entry["category"] = verdict.get("category", "Other")
            entry["summary"] = verdict.get("summary", entry["title"])
            entry["locals_say"] = verdict.get("locals_say", "")
            entry["scraped_at"] = datetime.now(timezone.utc).isoformat()
            existing[entry["video_id"]] = entry
            kept += 1
        print(f"Kept {kept} of {len(fresh_entries)} new videos as public-interest alerts")
    else:
        print("No new uploads found across tracked channels.")

    # Prune old alerts and cap the feed size
    cutoff = datetime.now(timezone.utc) - timedelta(days=PRUNE_DAYS)
    combined = [
        item for item in existing.values()
        if datetime.fromisoformat(item["published"]) >= cutoff
    ]
    combined.sort(key=lambda x: (x.get("urgency_score", 0), x["published"]), reverse=True)
    combined = combined[:MAX_ITEMS]

    with open(FEED_PATH, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(combined)} alerts to {FEED_PATH}")


if __name__ == "__main__":
    main()
