"""
Fetch every episode of every show in shows_raw.json.

Why this exists
---------------
The series-level record gives one overview (~150 characters) and one average
rating per show. That is far too little to ground the Features block of the
schema. The episode level gives, across the 500-show catalogue, roughly 45,000
overviews and 45,000 independent rating datapoints.

Per-episode ratings measure things a series average structurally cannot:

  slow_burn   - the slope of vote_average across a season. A slow burn opens
                low and climbs.
  serialised  - rating variance, and how far the finale sits above the mean.
                Episodic shows are flat; serialised shows spike at finales.
  intensity   - vote_count spikes mark the episodes people turned up for.

Run:
    python tmdb/fetch_episodes.py
"""

import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
load_dotenv(HERE.parent / ".env")

TOKEN = os.getenv("TMDB_TOKEN")
if not TOKEN:
    raise SystemExit("No TMDB_TOKEN found. Create a .env file with your token.")

BASE = "https://api.themoviedb.org/3"
HEADERS = {"Authorization": f"Bearer {TOKEN}", "accept": "application/json"}
PAUSE = 0.25

# TMDB refuses more than 20 sub-resources on one append_to_response. Shows with
# more seasons than this are fetched over several requests.
MAX_APPEND = 20


def get(path, **params):
    """One GET against the TMDB API, with a polite pause afterwards."""
    response = requests.get(f"{BASE}{path}", headers=HEADERS, params=params, timeout=30)
    response.raise_for_status()
    time.sleep(PAUSE)
    return response.json()


def chunked(items, size):
    """Split a list into consecutive chunks of at most `size`."""
    for start in range(0, len(items), size):
        yield items[start:start + size]


def fetch_episodes_for_show(show_id, season_numbers):
    """
    Fetch all episodes of one show.

    Rather than one request per season, we append up to 20 seasons onto a single
    /tv/{id} request. A 3-season show costs 1 request instead of 3; a 56-season
    show costs 3 instead of 56.
    """
    episodes = []

    for chunk in chunked(season_numbers, MAX_APPEND):
        append = ",".join(f"season/{number}" for number in chunk)
        payload = get(f"/tv/{show_id}", append_to_response=append)

        for number in chunk:
            season = payload.get(f"season/{number}")
            if not season:
                continue
            for episode in season.get("episodes", []):
                # Keep only the fields we will actually use. The full episode
                # record carries stills, crew and guest stars for every episode,
                # which would balloon the file for no benefit.
                episodes.append({
                    "show_id": show_id,
                    "season_number": episode.get("season_number"),
                    "episode_number": episode.get("episode_number"),
                    "name": episode.get("name"),
                    "overview": episode.get("overview") or "",
                    "air_date": episode.get("air_date"),
                    "runtime": episode.get("runtime"),
                    "vote_average": episode.get("vote_average"),
                    "vote_count": episode.get("vote_count"),
                    "episode_type": episode.get("episode_type"),
                    "guest_star_count": len(episode.get("guest_stars", [])),
                })

    return episodes


def main():
    shows = json.loads((HERE / "shows_raw.json").read_text(encoding="utf-8"))
    print(f"Loaded {len(shows)} shows")

    all_episodes = []

    for index, show in enumerate(shows, start=1):
        # Season 0 is TMDB's "Specials" bucket - out-of-continuity extras that
        # would distort any pacing measure, so it is excluded.
        season_numbers = [
            season["season_number"]
            for season in show.get("seasons", [])
            if season.get("season_number", 0) > 0
        ]
        if not season_numbers:
            continue

        try:
            episodes = fetch_episodes_for_show(show["id"], season_numbers)
        except requests.HTTPError as error:
            print(f"  skipped {show.get('name')}: {error}")
            continue

        all_episodes.extend(episodes)

        if index % 25 == 0:
            print(f"  {index}/{len(shows)} shows - {len(all_episodes)} episodes so far")

    out = HERE / "episodes.json"
    out.write_text(json.dumps(all_episodes, ensure_ascii=False), encoding="utf-8")

    with_overview = sum(1 for e in all_episodes if e["overview"])
    with_rating = sum(1 for e in all_episodes if (e["vote_count"] or 0) > 0)
    words = sum(len(e["overview"]) for e in all_episodes) // 5

    print(f"\nSaved {len(all_episodes)} episodes to {out.name}")
    print(f"  with an overview: {with_overview} ({with_overview / max(len(all_episodes), 1):.0%})")
    print(f"  with >=1 rating vote: {with_rating} ({with_rating / max(len(all_episodes), 1):.0%})")
    print(f"  episode-overview corpus: ~{words:,} words")


if __name__ == "__main__":
    main()
