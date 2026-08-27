"""
Pull Netflix TV shows from TMDB, with details, keywords and reviews.

Setup:
    pip install requests python-dotenv
    Create a .env file next to this script containing:
        TMDB_TOKEN=your_v4_read_access_token

Run:
    python fetch_shows.py
"""

import csv
import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

# .env lives in the project root, one level up from this script. Resolving the
# path explicitly (rather than a bare load_dotenv()) means the script finds the
# token no matter which directory you run it from, and there is exactly one
# copy of the file to update when the token is rotated.
HERE = Path(__file__).resolve().parent
load_dotenv(HERE.parent / ".env")

TOKEN = os.getenv("TMDB_TOKEN")
if not TOKEN:
    raise SystemExit("No TMDB_TOKEN found. Create a .env file with your token.")

BASE = "https://api.themoviedb.org/3"
HEADERS = {"Authorization": f"Bearer {TOKEN}", "accept": "application/json"}

NETFLIX_PROVIDER_ID = 8
REGION = "GB"          # change to US, etc.
TARGET_SHOWS = 500
PAUSE = 0.25           # seconds between requests — be polite


def get(path, **params):
    """One GET request against the TMDB API."""
    response = requests.get(f"{BASE}{path}", headers=HEADERS, params=params, timeout=20)
    response.raise_for_status()
    time.sleep(PAUSE)
    return response.json()


def discover_show_ids(target):
    """Page through Netflix TV shows, most popular first, collecting IDs."""
    ids, page = [], 1

    while len(ids) < target:
        data = get(
            "/discover/tv",
            with_watch_providers=NETFLIX_PROVIDER_ID,
            watch_region=REGION,
            sort_by="popularity.desc",
            page=page,
        )

        results = data.get("results", [])
        if not results:
            break

        ids.extend(show["id"] for show in results)
        print(f"  page {page}: {len(ids)} shows so far")

        if page >= data.get("total_pages", 1):
            break
        page += 1

    return ids[:target]


# Everything we bundle onto the /tv/{id} request. append_to_response lets TMDB
# attach sub-resources to one response instead of us making a request each —
# seven requests become one, which matters over 500 shows.
#
#   reviews, keywords, content_ratings  — as before
#   external_ids       — IDs on other databases; a stable join key
#   similar            — TMDB's own similarity, the ground truth proxy for §9.2
#   recommendations    — TMDB's behavioural recommendations, a second proxy
#   aggregate_credits  — full cast/crew across all seasons (cast overlap is a
#                        cheap similarity signal and grounds the ensemble axis)
APPEND = ",".join([
    "reviews",
    "keywords",
    "content_ratings",
    "external_ids",
    "similar",
    "recommendations",
    "aggregate_credits",
])


def fetch_show(show_id):
    """Fetch one show with all sub-resources bundled into a single request."""
    return get(f"/tv/{show_id}", append_to_response=APPEND)


def main():
    print("Discovering shows...")
    show_ids = discover_show_ids(TARGET_SHOWS)
    print(f"Found {len(show_ids)} show IDs\n")

    shows = []
    for index, show_id in enumerate(show_ids, start=1):
        try:
            shows.append(fetch_show(show_id))
        except requests.HTTPError as error:
            print(f"  skipped {show_id}: {error}")
            continue

        if index % 25 == 0:
            print(f"  fetched {index}/{len(show_ids)}")

    # Full data, for the tagging pipeline later
    with open(HERE / "shows_raw.json", "w", encoding="utf-8") as handle:
        json.dump(shows, handle, indent=2, ensure_ascii=False)

    # Flat summary, for eyeballing in a spreadsheet
    with open(HERE / "shows.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "id", "name", "first_air_date", "genres",
            "episodes", "vote_average", "vote_count",
            "review_count", "overview",
        ])

        for show in shows:
            writer.writerow([
                show["id"],
                show.get("name", ""),
                show.get("first_air_date", ""),
                "|".join(genre["name"] for genre in show.get("genres", [])),
                show.get("number_of_episodes", 0),
                show.get("vote_average", 0),
                show.get("vote_count", 0),
                len(show.get("reviews", {}).get("results", [])),
                (show.get("overview", "") or "").replace("\n", " "),
            ])

    # The number that actually decides your project's direction
    with_reviews = sum(
        1 for show in shows if show.get("reviews", {}).get("results")
    )
    total_reviews = sum(
        len(show.get("reviews", {}).get("results", [])) for show in shows
    )

    print(f"\nSaved {len(shows)} shows to shows_raw.json and shows.csv")
    print(f"Shows with at least one review: {with_reviews}/{len(shows)}")
    print(f"Total reviews retrieved: {total_reviews}")


if __name__ == "__main__":
    main()
