"""
Pull Netflix TV shows from TMDB, with details, keywords and reviews.

Setup:
    pip install requests python-dotenv
    .env in the project root containing:
        TMDB_TOKEN=your_v4_read_access_token

Run:
    python tmdb/fetch_shows.py              # whole Netflix GB catalogue
    python tmdb/fetch_shows.py --count 500  # a smaller sample

Two things make this fast enough to fetch thousands of shows rather than
hundreds.

RATE
    TMDB permits 50 requests/second and 20 connections per IP. The first
    version of this script slept 0.25s between requests on a single thread -
    4 requests/second, twelve times under the ceiling, so a full catalogue
    fetch would have taken about fifteen minutes of pure waiting. It now runs
    12 workers against a shared limiter set to 30 requests/second (see
    rate_limit.py).

SIZE
    The raw /tv/{id} response is about 224 KB per show, and 3,538 shows of that
    is roughly 800 MB. Almost all of it is never read:

        aggregate_credits   75% of the file, used nowhere in the project
        recommendations     10%, of which only the ids are used
        similar             10%, likewise

    So responses are trimmed on the way in - see `trim()`. That takes a show
    from 224 KB to about 4 KB and the whole catalogue from ~800 MB to ~14 MB,
    which is the difference between a file you can load in a script and one you
    cannot.
"""

import argparse
import csv
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from tmdb.rate_limit import WORKERS, RateLimiter  # noqa: E402

# .env lives in the project root, one level up from this script. Resolving the
# path explicitly (rather than a bare load_dotenv()) means the script finds the
# token no matter which directory you run it from, and there is exactly one
# copy of the file to update when the token is rotated.
load_dotenv(HERE.parent / ".env")

TOKEN = os.getenv("TMDB_TOKEN")
if not TOKEN:
    raise SystemExit("No TMDB_TOKEN found. Create a .env file with your token.")

BASE = "https://api.themoviedb.org/3"
HEADERS = {"Authorization": f"Bearer {TOKEN}", "accept": "application/json"}

NETFLIX_PROVIDER_ID = 8
REGION = "GB"
MAX_RETRIES = 4

# Sub-resources bundled onto each /tv/{id} request. append_to_response attaches
# them to one response instead of costing a request each.
APPEND = ",".join([
    "reviews",
    "keywords",
    "content_ratings",
    "external_ids",
    "similar",
    "recommendations",
    "aggregate_credits",
])

# Certificates we care about. Storing every country's rating was a meaningful
# slice of the file for two we actually read.
KEEP_CERT_COUNTRIES = {"GB", "US"}

limiter = RateLimiter()
_session = threading.local()


def session():
    """
    One requests.Session per thread.

    Sessions are not thread-safe, but a session per thread keeps HTTP
    keep-alive, which matters more than it sounds: without it every request
    pays a fresh TLS handshake, and over several thousand requests that
    dominates the wall clock.
    """
    if not hasattr(_session, "s"):
        _session.s = requests.Session()
        _session.s.headers.update(HEADERS)
    return _session.s


def get(path, **params):
    """One GET against the TMDB API, rate-limited and retried."""
    for attempt in range(MAX_RETRIES):
        try:
            limiter.wait()
            response = session().get(f"{BASE}{path}", params=params, timeout=30)
            # 429 means we misjudged the rate. Honour the server's own retry
            # hint rather than guessing.
            if response.status_code == 429:
                wait = int(response.headers.get("retry-after", "2"))
                print(f"    rate limited, waiting {wait}s")
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response.json()
        except requests.RequestException as error:
            status = getattr(error.response, "status_code", None)
            # 4xx other than 429 is our fault and will fail identically.
            if status is not None and 400 <= status < 500 and status != 429:
                raise
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(2 ** attempt)


# Paging /discover is lossy. TMDB sorts the whole result set live, so as you
# work through 178 pages the ordering shifts underneath you: some shows appear
# on two pages and others are never shown at all. A single popularity.desc sweep
# of a 3,541-show catalogue returned only 2,815 unique ids - 21% missing.
#
# Sweeping several orderings and taking the union recovers most of them, because
# a show that drifts out of view under one ordering is usually stable under
# another. Ascending sorts matter most: the shows lost from popularity.desc are
# the unpopular ones, which an ascending sweep sees first.
# Measured, not assumed. Over the Netflix GB catalogue:
#
#     popularity.desc      2831 ids
#     popularity.asc      + 711  -> 3542  (100% of the 3541 TMDB claims)
#     first_air_date.desc +   0
#     vote_count.desc     +   0
#
# Two opposing sweeps of the same ordering are enough: whatever drifts out of
# view going down is near the front going up. The other orderings added nothing
# and cost 356 requests, so they are not run.
SORT_ORDERS = [
    "popularity.desc",
    "popularity.asc",
]


def _sweep(sort_by, target=None):
    """One full pagination pass over the catalogue in a given ordering."""
    first = get("/discover/tv", with_watch_providers=NETFLIX_PROVIDER_ID,
                watch_region=REGION, sort_by=sort_by, page=1)
    total_pages = first.get("total_pages", 1)
    ids = [s["id"] for s in first.get("results", [])]

    # Only page as far as the target needs. Asking for 40 shows should not cost
    # 178 discover requests. The 1.5x margin covers the duplicates above.
    if target:
        total_pages = min(total_pages, max(1, -(-int(target * 1.5) // 20)))

    # Pages are independent, so they parallelise cleanly.
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {
            pool.submit(get, "/discover/tv", with_watch_providers=NETFLIX_PROVIDER_ID,
                        watch_region=REGION, sort_by=sort_by, page=p): p
            for p in range(2, total_pages + 1)
        }
        for future in as_completed(futures):
            try:
                ids.extend(s["id"] for s in future.result().get("results", []))
            except requests.RequestException as error:
                print(f"    page {futures[future]} failed: {error.__class__.__name__}")

    return ids, first.get("total_results", 0)


def discover_show_ids(target=None):
    """
    Collect Netflix TV show ids, sweeping several orderings to beat drift.

    A small `target` uses one ordering - the extra sweeps exist to reach the
    long tail, and a caller asking for 40 shows wants the popular ones.
    """
    orders = SORT_ORDERS if target is None else SORT_ORDERS[:1]

    seen, claimed = {}, 0
    for sort_by in orders:
        ids, total = _sweep(sort_by, target)
        claimed = max(claimed, total)
        before = len(seen)
        # dict.fromkeys-style insert keeps first-seen order, so the popularity
        # sweep still determines the ordering of the final list.
        for show_id in ids:
            seen.setdefault(show_id, None)
        print(f"  {sort_by:20} +{len(seen) - before:>4} new  (running total {len(seen)})")

    ids = list(seen)
    if claimed:
        print(f"  {len(ids)} unique of {claimed} claimed by TMDB "
              f"({100 * len(ids) / claimed:.0f}% coverage)")
    return ids[:target] if target else ids


def trim(show):
    """
    Keep the fields the project actually reads, and drop the rest.

    Everything removed here is either unused (aggregate_credits) or stored more
    cheaply (similar and recommendations become id lists, since only the ids are
    ever read). See the module docstring for the numbers.
    """
    keywords = show.get("keywords", {}).get("results", [])
    credits = show.get("aggregate_credits", {}) or {}

    return {
        "id": show["id"],
        "name": show.get("name", ""),
        "original_name": show.get("original_name", ""),
        "first_air_date": show.get("first_air_date", ""),
        "overview": show.get("overview") or "",
        "poster_path": show.get("poster_path"),
        "genres": show.get("genres", []),
        "number_of_episodes": show.get("number_of_episodes") or 0,
        "number_of_seasons": show.get("number_of_seasons") or 0,
        "episode_run_time": show.get("episode_run_time") or [],
        "vote_average": show.get("vote_average") or 0,
        "vote_count": show.get("vote_count") or 0,
        "popularity": show.get("popularity") or 0,
        "type": show.get("type"),
        "status": show.get("status"),
        "original_language": show.get("original_language"),
        "origin_country": show.get("origin_country", []),
        "external_ids": show.get("external_ids", {}),

        # Season numbers and episode counts only - fetch_episodes.py needs the
        # numbers, not the season overviews and poster paths.
        "seasons": [
            {"season_number": s.get("season_number"),
             "episode_count": s.get("episode_count")}
            for s in show.get("seasons", [])
        ],

        "keywords": {"results": [{"name": k["name"]} for k in keywords]},

        "content_ratings": {"results": [
            r for r in show.get("content_ratings", {}).get("results", [])
            if r.get("iso_3166_1") in KEEP_CERT_COUNTRIES
        ]},

        # Reviews are small and are research data - report S4 is about how few
        # there are - so they are kept in full.
        "reviews": show.get("reviews", {}),

        # Only the ids are ever read, and each entry was a full show record.
        "similar": [x["id"] for x in show.get("similar", {}).get("results", [])],
        "recommendations": [
            x["id"] for x in show.get("recommendations", {}).get("results", [])
        ],

        # Not used today, but cast overlap is a plausible similarity signal and
        # re-fetching 3,500 shows to recover it would be expensive. Twelve names
        # costs about 400 bytes; the full credit list cost 108 KB.
        "top_cast": [
            {"id": c.get("id"), "name": c.get("name")}
            for c in (credits.get("cast") or [])[:12]
        ],
    }


def fetch_show(show_id):
    return trim(get(f"/tv/{show_id}", append_to_response=APPEND))


def write_outputs(shows):
    (HERE / "shows_raw.json").write_text(
        json.dumps(shows, ensure_ascii=False), encoding="utf-8"
    )

    with open(HERE / "shows.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "id", "name", "first_air_date", "genres", "episodes",
            "vote_average", "vote_count", "review_count", "overview",
        ])
        for show in shows:
            writer.writerow([
                show["id"], show["name"], show["first_air_date"],
                "|".join(g["name"] for g in show["genres"]),
                show["number_of_episodes"], show["vote_average"], show["vote_count"],
                len(show.get("reviews", {}).get("results", [])),
                show["overview"].replace("\n", " "),
            ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=None,
                        help="stop after N shows (default: the whole catalogue)")
    args = parser.parse_args()

    started = time.monotonic()
    print("Discovering shows...")
    show_ids = discover_show_ids(args.count)
    print(f"Fetching {len(show_ids)} shows with {WORKERS} workers\n")

    fetch_started = time.monotonic()
    shows, done, failed = [], 0, 0
    out = HERE / "shows_raw.json"

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(fetch_show, sid): sid for sid in show_ids}
        for future in as_completed(futures):
            done += 1
            try:
                shows.append(future.result())
            except requests.RequestException as error:
                failed += 1
                print(f"  skipped {futures[future]}: {error.__class__.__name__}")

            if done % 250 == 0:
                rate = done / (time.monotonic() - fetch_started)
                print(f"  {done}/{len(show_ids)}  ({rate:.1f} req/s)")
                # Checkpoint. A crash at show 3,400 should not cost the first 3,399.
                out.write_text(json.dumps(shows, ensure_ascii=False), encoding="utf-8")

    # Restore popularity order; as_completed returns whatever finishes first.
    order = {sid: i for i, sid in enumerate(show_ids)}
    shows.sort(key=lambda s: order.get(s["id"], 1 << 30))

    write_outputs(shows)

    elapsed = time.monotonic() - started
    size_mb = out.stat().st_size / 1e6
    with_reviews = sum(1 for s in shows if s.get("reviews", {}).get("results"))
    total_reviews = sum(len(s.get("reviews", {}).get("results", [])) for s in shows)

    print(f"\nSaved {len(shows)} shows ({failed} failed) in {elapsed/60:.1f} min "
          f"at {done/elapsed:.1f} req/s")
    print(f"  shows_raw.json: {size_mb:.1f} MB ({1000*size_mb/max(len(shows),1):.1f} KB per show)")
    print(f"  with at least one review: {with_reviews}/{len(shows)}")
    print(f"  total reviews retrieved: {total_reviews}")


if __name__ == "__main__":
    main()
