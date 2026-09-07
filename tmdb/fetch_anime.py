"""
Widen the anime catalogue beyond Netflix GB.

Reads   tmdb/shows_raw.json, tmdb/movies_raw.json
Writes  the same two files, with anime titles added

WHY THIS BREAKS A RULE THE PROJECT SET ITSELF

Everything else here is the Netflix GB listing: `/discover` filtered by
`with_watch_providers=8` and `watch_region=GB`. That was a deliberate scope -
recommending something nobody can watch is a poor recommendation - and it is
what the report's catalogue figures describe.

Netflix GB carries about 200 anime series. That is thin for a whole tab: query
almost any of them and the same few dozen titles come back, because there is
nothing else in the space to find. So this script fetches anime regardless of
where it streams.

The provenance is kept rather than blurred. Every record now carries
`on_netflix_gb`, true for everything the original sweeps found and false for
anything added here, so:

  * the report can still state exactly what the Netflix GB catalogue contains
  * the evaluation can be re-run over either set
  * the interface can tell someone a recommendation is not on Netflix

HOW ANIME IS FOUND

TMDB has no anime flag, so this uses the two things it does record: the
Animation genre (16 for both films and television) and Japanese as the original
language. That is the same test build_space.is_anime applies, minus the keyword
half - a keyword search cannot be expressed as a discover filter, and titles
that only match the keyword are picked up anyway once is_anime runs over them.

Run:  venv\\Scripts\\python tmdb/fetch_anime.py
      venv\\Scripts\\python tmdb/fetch_anime.py --pages 20
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from fetch_shows import WORKERS, get, trim as trim_show  # noqa: E402
from fetch_movies import APPEND as MOVIE_APPEND  # noqa: E402
from fetch_movies import trim as trim_movie  # noqa: E402

ANIMATION_GENRE = 16
JAPANESE = "ja"

# Sub-resources for television, matching fetch_shows.
SHOW_APPEND = ",".join([
    "reviews", "keywords", "content_ratings", "external_ids",
    "similar", "recommendations", "aggregate_credits",
])

# TMDB serves 20 per page and refuses page numbers above 500.
PER_PAGE = 20
DEFAULT_PAGES = 30


def discover(path, pages):
    """Anime ids from one endpoint, in popularity order."""
    ids = []
    for page in range(1, pages + 1):
        try:
            payload = get(path, with_genres=ANIMATION_GENRE,
                          with_original_language=JAPANESE,
                          sort_by="popularity.desc", page=page)
        except requests.RequestException as error:
            print(f"    page {page} failed: {error.__class__.__name__}")
            continue
        results = payload.get("results", [])
        ids.extend(item["id"] for item in results)
        if page >= payload.get("total_pages", 1) or not results:
            break
    return ids


def fetch_many(ids, fetch_one, label):
    """Detail requests in parallel, skipping whatever fails."""
    out, done, failed = [], 0, 0
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(fetch_one, i): i for i in ids}
        for future in as_completed(futures):
            done += 1
            try:
                out.append(future.result())
            except requests.RequestException:
                failed += 1
            if done % 250 == 0:
                print(f"  {label} {done}/{len(ids)} "
                      f"({done / (time.monotonic() - started):.0f}/s)")
    return out, failed


def merge(path, existing, additions, label):
    """
    Add new records to a raw file, keeping provenance.

    Anything already present keeps its own record and is marked as being on
    Netflix GB. Additions are appended after it, so the popularity ordering the
    interface relies on still starts with the Netflix catalogue.
    """
    for record in existing:
        record.setdefault("on_netflix_gb", True)

    known = {record["id"] for record in existing}
    fresh = []
    for record in additions:
        if record["id"] in known:
            continue
        record["on_netflix_gb"] = False
        known.add(record["id"])
        fresh.append(record)

    combined = existing + fresh
    path.write_text(json.dumps(combined, ensure_ascii=False), encoding="utf-8")
    print(f"{label}: {len(existing)} kept + {len(fresh)} new = {len(combined)}")
    return len(fresh)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int, default=DEFAULT_PAGES,
                        help=f"pages per endpoint, {PER_PAGE} titles each "
                             f"(default {DEFAULT_PAGES})")
    args = parser.parse_args()

    started = time.monotonic()

    print(f"Discovering anime series ({args.pages} pages)...")
    show_ids = discover("/discover/tv", args.pages)
    print(f"  {len(show_ids)} ids")

    print(f"Discovering anime films ({args.pages} pages)...")
    movie_ids = discover("/discover/movie", args.pages)
    print(f"  {len(movie_ids)} ids\n")

    shows_path = HERE / "shows_raw.json"
    movies_path = HERE / "movies_raw.json"
    shows = json.loads(shows_path.read_text(encoding="utf-8"))
    movies = json.loads(movies_path.read_text(encoding="utf-8"))

    # Only fetch details for ids the files do not already hold.
    have_shows = {s["id"] for s in shows}
    have_movies = {m["id"] for m in movies}
    show_ids = [i for i in show_ids if i not in have_shows]
    movie_ids = [i for i in movie_ids if i not in have_movies]
    print(f"New to fetch: {len(show_ids)} series, {len(movie_ids)} films\n")

    new_shows, show_failures = fetch_many(
        show_ids, lambda i: trim_show(get(f"/tv/{i}", append_to_response=SHOW_APPEND)),
        "series")
    new_movies, movie_failures = fetch_many(
        movie_ids, lambda i: trim_movie(get(f"/movie/{i}", append_to_response=MOVIE_APPEND)),
        "films")

    print()
    added_shows = merge(shows_path, shows, new_shows, "Series")
    added_movies = merge(movies_path, movies, new_movies, "Films")

    print(f"\nAdded {added_shows + added_movies} titles in "
          f"{(time.monotonic() - started) / 60:.1f} min "
          f"({show_failures + movie_failures} failed)")
    print("\nNow rebuild, or the app will not see any of it:")
    print("  venv\\Scripts\\python app/build_space.py")
    print("  venv\\Scripts\\python app/build_movie_space.py")


if __name__ == "__main__":
    main()
