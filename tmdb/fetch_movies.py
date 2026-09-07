"""
Fetch the Netflix GB film catalogue from TMDB.

Reads   nothing (TMDB_TOKEN from .env)
Writes  tmdb/movies_raw.json

TMDB keeps films and television in separate endpoints with different field
names, so this is a sibling of fetch_shows.py rather than a flag on it. It
borrows that file's plumbing - the rate limiter, the retrying `get`, the
threaded discovery sweep - because duplicating a rate limiter is how you end up
with two of them running at once against the same 50 req/s ceiling.

WHAT IS DIFFERENT FROM TELEVISION

  /discover/movie   instead of /discover/tv
  /movie/{id}       instead of /tv/{id}
  title             instead of name
  release_date      instead of first_air_date
  release_dates     instead of content_ratings, and nested differently
  keywords.keywords instead of keywords.results

The last two are the ones that bite. Films bury their certificate two levels
down inside a per-country list of release events, and TMDB uses a different key
for the keyword array on films than on shows for no reason I can find.

Everything written out is normalised into the SAME SHAPE as shows_raw.json -
`name`, `first_air_date`, `keywords.results`, `content_ratings.results` - so
build_space.py can read either without a second code path. The fields that only
make sense for one kind (runtime, belongs_to_collection) ride alongside, and a
`kind` field says which is which.
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

from fetch_shows import (  # noqa: E402
    KEEP_CERT_COUNTRIES,
    NETFLIX_PROVIDER_ID,
    REGION,
    WORKERS,
    get,
)

# Films have no episodes, so there is no second pipeline stage to feed. These
# are everything the feature space and the interface read.
APPEND = ",".join([
    "keywords",
    "release_dates",
    "similar",
    "recommendations",
    "credits",
])

# Several orderings, because TMDB's discover paging drifts as popularity
# updates underneath you and one sweep silently misses titles.
SORT_ORDERS = [
    "popularity.desc",
    "vote_count.desc",
    "primary_release_date.desc",
    "revenue.desc",
]


def certificates(movie):
    """
    Pull GB and US certificates out of TMDB's release_dates structure.

    For television a certificate is one field. For film it is buried inside a
    list of release events per country - theatrical, digital, physical - each
    with its own certification string, and often blank. Take the first
    non-empty one per country and present it in the same shape shows use.
    """
    out = []
    for country in movie.get("release_dates", {}).get("results", []):
        code = country.get("iso_3166_1")
        if code not in KEEP_CERT_COUNTRIES:
            continue
        for release in country.get("release_dates", []):
            rating = (release.get("certification") or "").strip()
            if rating:
                out.append({"iso_3166_1": code, "rating": rating})
                break
    return {"results": out}


def as_ids(payload):
    """TMDB's similar/recommendations blocks, reduced to bare ids."""
    return [item["id"] for item in (payload or {}).get("results", [])]


def trim(movie):
    """
    Keep what the project reads, in the same shape shows_raw.json uses.

    The full /movie response is roughly 40 KB with credits attached. Over a few
    thousand films that is most of a gigabyte for perhaps 2 KB of fields anyone
    ever looks at.
    """
    return {
        "kind": "movie",
        "id": movie["id"],
        # Normalised to the television field names so build_space.py needs no
        # second code path.
        "name": movie.get("title") or movie.get("original_title") or "",
        "first_air_date": movie.get("release_date") or "",
        "overview": movie.get("overview") or "",
        "poster_path": movie.get("poster_path"),
        "genres": movie.get("genres", []),
        "vote_average": movie.get("vote_average") or 0,
        "vote_count": movie.get("vote_count") or 0,
        "popularity": movie.get("popularity") or 0,
        "original_language": movie.get("original_language"),
        # Films carry production_countries rather than origin_country. Reduced
        # to the same list-of-codes shape, because the anime test reads it.
        "origin_country": [c["iso_3166_1"] for c in movie.get("production_countries", [])],
        "keywords": {"results": movie.get("keywords", {}).get("keywords", [])},
        "content_ratings": certificates(movie),
        "similar": as_ids(movie.get("similar")),
        "recommendations": as_ids(movie.get("recommendations")),
        # Film-only, and useful: runtime replaces episode length, and a
        # collection is the closest thing a film has to being part of a run.
        "runtime": movie.get("runtime") or 0,
        "collection": (movie.get("belongs_to_collection") or {}).get("name"),
        "cast_size": len(movie.get("credits", {}).get("cast", [])),
    }


def sweep(sort_by, target=None):
    """One full pagination pass over the film catalogue in a given ordering."""
    first = get("/discover/movie", with_watch_providers=NETFLIX_PROVIDER_ID,
                watch_region=REGION, sort_by=sort_by, page=1)
    total_pages = first.get("total_pages", 1)
    ids = [m["id"] for m in first.get("results", [])]

    if target:
        total_pages = min(total_pages, max(1, -(-int(target * 1.5) // 20)))
    # TMDB refuses page numbers above 500 whatever the result count says.
    total_pages = min(total_pages, 500)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {
            pool.submit(get, "/discover/movie", with_watch_providers=NETFLIX_PROVIDER_ID,
                        watch_region=REGION, sort_by=sort_by, page=page): page
            for page in range(2, total_pages + 1)
        }
        for future in as_completed(futures):
            try:
                ids.extend(m["id"] for m in future.result().get("results", []))
            except requests.RequestException as error:
                print(f"    page {futures[future]} failed: {error.__class__.__name__}")

    return ids, first.get("total_results", 0)


def discover_movie_ids(target=None):
    """Collect Netflix GB film ids across several orderings."""
    orders = SORT_ORDERS if target is None else SORT_ORDERS[:1]

    seen, claimed = {}, 0
    for sort_by in orders:
        ids, total = sweep(sort_by, target)
        claimed = max(claimed, total)
        before = len(seen)
        for movie_id in ids:
            seen.setdefault(movie_id, None)
        print(f"  {sort_by:26} +{len(seen) - before:>4} new  (running total {len(seen)})")

    ids = list(seen)
    if claimed:
        print(f"  {len(ids)} unique of {claimed} claimed by TMDB "
              f"({100 * len(ids) / claimed:.0f}% coverage)")
    return ids[:target] if target else ids


def fetch_movie(movie_id):
    return trim(get(f"/movie/{movie_id}", append_to_response=APPEND))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=None,
                        help="stop after N films (default: the whole catalogue)")
    args = parser.parse_args()

    started = time.monotonic()
    print("Discovering films...")
    movie_ids = discover_movie_ids(args.count)
    print(f"Fetching {len(movie_ids)} films with {WORKERS} workers\n")

    movies, done, failed = [], 0, 0
    fetch_started = time.monotonic()

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(fetch_movie, mid): mid for mid in movie_ids}
        for future in as_completed(futures):
            done += 1
            try:
                movies.append(future.result())
            except requests.RequestException as error:
                failed += 1
                print(f"  skipped {futures[future]}: {error.__class__.__name__}")
            if done % 250 == 0:
                rate = done / (time.monotonic() - fetch_started)
                print(f"  {done}/{len(movie_ids)}  ({rate:.0f}/s)")

    # Discovery order is popularity order, and the interface relies on that, so
    # put the results back into it after the threads scrambled them.
    position = {movie_id: i for i, movie_id in enumerate(movie_ids)}
    movies.sort(key=lambda m: position.get(m["id"], len(position)))

    out = HERE / "movies_raw.json"
    out.write_text(json.dumps(movies, ensure_ascii=False), encoding="utf-8")

    elapsed = time.monotonic() - started
    with_certificate = sum(1 for m in movies if m["content_ratings"]["results"])
    with_keywords = sum(1 for m in movies if m["keywords"]["results"])
    print(f"\n{len(movies)} films written to {out.name} in {elapsed / 60:.1f} min"
          f" ({failed} failed)")
    print(f"  {with_certificate} have a GB or US certificate "
          f"({with_certificate / max(len(movies), 1):.0%})")
    print(f"  {with_keywords} have keywords "
          f"({with_keywords / max(len(movies), 1):.0%})")


if __name__ == "__main__":
    main()
