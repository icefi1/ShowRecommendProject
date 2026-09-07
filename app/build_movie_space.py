"""
Build the blocked feature space for films.

Reads   tmdb/movies_raw.json
Writes  app/feature_space_movie.npz, app/feature_space_movie.json

A sibling of build_space.py rather than a flag on it, because the structure
block is genuinely a different set of axes. Seven of television's thirteen -
episode count, season count, miniseries, per-episode rating variance, slow-burn
slope, finale delta, standout-episode ratio - measure how a story is spread
across a run of episodes. A film has one episode, so those are not "zero for
films", they are undefined, and putting them in as zeros would tell the distance
metric that every film is identical along seven axes.

The genre and keyword blocks work the same way as television, with one thing
worth noticing for the report: **TMDB's film genre list contains Horror,
Thriller, Romance and History, and its television list does not.** The taxonomy
gap in section 2.3 is specific to television, not to TMDB as a whole, which
makes it a stronger finding rather than a weaker one - the vocabulary exists,
and television simply does not get it.

Run:  venv\\Scripts\\python app/build_movie_space.py
"""

import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TMDB = ROOT / "tmdb"
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from app.build_space import (  # noqa: E402
    CERTIFICATE_SCALE,
    MIN_KEYWORD_SHOWS,
    US_CERTIFICATE_SCALE,
    is_anime,
    percentile_rank,
)


def certificate_label(movie):
    """The awarded certificate, preferring GB and falling back to US."""
    ratings = movie.get("content_ratings", {}).get("results", [])
    for wanted, scale in (("GB", CERTIFICATE_SCALE), ("US", US_CERTIFICATE_SCALE)):
        for rating in ratings:
            if rating.get("iso_3166_1") == wanted and rating.get("rating") in scale:
                return rating["rating"], scale[rating["rating"]]
    return None, None


def build():
    movies = json.loads((TMDB / "movies_raw.json").read_text(encoding="utf-8"))
    print(f"{len(movies)} films")

    # ----------------------------------------------------------------- genre
    genre_names = sorted({g["name"] for m in movies for g in m.get("genres", [])})
    genre_index = {name: i for i, name in enumerate(genre_names)}
    genre_matrix = np.zeros((len(movies), len(genre_names)), dtype=np.float32)
    for row, movie in enumerate(movies):
        for genre in movie.get("genres", []):
            genre_matrix[row, genre_index[genre["name"]]] = 1.0

    # -------------------------------------------------------------- keywords
    film_keywords = [
        {k["name"] for k in m.get("keywords", {}).get("results", [])} for m in movies
    ]
    document_frequency = Counter(k for keywords in film_keywords for k in keywords)
    vocabulary = sorted(
        term for term, count in document_frequency.items() if count >= MIN_KEYWORD_SHOWS
    )
    keyword_index = {term: i for i, term in enumerate(vocabulary)}

    keyword_matrix = np.zeros((len(movies), len(vocabulary)), dtype=np.float32)
    for row, keywords in enumerate(film_keywords):
        for term in keywords:
            column = keyword_index.get(term)
            if column is None:
                continue
            idf = math.log(len(movies) / (1 + document_frequency[term])) + 1.0
            keyword_matrix[row, column] = idf

    norms = np.linalg.norm(keyword_matrix, axis=1, keepdims=True)
    keyword_matrix /= np.where(norms == 0, 1.0, norms)

    # ------------------------------------------------------------- structure
    def certificate(movie):
        _, value = certificate_label(movie)
        # Unrated sits mid-scale rather than at an extreme, exactly as it does
        # for television. It matters more here: only 55% of films carry a GB or
        # US certificate against 95% of shows, so this is the weakest axis in
        # the block and worth saying so in the report.
        return 0.5 if value is None else value

    def year(movie):
        try:
            return int((movie.get("first_air_date") or "")[:4])
        except ValueError:
            return 0

    raw_structure = {
        "maturity": [certificate(m) for m in movies],
        "audience_rating": [m.get("vote_average") or 0 for m in movies],
        # Replaces episode_length. The only duration a film has.
        "runtime": [m.get("runtime") or 0 for m in movies],
        # How widely seen. A blockbuster and an obscure festival film are a
        # different proposition even when their subject matter matches, and
        # television has no equivalent axis because a series' vote count is
        # confounded by how long it ran.
        "audience_reach": [m.get("vote_count") or 0 for m in movies],
        # Age of the film. Nothing in the television block captures this, and
        # for film it separates a 1970s thriller from a 2023 one.
        "release_recency": [year(m) for m in movies],
        # The closest a film gets to serialisation: is it part of a franchise.
        "part_of_series": [1.0 if m.get("collection") else 0.0 for m in movies],
        # Stands in for guest_star_mean, which measured ensemble size on
        # television. Same quantity, one credit list instead of many.
        "ensemble_size": [m.get("cast_size") or 0 for m in movies],
    }

    structure_names = sorted(raw_structure)
    structure_matrix = np.zeros((len(movies), len(structure_names)), dtype=np.float32)
    for column, name in enumerate(structure_names):
        values = raw_structure[name]
        # Percentile-rank everything except the two that are already 0-1: a
        # certificate is a position on a fixed scale, and a franchise flag is a
        # yes or no. Ranking those would distort a meaning that is already
        # correct.
        if name in ("maturity", "part_of_series"):
            structure_matrix[:, column] = np.asarray(values, dtype=np.float32)
        else:
            structure_matrix[:, column] = percentile_rank(values)

    # ------------------------------------------------------------- catalogue
    catalogue = [
        {
            "id": m["id"],
            "name": m.get("name", ""),
            "year": (m.get("first_air_date") or "")[:4],
            "overview": m.get("overview") or "",
            "poster": m.get("poster_path"),
            "genres": [g["name"] for g in m.get("genres", [])],
            # The interface shows "episodes" for television; for a film the
            # comparable single number is its runtime, and the field is reused
            # so one card renderer serves both.
            "episodes": 1,
            "seasons": 1,
            "runtime": m.get("runtime") or 0,
            "rating": round(m.get("vote_average") or 0, 1),
            "certificate": certificate_label(m)[0],
            "maturity": round(certificate(m), 3),
            "kind": "movie",
            "is_anime": is_anime(m),
            "collection": m.get("collection"),
            "keywords_all": sorted({k["name"] for k in m.get("keywords", {}).get("results", [])}),
            "tmdb_similar": m.get("similar") or [],
            "tmdb_recommended": m.get("recommendations") or [],
        }
        for m in movies
    ]

    np.savez_compressed(
        OUT / "feature_space_movie.npz",
        genre=genre_matrix,
        keywords=keyword_matrix,
        structure=structure_matrix,
    )
    (OUT / "feature_space_movie.json").write_text(
        json.dumps({
            "catalogue": catalogue,
            "blocks": {
                "genre": genre_names,
                "keywords": vocabulary,
                "structure": structure_names,
            },
        }, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"  genre block:     {len(genre_names)} dims  {genre_names}")
    print(f"  keyword block:   {len(vocabulary)} dims (from {len(document_frequency)} raw,"
          f" min {MIN_KEYWORD_SHOWS} films)")
    print(f"  structure block: {len(structure_names)} dims  {structure_names}")
    print(f"  anime films:     {sum(1 for c in catalogue if c['is_anime'])}")
    print("Wrote feature_space_movie.npz and feature_space_movie.json")


if __name__ == "__main__":
    build()
