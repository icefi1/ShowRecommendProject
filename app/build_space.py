"""
Build the blocked feature space from the TMDB data on disk.

Reads   tmdb/shows_raw.json, tmdb/episode_features.csv (optional)
Writes  app/feature_space.npz, app/feature_space.json

The vector for each show is partitioned into three blocks. Distance is computed
per block and the blocks are weighted at query time (report S6.4). Flat cosine
over one concatenated vector produces mush: two shows come out equidistant for
unrelated reasons and the user cannot tell which reason applied.

  genre      15 dims   binary, from TMDB's TV genre list
  keywords   N dims    TF-IDF over TMDB keywords
  structure  M dims    measured pacing and form features, percentile-normalised

Nothing here is an LLM judgement. Every number is derived from a field TMDB
returned, which means every explanation the interface gives traces back to a
source the report can point at.
"""

import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TMDB = ROOT / "tmdb"
OUT = Path(__file__).resolve().parent

# A keyword appearing on a single show cannot create similarity with any other
# show; it only adds a dimension of noise. 65% of the raw vocabulary is
# singletons, so a keyword must appear on at least this many shows to earn a
# dimension.
MIN_KEYWORD_SHOWS = 3

# UK certificates mapped onto a 0-1 "how adult is this" scale.
CERTIFICATE_SCALE = {"U": 0.0, "PG": 0.2, "12": 0.4, "12A": 0.4, "15": 0.7, "18": 1.0}

# US TV ratings, used when no GB certificate exists. Only 69% of the catalogue
# carries a GB rating; falling back to US takes coverage to 95%, which matters
# because an unrated show defaults to mid-scale and then ranks against
# everything. Teen Titans Go! had no GB certificate and was pulling 15-rated
# shows into its results.
#
# Positioned against the GB scale rather than treated as equivalent - TV-14 has
# no exact GB counterpart, so it sits between 12 and 15.
US_CERTIFICATE_SCALE = {
    "TV-Y": 0.0,     # all children
    "TV-G": 0.05,    # general audiences
    "TV-Y7": 0.15,   # 7 and over
    "TV-PG": 0.30,   # parental guidance
    "TV-14": 0.60,   # between GB 12 and 15
    "TV-MA": 1.00,   # mature only, ~18
    # NR is deliberately absent: "not rated" is missing data, not a rating, and
    # mapping it to a number would invent information.
}


def percentile_rank(values):
    """
    Map values onto 0-1 by rank rather than magnitude.

    Trade-off for the report: this discards absolute scale, so 0.9 means "more
    than 90% of the catalogue" rather than any particular count. In exchange it
    is immune to outliers - Sesame Street's 3,551 episodes would otherwise
    compress every other show into the bottom 2% of that axis - and it puts
    every structure axis on the same footing, which is what a shared distance
    metric requires.
    """
    values = np.asarray(values, dtype=float)
    order = values.argsort()
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks / max(len(values) - 1, 1)


def load_episode_features():
    """Per-show pacing features, if fetch_episodes and episode_features have run."""
    path = TMDB / "episode_features.csv"
    if not path.exists():
        return {}

    rows = {}
    with open(path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[int(row["id"])] = row
    return rows


def as_ids(value):
    """
    Read a similar/recommendations field in either shape.

    fetch_shows.py now stores these as bare id lists; older files hold TMDB's
    full `{"results": [{...}]}` payload. Accepting both means the space can be
    rebuilt from data fetched before the trim without re-downloading it.
    """
    if not value:
        return []
    if isinstance(value, dict):
        return [x["id"] for x in value.get("results", [])]
    return [x["id"] if isinstance(x, dict) else x for x in value]


def numeric(row, key):
    """Read one float from a CSV row, treating blanks and junk as zero."""
    try:
        return float(row.get(key) or 0)
    except (TypeError, ValueError):
        return 0.0


def build():
    shows = json.loads((TMDB / "shows_raw.json").read_text(encoding="utf-8"))
    episode_features = load_episode_features()
    print(f"{len(shows)} shows, {len(episode_features)} with episode features")

    # ----------------------------------------------------------------- genre
    genre_names = sorted({g["name"] for s in shows for g in s.get("genres", [])})
    genre_index = {name: i for i, name in enumerate(genre_names)}
    genre_matrix = np.zeros((len(shows), len(genre_names)), dtype=np.float32)

    for row, show in enumerate(shows):
        for genre in show.get("genres", []):
            genre_matrix[row, genre_index[genre["name"]]] = 1.0

    # -------------------------------------------------------------- keywords
    show_keywords = [
        {k["name"] for k in s.get("keywords", {}).get("results", [])} for s in shows
    ]
    document_frequency = Counter(k for keywords in show_keywords for k in keywords)
    vocabulary = sorted(
        term for term, count in document_frequency.items() if count >= MIN_KEYWORD_SHOWS
    )
    keyword_index = {term: i for i, term in enumerate(vocabulary)}

    keyword_matrix = np.zeros((len(shows), len(vocabulary)), dtype=np.float32)
    total_shows = len(shows)

    for row, keywords in enumerate(show_keywords):
        for term in keywords:
            column = keyword_index.get(term)
            if column is None:
                continue
            # Standard smoothed IDF. A keyword on 4 shows says far more about
            # similarity than "based on novel or book", which is on 63.
            idf = math.log(total_shows / (1 + document_frequency[term])) + 1.0
            keyword_matrix[row, column] = idf

    # L2-normalise, so the dot product of two rows is their cosine similarity
    # and a show with 50 keywords cannot dominate one with 8.
    norms = np.linalg.norm(keyword_matrix, axis=1, keepdims=True)
    keyword_matrix /= np.where(norms == 0, 1.0, norms)

    # ------------------------------------------------------------- structure
    def certificate_label(show):
        """
        The awarded certificate, preferring GB and falling back to US.

        Returns (label, value) or (None, None). GB is preferred because the
        catalogue is Netflix GB; US fills the 31% of shows GB does not rate.
        """
        ratings = show.get("content_ratings", {}).get("results", [])

        for rating in ratings:
            if rating.get("iso_3166_1") == "GB":
                label = rating.get("rating")
                if label in CERTIFICATE_SCALE:
                    return label, CERTIFICATE_SCALE[label]

        for rating in ratings:
            if rating.get("iso_3166_1") == "US":
                label = rating.get("rating")
                if label in US_CERTIFICATE_SCALE:
                    return label, US_CERTIFICATE_SCALE[label]

        return None, None

    def certificate(show):
        _, value = certificate_label(show)
        # Unrated sits mid-scale rather than at an extreme, so a missing
        # certificate neither blocks a show from adult results nor pushes it
        # into children's ones. It is still the worst case - 5% of the
        # catalogue - and is worth reducing with another source.
        return 0.5 if value is None else value

    raw_structure = {
        "maturity": [certificate(s) for s in shows],
        "episode_count": [s.get("number_of_episodes") or 0 for s in shows],
        "season_count": [s.get("number_of_seasons") or 0 for s in shows],
        "episode_length": [(s.get("episode_run_time") or [0])[0] for s in shows],
        "audience_rating": [s.get("vote_average") or 0 for s in shows],
        "is_miniseries": [1.0 if s.get("type") == "Miniseries" else 0.0 for s in shows],
    }

    # These only exist once the episode pipeline has run. The space is built
    # either way so the interface is never blocked on the fetch.
    if episode_features:
        for key in (
            "rating_stdev",
            "slow_burn_slope",
            "finale_delta",
            "vote_peak_ratio",
            "guest_star_mean",
            "entity_density",
            "runtime_stdev",
        ):
            raw_structure[key] = [
                numeric(episode_features.get(s["id"], {}), key) for s in shows
            ]

    structure_names = sorted(raw_structure)
    structure_matrix = np.zeros((len(shows), len(structure_names)), dtype=np.float32)

    for column, name in enumerate(structure_names):
        values = raw_structure[name]
        # Axes already meaningful on 0-1 keep their values; the rest are ranked.
        if name in ("maturity", "is_miniseries"):
            structure_matrix[:, column] = values
        else:
            structure_matrix[:, column] = percentile_rank(values)

    # ------------------------------------------------------------------ save
    np.savez_compressed(
        OUT / "feature_space.npz",
        genre=genre_matrix,
        keywords=keyword_matrix,
        structure=structure_matrix,
    )

    catalogue = [
        {
            "id": s["id"],
            "name": s.get("name", ""),
            "year": (s.get("first_air_date") or "")[:4],
            "overview": s.get("overview") or "",
            "poster": s.get("poster_path"),
            "genres": [g["name"] for g in s.get("genres", [])],
            "episodes": s.get("number_of_episodes") or 0,
            "seasons": s.get("number_of_seasons") or 0,
            "rating": round(s.get("vote_average") or 0, 1),
            # The awarded GB certificate and its position on a 0-1 scale.
            # Immutable: it comes from the classification body via TMDB, is
            # never predicted and never voted on. The similarity engine uses
            # `maturity` to penalise recommendations far from the query's
            # rating - see similarity.py.
            "certificate": certificate_label(s)[0],
            "maturity": round(certificate(s), 3),
            # Every keyword TMDB has for this show, for display only.
            #
            # Deliberately not the same set as the keyword BLOCK above. That
            # block drops any keyword appearing on fewer than three shows,
            # because a keyword on one show cannot create similarity with
            # anything and only adds a noisy dimension. But those same rare
            # keywords are often the most recognisable ones a person would look
            # for - Breaking Bad loses "crystal meth", "meth lab" and "dea
            # agent" to that filter - so showing only the block's vocabulary
            # would look broken to anyone who knows the show.
            #
            # Two different jobs: what the engine ranks on, and what a reader is
            # told the show is about.
            "keywords_all": sorted(
                {k["name"] for k in s.get("keywords", {}).get("results", [])}
            ),
            # TMDB's own similarity, kept as the ground-truth proxy for S9.2.
            # Stored as bare id lists by fetch_shows.py - the full show records
            # TMDB returns here were 20% of the raw file and only the ids are
            # ever read. The dict form is still accepted so an older
            # shows_raw.json does not have to be re-fetched.
            "tmdb_similar": as_ids(s.get("similar")),
            "tmdb_recommended": as_ids(s.get("recommendations")),
        }
        for s in shows
    ]

    (OUT / "feature_space.json").write_text(
        json.dumps(
            {
                "catalogue": catalogue,
                "blocks": {
                    "genre": genre_names,
                    "keywords": vocabulary,
                    "structure": structure_names,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"  genre block:     {len(genre_names)} dims")
    print(f"  keyword block:   {len(vocabulary)} dims "
          f"(from {len(document_frequency)} raw, min {MIN_KEYWORD_SHOWS} shows)")
    print(f"  structure block: {len(structure_names)} dims")
    print(f"    {structure_names}")
    print("Wrote feature_space.npz and feature_space.json")


if __name__ == "__main__":
    build()
