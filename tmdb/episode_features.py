"""
Derive per-show pacing and structure features from episode-level TMDB data.

Reads  tmdb/episodes.json   (written by fetch_episodes.py)
Writes tmdb/episode_features.csv

Everything here is computed from two columns TMDB gives per episode -
vote_average and vote_count - plus runtime, episode_type and guest star counts.
No model, no LLM, no external data. These are the axes the project can ground
in measurement rather than opinion.
"""

import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def least_squares_slope(ys):
    """
    Slope of the best-fit line through ys, with x = 0, 1, 2, ...

    Written out rather than pulled from a library because it is four lines and
    the viva question "what does this number mean?" deserves a direct answer:
    it is the average change in rating per episode. Positive means the show
    got better as it went on.
    """
    n = len(ys)
    if n < 3:
        return 0.0

    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator = sum((x - mean_x) ** 2 for x in xs)

    return numerator / denominator if denominator else 0.0


def entity_density(texts):
    """
    Proportion of words that look like proper nouns - a capitalised word that
    is not the first word of a sentence.

    A rough proxy for how many named people, places and factions a viewer has
    to track, which is what "convoluted plot" usually means in practice. Rough
    is acceptable here: it only has to rank shows against each other, not be
    correct in absolute terms.
    """
    proper, total = 0, 0

    for text in texts:
        for sentence in text.replace("!", ".").replace("?", ".").split("."):
            words = sentence.split()
            for position, word in enumerate(words):
                stripped = word.strip(",;:'\"()[]")
                if not stripped.isalpha():
                    continue
                total += 1
                if position > 0 and stripped[0].isupper():
                    proper += 1

    return proper / total if total else 0.0


def features_for_show(episodes):
    """Compute one feature row from all episodes of a single show."""
    episodes = sorted(episodes, key=lambda e: (e["season_number"], e["episode_number"]))

    # Only episodes with real votes carry a meaningful rating.
    rated = [e for e in episodes if (e["vote_count"] or 0) > 0]
    ratings = [e["vote_average"] for e in rated]

    runtimes = [e["runtime"] for e in episodes if e["runtime"]]
    votes = [e["vote_count"] for e in rated if e["vote_count"]]
    overviews = [e["overview"] for e in episodes if e["overview"]]

    # Season 1 trajectory: does the show start weak and climb? That is the
    # measurable core of "slow burn".
    season_one = [e["vote_average"] for e in rated if e["season_number"] == 1]
    slow_burn_slope = least_squares_slope(season_one)

    # Finales. A serialised show builds to them and they rate well above the
    # mean; an episodic show treats them as just another episode.
    finales = [e["vote_average"] for e in rated if e["episode_type"] == "finale"]
    mean_rating = st.mean(ratings) if ratings else 0.0
    finale_delta = (st.mean(finales) - mean_rating) if finales else 0.0

    return {
        "episode_count": len(episodes),
        "rated_episode_count": len(rated),
        "mean_rating": round(mean_rating, 3),
        # Flat ratings => every episode is interchangeable => episodic.
        "rating_stdev": round(st.pstdev(ratings), 3) if len(ratings) > 1 else 0.0,
        "slow_burn_slope": round(slow_burn_slope, 4),
        "finale_delta": round(finale_delta, 3),
        # How far the most-watched episode stands above the typical one.
        "vote_peak_ratio": round(max(votes) / st.median(votes), 2) if len(votes) > 1 and st.median(votes) else 0.0,
        "runtime_median": st.median(runtimes) if runtimes else "",
        # Spread of runtimes: anthologies and miniseries vary, procedurals don't.
        "runtime_stdev": round(st.pstdev(runtimes), 2) if len(runtimes) > 1 else 0.0,
        # Procedurals bring a fresh guest cast every week; serialised drama
        # reuses a standing cast. A direct, cheap episodic/serialised signal.
        "guest_star_mean": round(st.mean([e["guest_star_count"] for e in episodes]), 2) if episodes else 0.0,
        "entity_density": round(entity_density(overviews), 4),
        "overview_coverage": round(len(overviews) / len(episodes), 3) if episodes else 0.0,
    }


def main():
    episodes = json.loads((HERE / "episodes.json").read_text(encoding="utf-8"))
    shows = json.loads((HERE / "shows_raw.json").read_text(encoding="utf-8"))
    names = {s["id"]: s.get("name", "") for s in shows}

    by_show = defaultdict(list)
    for episode in episodes:
        by_show[episode["show_id"]].append(episode)

    print(f"{len(episodes):,} episodes across {len(by_show)} shows")

    rows = []
    for show_id, show_episodes in by_show.items():
        row = {"id": show_id, "name": names.get(show_id, "")}
        row.update(features_for_show(show_episodes))
        rows.append(row)

    out = HERE / "episode_features.csv"
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} feature rows to {out.name}")


if __name__ == "__main__":
    main()
