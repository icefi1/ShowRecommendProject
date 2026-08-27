"""
Export show text for labelling, as a readable batch file.

This is the no-API-key path. label_shows.py calls the Anthropic API and costs
money; this script instead writes the same assembled text to a file, so labels
can be produced by any annotator - a person, or a model being read the file -
and dropped back into the same labels.jsonl format that training reads.

The two paths produce identical input text, so labels from either are
comparable, and the choice of annotator is documented rather than baked in.

Sampling is stratified by genre. The catalogue is roughly 35% animation
(TMDB's Netflix GB listing, sorted by popularity, is anime-heavy), so taking
the top N by popularity would produce a training set that teaches the model
about anime and little else.

Run:
    python labelling/export_batch.py --count 30
    python labelling/export_batch.py --count 30 --skip 30   # the next batch
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT))
from labelling.label_shows import build_show_text  # noqa: E402


def stratified_order(shows):
    """
    Interleave shows by their primary genre.

    Round-robin across genre buckets: one Drama, one Comedy, one Animation, and
    so on, then round again. Taking any prefix of the result gives a spread
    across the catalogue rather than a block of whatever is most popular.
    """
    buckets = defaultdict(list)
    for show in shows:
        genres = show.get("genres", [])
        primary = genres[0]["name"] if genres else "Unknown"
        buckets[primary].append(show)

    # Largest buckets first, so common genres are represented proportionally
    # early rather than being crowded out by rare ones.
    ordered_buckets = sorted(buckets.values(), key=len, reverse=True)

    result = []
    position = 0
    while len(result) < len(shows):
        added = False
        for bucket in ordered_buckets:
            if position < len(bucket):
                result.append(bucket[position])
                added = True
        if not added:
            break
        position += 1
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--out", default="batch.txt")
    args = parser.parse_args()

    shows = json.loads((ROOT / "tmdb" / "shows_raw.json").read_text(encoding="utf-8"))

    episodes_by_show = {}
    episodes_path = ROOT / "tmdb" / "episodes.json"
    if episodes_path.exists():
        for episode in json.loads(episodes_path.read_text(encoding="utf-8")):
            episodes_by_show.setdefault(episode["show_id"], []).append(episode)

    # Skip anything already labelled, so batches do not overlap.
    labelled = set()
    labels_path = HERE / "labels.jsonl"
    if labels_path.exists():
        for line in labels_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                labelled.add(json.loads(line)["id"])

    candidates = [s for s in stratified_order(shows) if s["id"] not in labelled]
    batch = candidates[args.skip:args.skip + args.count]

    parts = []
    for show in batch:
        parts.append(f"===== SHOW {show['id']} =====")
        parts.append(build_show_text(show, episodes_by_show))
        parts.append("")

    out_path = HERE / args.out
    out_path.write_text("\n".join(parts), encoding="utf-8")

    print(f"{len(labelled)} already labelled, {len(candidates)} remaining")
    print(f"Wrote {len(batch)} shows to {out_path.name}")
    for show in batch:
        genres = ", ".join(g["name"] for g in show.get("genres", []))
        print(f"  {show['id']:>7}  {show.get('name', '')[:38]:40} {genres[:34]}")


if __name__ == "__main__":
    main()
