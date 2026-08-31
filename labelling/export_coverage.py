"""
Export a batch chosen to cover under-represented AXES, not TMDB genres.

Why this exists
---------------
export_batch.py stratifies by primary TMDB genre. That failed in a way worth
recording: TMDB TV has only 15 genres and no Horror, Romance, Thriller, History
or Fantasy among them, so those strata cannot be selected and never appear.
Batch 1 therefore contained no horror show at all, and the trained model scored
The Haunting of Hill House at 0.10 on `horror` - it had never seen one.

The catalogue is not missing these shows. 85 shows carry a `romance` keyword and
none can be labelled Romance; Stranger Things is horror labelled
"Action & Adventure, Mystery, Sci-Fi & Fantasy". Only the taxonomy is missing.

So this script selects on keyword evidence for specific axes instead, and
reports the label variance already present per axis so the gap is visible
before training rather than after.

Run:
    python labelling/export_coverage.py --count 26
"""

import argparse
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT))
from labelling.label_shows import build_show_text  # noqa: E402
from labelling.schema import AXIS_NAMES  # noqa: E402

# Keyword evidence for the axes batch 1 could not teach. These are search terms
# over TMDB keywords, not axis definitions - they only have to FIND candidate
# shows, and the annotator still scores each one on its merits.
# Keyword evidence for finding candidate shows, per axis. These are search
# terms over TMDB keywords, not axis definitions - they only have to SURFACE
# plausible candidates, and the annotator still scores each show on its merits.
#
# Deliberately broader than the axes currently weak. Which axes get targeted is
# decided at run time from measured label variance (see pick_targets), so a dict
# that only covered today's gaps would leave the sampler fighting the last
# battle once those gaps close - which is exactly what happened after batch 2.
AXIS_EVIDENCE = {
    "horror": ["horror", "slasher", "zombie", "vampire", "demon", "haunted",
               "ghost", "monster", "supernatural", "gore", "creature"],
    "romance": ["romance", "romantic", "love triangle", "dating", "wedding",
                "marriage", "relationship", "obsessive love", "romcom"],
    "historical": ["period drama", "historical", "world war", "1940s", "1960s",
                   "based on true story", "british history", "monarchy", "war"],
    "fantasy": ["magic", "dragon", "witch", "wizard", "fantasy world",
                "mythology", "sword and sorcery", "high fantasy", "isekai"],
    "thriller": ["thriller", "psychological thriller", "conspiracy", "espionage",
                 "spy", "kidnapping", "manhunt", "survival"],
    "cynical": ["satire", "dark comedy", "corruption", "dystopia", "black comedy"],

    # Axes with no direct keyword vocabulary in TMDB - the original finding.
    # These terms reach them obliquely, through subject matter that tends to
    # come with the quality.
    "jumpscares": ["supernatural horror", "haunted house", "slasher", "ghost",
                   "possession", "found footage", "teen horror"],
    "dialogue_driven": ["courtroom", "lawyer", "politics", "talk show", "sitcom",
                        "workplace comedy", "interview", "negotiation", "trial"],
    "plot_complexity": ["conspiracy", "time travel", "parallel world", "amnesia",
                        "multiple timelines", "unreliable narrator", "heist"],
    "plot_twists": ["twist ending", "murder mystery", "whodunit", "betrayal",
                    "secret identity", "double agent", "revelation"],
    "ensemble": ["ensemble cast", "family drama", "group of friends", "workplace",
                 "high school", "anthology", "large family"],
    "emotional_intensity": ["tragedy", "grief", "terminal illness", "death of a child",
                            "melodrama", "loss", "war crime", "addiction"],
    "melancholy": ["nostalgia", "loneliness", "memories", "coming of age",
                   "slice of life", "aging", "regret"],
    "sentimental": ["heartfelt", "feelgood", "family", "friendship", "reunion",
                    "inspirational", "healing"],
}

# Axes settled by TMDB rather than judged. They are not votable and are written
# straight from the catalogue, so low label variance on them costs nothing and
# they must not be chased.
try:
    from labelling.schema import FACT_AXES
except ImportError:  # running as a script from inside labelling/
    FACT_AXES = []


def pick_targets(rows, how_many=6):
    """
    Choose which axes this batch should chase, from measured label variance.

    An axis the existing labels barely vary on teaches the model nothing, so the
    weakest are the ones worth feeding. Only axes with evidence terms can
    actually be searched for, and fact axes are excluded because TMDB settles
    them.
    """
    if not rows:
        return list(AXIS_EVIDENCE)[:how_many]

    spreads = []
    for axis in AXIS_EVIDENCE:
        if axis in FACT_AXES:
            continue
        values = [r["labels"][axis] for r in rows if axis in r["labels"]]
        if values:
            spreads.append((st.pstdev(values), axis))

    spreads.sort()
    return [axis for _, axis in spreads[:how_many]]


def report_current_variance():
    """Show which axes the existing labels already teach, and which they cannot."""
    path = HERE / "labels.jsonl"
    if not path.exists():
        print("No labels yet.\n")
        return set()

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"Existing labels: {len(rows)} shows\n")
    print("  Axes with the least signal (an axis with no spread teaches nothing):")

    spreads = []
    for axis in AXIS_NAMES:
        values = [r["labels"][axis] for r in rows]
        spreads.append((st.pstdev(values), max(values), axis))
    spreads.sort()

    weak = set()
    for sd, high, axis in spreads[:10]:
        flag = "  <-- no positive examples" if high < 0.5 else ""
        print(f"    {axis:22} sd={sd:.3f}  max={high:.2f}{flag}")
        weak.add(axis)

    print()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=26)
    parser.add_argument("--out", default="batch.txt")
    args = parser.parse_args()

    rows = report_current_variance()
    targets = pick_targets(rows)
    print(f"  Targeting this batch at: {', '.join(targets)}\n")

    shows = json.loads((ROOT / "tmdb" / "shows_raw.json").read_text(encoding="utf-8"))

    labelled = set()
    labels_path = HERE / "labels.jsonl"
    if labels_path.exists():
        for line in labels_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                labelled.add(json.loads(line)["id"])

    # Score each unlabelled show for how much evidence it carries per target
    # axis, then round-robin across axes so no single axis dominates the batch.
    by_axis = {axis: [] for axis in targets}
    for show in shows:
        if show["id"] in labelled:
            continue
        keywords = " ".join(
            k["name"].lower() for k in show.get("keywords", {}).get("results", [])
        )
        for axis in targets:
            terms = AXIS_EVIDENCE[axis]
            hits = sum(1 for term in terms if term in keywords)
            if hits:
                by_axis[axis].append((hits, show.get("vote_count", 0), show))

    for axis in by_axis:
        # Most evidence first; popularity breaks ties, since better-known shows
        # have more episode text for the annotator to work from.
        by_axis[axis].sort(key=lambda row: (-row[0], -row[1]))

    picked, seen = [], set()
    position = 0
    while len(picked) < args.count:
        added = False
        for axis, candidates in by_axis.items():
            if len(picked) >= args.count:
                break
            while position < len(candidates):
                show = candidates[position][2]
                if show["id"] not in seen:
                    seen.add(show["id"])
                    picked.append((axis, show))
                    added = True
                    break
                position += 1
        if not added:
            break
        position += 1

    episodes_by_show = {}
    episodes_path = ROOT / "tmdb" / "episodes.json"
    if episodes_path.exists():
        for episode in json.loads(episodes_path.read_text(encoding="utf-8")):
            episodes_by_show.setdefault(episode["show_id"], []).append(episode)

    parts = []
    for _, show in picked:
        parts.append(f"===== SHOW {show['id']} =====")
        parts.append(build_show_text(show, episodes_by_show))
        parts.append("")

    (HERE / args.out).write_text("\n".join(parts), encoding="utf-8")

    print(f"Selected {len(picked)} shows targeting under-covered axes:\n")
    for axis, show in picked:
        print(f"  [{axis:11}] {show['id']:>7}  {show.get('name', '')[:44]}")


if __name__ == "__main__":
    main()
