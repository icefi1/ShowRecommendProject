"""
Collect human relevance judgements (report S9.2).

Every accuracy number so far is agreement with TMDB's own recommender, which is
a floor rather than the claim: scoring well means imitating the kind of system
this project argues is insufficient. This tool collects the answer key that does
carry the claim - a person saying whether a recommendation is any good.

HOW THE PAIRS ARE CHOSEN (pooling)

Judging every show against every other show is impossible: 3,542 x 3,541 pairs.
The standard answer, from the Cranfield paradigm and used by TREC since 1992, is
POOLING. For each query show, take the top few results from every system being
compared, merge them into one set, and judge that set. Each judgement is then
reused by every system, and a candidate no system returned cannot affect the
comparison - it would have been scored as "not returned" by all of them anyway.

Two properties matter and both are enforced here:

  * The pool is SHUFFLED, so the judge cannot tell which system produced a
    candidate, or which system ranked it first.
  * The pool is FIXED and shared. Every judge sees the same pairs in the same
    order, which is what makes agreement between judges measurable.

WHAT A JUDGE IS ASKED

"Would you recommend this to someone who liked X?" - not "is it similar", which
invites judging by genre label, and not "did you enjoy it", which is the
evaluative question the whole project keeps separate from the descriptive one.

Query shows are sampled from the most popular part of the catalogue so a judge
has a chance of knowing them. Unfamiliar is still a real answer: `?` records it
rather than forcing a guess, and the scorer reports how often it was used.

Run:
    venv\\Scripts\\python evaluation/judge.py --judge yourname

    --queries N     how many query shows to build the pool from (default 20)
    --depth N       results per system per query (default 5)
    --summary       show progress and stop, judging nothing
"""

import argparse
import json
import random
import sys
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from app.similarity import FeatureSpace  # noqa: E402
from evaluation.retrieval_accuracy import build_systems  # noqa: E402

# Which shows each judge says they have seen, keyed by lowercased name. The
# query show in a pair has to be one the judge knows - "would you recommend this
# to someone who liked X" is unanswerable otherwise - while the candidate only
# has to be judgeable from its poster and description.
FAMILIAR_FILE = HERE / "familiar.json"

POOL_FILE = HERE / "judging_pool.json"
JUDGEMENTS_FILE = HERE / "judgements.jsonl"

# Only these systems contribute candidates. The popularity and random controls
# are deliberately left out: they return shows no other system returns, so each
# one adds five junk pairs per query to a human's workload, and the TMDB run
# already settled them at 0.003 precision. Human time is the scarce resource
# here, and it should be spent separating the systems that are actually close.
POOLED_SYSTEMS = ("blocked", "blocked-cert", "flat", "embeddings", "embeddings+m")

# Query shows are drawn from this many of the most popular titles. The catalogue
# arrives from TMDB in popularity order, so this is simply the first slice of it.
# Judging is worthless if the judge has not heard of either show, and popularity
# is the only proxy for that available without asking them first.
POPULAR_POOL = 400

SEED = 20250831

VERDICTS = {
    "y": "yes",        # would recommend it to someone who liked the query
    "n": "no",         # would not
    "m": "maybe",      # defensible but weak
    "?": "unfamiliar",  # cannot judge - does not know one of the shows
}


def load_familiar():
    """Every judge's list of shows they know, as {judge: [show_id, ...]}."""
    if not FAMILIAR_FILE.exists():
        return {}
    return json.loads(FAMILIAR_FILE.read_text(encoding="utf-8"))


def save_familiar(judge, show_ids):
    """Replace one judge's list. Judges are keyed lowercased, as in scoring."""
    everyone = load_familiar()
    everyone[judge.strip().casefold()] = sorted(set(int(i) for i in show_ids))
    FAMILIAR_FILE.write_text(json.dumps(everyone, indent=1), encoding="utf-8")
    return everyone


def load_pool():
    return json.loads(POOL_FILE.read_text(encoding="utf-8")) if POOL_FILE.exists() else None


def build_pool(space, queries, depth, query_rows=None):
    """
    Choose the query shows, collect every system's top `depth` for each, and
    shuffle the merged candidates so the judge cannot see where they came from.

    `query_rows` names the query shows explicitly - the screening step passes
    the ones the judge says they have seen. Left out, the queries are sampled
    from the most popular titles, which was the original guess at familiarity
    and turned out to be a poor one: the first session came back 80% "don't
    know it".
    """
    everything = build_systems(space, Namespace(skip_embeddings=False), max_k=depth)
    systems = {name: rank for name, rank in everything.items() if name in POOLED_SYSTEMS}
    rng = random.Random(SEED)

    if query_rows is None:
        chosen = rng.sample(range(min(POPULAR_POOL, len(space.catalogue))), queries)
        picked_by = "popularity"
    else:
        chosen = sorted(query_rows)
        if len(chosen) > queries:
            chosen = sorted(rng.sample(chosen, queries))
        picked_by = "screened"
    query_rows = chosen

    entries = []
    for row in query_rows:
        # sources is kept for later analysis - which system found what - and is
        # deliberately never shown while judging.
        sources = {}
        for name, rank in systems.items():
            for candidate in list(rank(row))[:depth]:
                sources.setdefault(int(candidate), []).append(name)

        candidates = sorted(sources)
        rng.shuffle(candidates)
        entries.append({
            "query_id": space.catalogue[row]["id"],
            "candidates": [space.catalogue[c]["id"] for c in candidates],
            "sources": {str(space.catalogue[c]["id"]): sources[c] for c in candidates},
        })

    pool = {
        "seed": SEED,
        "depth": depth,
        # Recorded so the scorer reports exactly the systems that were pooled.
        # Scoring a system whose results nobody judged would count every one of
        # them as irrelevant purely for never having been looked at.
        "systems": list(systems),
        "queries": len(query_rows),
        # How the query shows were chosen, so a rebuilt pool is not silently
        # compared against one built the old way.
        "picked_by": picked_by,
        "built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "entries": entries,
    }
    POOL_FILE.write_text(json.dumps(pool, indent=1), encoding="utf-8")
    return pool


def load_judgements():
    """Every judgement made so far, as a list of records."""
    if not JUDGEMENTS_FILE.exists():
        return []
    return [json.loads(line) for line in JUDGEMENTS_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]


def record(judgement):
    """Append one judgement immediately, so quitting never loses work."""
    with open(JUDGEMENTS_FILE, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(judgement) + "\n")


def describe(show, width=88):
    """One block of text about a show: what a judge needs and nothing else."""
    genres = ", ".join(show.get("genres") or []) or "no genre listed"
    overview = (show.get("overview") or "").strip()
    if len(overview) > 300:
        # rstrip first, or an overview ending in a full stop shows four dots.
        overview = overview[:297].rsplit(" ", 1)[0].rstrip(".,;: ") + "..."

    lines = [
        f"  {show['name']} ({show.get('year', '?')})",
        f"  {genres} - {show.get('episodes', '?')} episodes - rated {show.get('certificate') or 'unrated'}",
    ]
    words, line = overview.split(), "  "
    for word in words:
        if len(line) + len(word) + 1 > width:
            lines.append(line)
            line = "  "
        line += word + " "
    lines.append(line.rstrip())
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge", help="your name, so agreement between judges can be measured")
    parser.add_argument("--queries", type=int, default=20, help="query shows in the pool (default 20)")
    parser.add_argument("--depth", type=int, default=5, help="results per system per query (default 5)")
    parser.add_argument("--summary", action="store_true", help="show progress and exit")
    parser.add_argument("--build-pool", action="store_true",
                        help="build the pool and exit, without judging anything. Run this "
                             "once, then give judging_pool.json to every judge so they all "
                             "see the same pairs")
    args = parser.parse_args()

    space = FeatureSpace()
    by_id = {show["id"]: show for show in space.catalogue}

    pool = load_pool()
    if pool is None:
        if args.summary:
            print("No pool built yet. Run with --build-pool to make one.")
            return
        print(f"Building a judging pool: {args.queries} query shows, "
              f"top {args.depth} from each system ...")
        pool = build_pool(space, args.queries, args.depth)
        print(f"Wrote {POOL_FILE.name}")
    elif args.build_pool:
        print(f"{POOL_FILE.name} already exists - delete it first to build a different pool.")
        return

    if args.build_pool:
        return

    pairs = [(e["query_id"], c) for e in pool["entries"] for c in e["candidates"]]
    judgements = load_judgements()

    # ------------------------------------------------------------- progress
    judges = sorted({j["judge"] for j in judgements})
    print(f"\nPool: {len(pool['entries'])} query shows, {len(pairs)} pairs to judge "
          f"(depth {pool['depth']}, seed {pool['seed']})")
    for name in judges:
        done = sum(1 for j in judgements if j["judge"] == name)
        print(f"  {name:16} {done:4}/{len(pairs)}  ({done / len(pairs):.0%})")
    if not judges:
        print("  no judgements yet")

    if args.summary:
        return
    if not args.judge:
        print("\nPass --judge yourname to start judging.")
        return

    done = {(j["query_id"], j["candidate_id"]) for j in judgements if j["judge"] == args.judge}
    remaining = [p for p in pairs if p not in done]
    if not remaining:
        print(f"\n{args.judge} has judged every pair. Score them with:")
        print("  venv\\Scripts\\python evaluation/retrieval_accuracy.py --truth human")
        return

    print(f"\n{len(remaining)} pairs left for {args.judge}.")
    print("\nFor each one: would you recommend the second show to someone who liked")
    print("the first? Answer on what the show is LIKE, not whether it is good.")
    print("\n  y = yes    n = no    m = maybe    ? = I don't know one of them")
    print("  s = skip for now    q = save and quit\n")

    for number, (query_id, candidate_id) in enumerate(remaining, 1):
        print("=" * 90)
        print(f"[{number} of {len(remaining)}]  SOMEONE LIKED:")
        print(describe(by_id[query_id]))
        print("\n  WOULD YOU RECOMMEND THEM:")
        print(describe(by_id[candidate_id]))
        print()

        while True:
            answer = input("  y / n / m / ? / s / q  > ").strip().lower()
            if answer == "q":
                print(f"\nSaved. {len(remaining) - number + 1} pairs still to do - "
                      f"rerun the same command to carry on.")
                return
            if answer == "s":
                break
            if answer in VERDICTS:
                record({
                    "judge": args.judge,
                    "query_id": query_id,
                    "candidate_id": candidate_id,
                    "verdict": VERDICTS[answer],
                    "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                })
                break
            print("  Please answer y, n, m, ?, s or q.")
        print()

    print("\nAll done. Score them with:")
    print("  venv\\Scripts\\python evaluation/retrieval_accuracy.py --truth human")


if __name__ == "__main__":
    main()
