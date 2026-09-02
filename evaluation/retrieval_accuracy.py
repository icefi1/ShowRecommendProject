"""
Retrieval accuracy against TMDB's own related-titles lists (report S9.1, S9.2).

The question this answers: when the system returns five shows, how many of them
does an independent source also consider related? That number is meaningless on
its own, so it is measured for six systems at once and reported side by side.

  1. blocked        the engine as deployed - per-block cosine, weighted, plus
                    the certificate penalty
  2. flat           the same vectors concatenated into one list and compared
                    with a single cosine (report S9.5 ablation: does blocking
                    earn its place?)
  3. embeddings     show text through a frozen sentence transformer, plain
                    cosine (report S9.1 baseline: the black box this project
                    has to stay competitive with)
  4. embeddings+m   the same baseline with the certificate penalty bolted on,
                    which separates "our feature space helps" from "the age
                    rating rule helps"
  5. popular        the most popular shows in the catalogue, ignoring the query
                    entirely - the standard non-personalised control
  6. random         five shows drawn at random - what chance looks like

WHAT THE GROUND TRUTH IS, AND WHAT IT IS NOT

`fetch_shows.py` stored TMDB's own "recommendations" list for every show, so an
answer key already exists for 94% of the catalogue at no extra cost. But it is
another recommender's opinion, built partly from user behaviour, so scoring well
here means agreeing with TMDB. That is a floor - evidence the space is not
returning noise - and not the project's claim, which is about steering and
explanation and is tested with people instead. Reported precision should be read
in that spirit.

Only shows with at least k related titles inside the catalogue are queried at
each k, so a perfect system could score 1.0 rather than being capped by missing
answers.

Run:  venv\\Scripts\\python evaluation/retrieval_accuracy.py
      venv\\Scripts\\python evaluation/retrieval_accuracy.py --queries 200
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from app.similarity import DEFAULT_WEIGHTS, MATURITY_PENALTY, FeatureSpace  # noqa: E402

# Precision is reported at these cut-offs. 20 is not included: TMDB returns at
# most 20 related titles and no show in this catalogue has all 20 present, so
# there is no query set for it.
K_VALUES = (5, 10)

# Encoding 3,542 shows takes a couple of minutes on CPU and never changes unless
# the catalogue does, so it is cached next to this script.
EMBEDDING_CACHE = HERE / "embeddings.npz"

SEED = 20250831

# Resamples used for the confidence intervals. A precision of 0.134 measured on
# 1,561 shows is an estimate, not a constant: query a different 1,561 shows and
# it would come out slightly differently. Bootstrapping measures how much.
BOOTSTRAP_SAMPLES = 2000

# Every other system is also reported as a gap against this one, because the
# question the dissertation asks is not "how good is the space" but "is it
# better than the alternatives".
REFERENCE = "blocked"


# --------------------------------------------------------------- ground truth


def ground_truth(space, field):
    """
    For every show, the rows TMDB considers related to it.

    Ids are translated to row numbers once here, so the measurement loop never
    has to touch the catalogue again. Shows TMDB names that are not in this
    catalogue (films, non-Netflix titles) are dropped - they cannot be
    retrieved, so counting them would punish every system equally for nothing.
    """
    truth = {}
    for row, show in enumerate(space.catalogue):
        ids = show.get(field) or []
        rows = {space.index_by_id[i] for i in ids if i in space.index_by_id}
        rows.discard(row)
        truth[row] = rows
    return truth


# ------------------------------------------------------------------- scoring


def unit_rows(matrix):
    """
    Scale every row to length 1.

    Once rows are unit length, the dot product of two of them *is* their cosine
    similarity, so the whole catalogue can be scored with one matrix-vector
    multiply instead of a division per candidate.
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)


def maturity_penalty(scores, space, query_row):
    """The same certificate demotion the deployed engine applies."""
    gap = np.abs(space.maturity - space.maturity[query_row])
    return scores * (1.0 - MATURITY_PENALTY * gap)


def top_rows(scores, query_row, k):
    """The k highest-scoring rows, best first, never the query itself."""
    scores = scores.copy()
    scores[query_row] = -np.inf
    top = np.argpartition(-scores, k)[:k]
    return top[np.argsort(-scores[top])]


# ------------------------------------------------------------------- systems


def build_systems(space, args):
    """
    One ranking function per system: row index in, ordered rows out.

    Each is a closure over matrices prepared once, because the cost that matters
    is per query and there are thousands of them.
    """
    systems = {}
    max_k = max(K_VALUES)

    # 1. The engine as deployed. Called through its real entry point rather than
    #    reimplemented here, so this row measures what actually ships.
    def blocked(row):
        show_id = space.catalogue[row]["id"]
        results = space.similar_to_show(show_id, limit=max_k)
        return [space.index_by_id[r["id"]] for r in results]

    systems["blocked"] = blocked

    # 1b. The same engine with the certificate penalty switched off. MATURITY_
    #     PENALTY = 0.5 was set by reasoning about U-rated cartoons ranking
    #     against 18-rated drama, and never measured; this is what measures it.
    #     It reaches past the public interface into _combine because the whole
    #     point is to run the deployed scoring MINUS one step - reimplementing
    #     the scoring here would test a different thing.
    def blocked_no_certificate(row):
        query = {name: matrix[row] for name, matrix in space.blocks.items()}
        combined, _ = space._combine(query, DEFAULT_WEIGHTS)
        return top_rows(combined, row, max_k)

    systems["blocked-cert"] = blocked_no_certificate

    # 2. The blocking ablation: same numbers, one flat vector, one cosine. The
    #    certificate penalty is kept so that blocking is the ONLY difference
    #    between this row and the one above - otherwise two changes would be
    #    measured as one.
    flat = unit_rows(np.hstack([space.blocks[b] for b in ("genre", "keywords", "structure")]))

    def flat_cosine(row):
        scores = flat @ flat[row]
        return top_rows(maturity_penalty(scores, space, row), row, max_k)

    systems["flat"] = flat_cosine

    # 3 and 4. The black-box baseline, without and with the certificate rule.
    if not args.skip_embeddings:
        embeddings = unit_rows(load_embeddings(space))

        def embedding_cosine(row):
            return top_rows(embeddings @ embeddings[row], row, max_k)

        def embedding_cosine_maturity(row):
            scores = embeddings @ embeddings[row]
            return top_rows(maturity_penalty(scores, space, row), row, max_k)

        systems["embeddings"] = embedding_cosine
        systems["embeddings+m"] = embedding_cosine_maturity

    # 5. Non-personalised control. The catalogue arrives from TMDB in
    #    popularity order, so the most popular shows are simply the first rows.
    #    Recommenders are routinely beaten by this; it is here to check.
    def popular(row):
        rows = [r for r in range(max_k + 1) if r != row]
        return rows[:max_k]

    systems["popular"] = popular

    # 6. Chance. Gives the table a floor, so a reader can see the scale the
    #    other numbers live on.
    rng = np.random.default_rng(SEED)

    def random_rows(row):
        picked = rng.choice(len(space.catalogue), size=max_k + 1, replace=False)
        return [r for r in picked if r != row][:max_k]

    systems["random"] = random_rows

    return systems


def load_embeddings(space):
    """
    Sentence-transformer vectors for every show, cached on disk.

    Deliberately built from the SAME text the trained model reads
    (`train_model.load_corpus`), so this baseline is a fair comparison rather
    than a straw man handed worse input.
    """
    if EMBEDDING_CACHE.exists():
        cached = np.load(EMBEDDING_CACHE)
        if len(cached["ids"]) == len(space.catalogue):
            print(f"Embeddings: reusing {EMBEDDING_CACHE.name}")
            order = {int(i): n for n, i in enumerate(cached["ids"])}
            return cached["vectors"][[order[s["id"]] for s in space.catalogue]]
        print("Embeddings: cache is a different size to the catalogue, rebuilding")

    # Imported here rather than at the top because it pulls in torch, which
    # takes several seconds and is not needed for the other five systems.
    # train_model first: importing it is what points HuggingFace's cache at the
    # D: drive, and C: does not have room for the download.
    from training.train_model import ENCODER, load_corpus

    from sentence_transformers import SentenceTransformer

    print(f"Embeddings: encoding {len(space.catalogue)} shows with {ENCODER} ...")
    _, corpus = load_corpus()
    encoder = SentenceTransformer(ENCODER)

    ids = [s["id"] for s in space.catalogue]
    vectors = encoder.encode(
        [corpus.get(i, s["name"]) for i, s in zip(ids, space.catalogue)],
        show_progress_bar=True,
        batch_size=16,
    )
    np.savez_compressed(EMBEDDING_CACHE, ids=np.array(ids), vectors=np.asarray(vectors))
    return np.asarray(vectors)


def bootstrap_ci(values, rng, confidence=95):
    """
    A 95% confidence interval for the mean of `values`, by resampling.

    The idea, which is easier than it sounds: pretend the query shows we have
    are the whole world, draw a same-sized sample from them WITH replacement,
    and take its mean. Do that 2,000 times and the middle 95% of those means is
    the range the true value plausibly sits in. No assumption that anything is
    normally distributed, which matters here because per-query precision is a
    handful of discrete values (0, 0.2, 0.4 ...) and nothing like a bell curve.

    Passed a difference between two systems it gives a PAIRED interval: both
    systems answered the same queries, so subtracting first and resampling the
    differences cancels out the fact that some shows are simply easier.
    """
    tail = (100 - confidence) / 2
    draws = rng.choice(values, size=(BOOTSTRAP_SAMPLES, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(draws, tail)), float(np.percentile(draws, 100 - tail))


# --------------------------------------------------------------- measurement


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--truth", default="tmdb_recommended",
        choices=["tmdb_recommended", "tmdb_similar"],
        help="which TMDB list to treat as the answer key (default: recommended, "
             "which covers 94%% of the catalogue; similar covers 61%%)",
    )
    parser.add_argument(
        "--queries", type=int, default=0,
        help="measure only this many query shows, for a quick run (default: all)",
    )
    parser.add_argument(
        "--skip-embeddings", action="store_true",
        help="leave out the sentence-transformer rows, which need torch",
    )
    args = parser.parse_args()

    space = FeatureSpace()
    truth = ground_truth(space, args.truth)
    systems = build_systems(space, args)

    # A separate query set per cut-off: to be scored at k, a show must have at
    # least k related titles available, or 1.0 would be unreachable and the
    # number would say more about TMDB's coverage than about the system.
    query_sets = {
        k: sorted(row for row, related in truth.items() if len(related) >= k)
        for k in K_VALUES
    }
    if args.queries:
        rng = np.random.default_rng(SEED)
        for k, rows in query_sets.items():
            if len(rows) > args.queries:
                query_sets[k] = sorted(rng.choice(rows, size=args.queries, replace=False).tolist())

    for k in K_VALUES:
        print(f"precision@{k}: {len(query_sets[k])} query shows")
    print()

    # Precision is kept per query rather than as a running total, because the
    # confidence intervals need the individual values, not just their mean.
    per_query, timings = {}, {}
    for name, rank in systems.items():
        started = time.perf_counter()
        per_query[name] = {}
        for k in K_VALUES:
            hits = np.empty(len(query_sets[k]), dtype=np.float64)
            for n, row in enumerate(query_sets[k]):
                ranked = list(rank(row))[:k]
                hits[n] = sum(1 for r in ranked if r in truth[row]) / k
            per_query[name][k] = hits
        timings[name] = time.perf_counter() - started

    rng = np.random.default_rng(SEED)
    results = {}
    for name in systems:
        entry = {"seconds": timings[name], "precision": {}, "ci": {}, "gap": {}}
        for k in K_VALUES:
            values = per_query[name][k]
            entry["precision"][k] = float(values.mean())
            entry["ci"][k] = bootstrap_ci(values, rng)
            if name != REFERENCE:
                # How far REFERENCE is ahead. An interval that stays above zero
                # means the lead is not an accident of which shows were asked.
                entry["gap"][k] = bootstrap_ci(per_query[REFERENCE][k] - values, rng)
        results[name] = entry

        line = f"  {name:14} "
        for k in K_VALUES:
            low, high = entry["ci"][k]
            line += f"P@{k} {entry['precision'][k]:.3f} [{low:.3f}, {high:.3f}]  "
        print(line + f"({entry['seconds']:.1f}s)")

    print()
    print(f"  gap to `{REFERENCE}` (95% CI on the paired difference):")
    for name, entry in results.items():
        if not entry["gap"]:
            continue
        parts = []
        for k in K_VALUES:
            low, high = entry["gap"][k]
            # Three outcomes, not two: the reference can also LOSE, and an
            # interval sitting entirely below zero says so.
            if low > 0:
                verdict = f"{REFERENCE} ahead"
            elif high < 0:
                verdict = f"{REFERENCE} BEHIND"
            else:
                verdict = "not separated"
            parts.append(f"@{k} {results[REFERENCE]['precision'][k] - entry['precision'][k]:+.3f} "
                         f"[{low:.3f}, {high:.3f}] {verdict}")
        print(f"  {name:14} " + "   ".join(parts))

    write_report(args, query_sets, results)


def write_report(args, query_sets, results):
    """Leave the table on disk so the dissertation can quote it verbatim."""
    lines = [
        "# Retrieval accuracy",
        "",
        f"Answer key: TMDB `{args.truth}`, restricted to titles inside this catalogue.",
        "It is another recommender's opinion, not truth - see the note in",
        "`evaluation/retrieval_accuracy.py`. Treat these numbers as a floor.",
        "",
        f"Intervals are 95% bootstrap ({BOOTSTRAP_SAMPLES} resamples of the query set).",
        "",
        "| System | " + " | ".join(f"precision@{k}" for k in K_VALUES) + " |",
        "|---|" + "---|" * len(K_VALUES),
    ]
    for name, entry in results.items():
        cells = []
        for k in K_VALUES:
            low, high = entry["ci"][k]
            cells.append(f"{entry['precision'][k]:.3f} [{low:.3f}, {high:.3f}]")
        lines.append(f"| `{name}` | " + " | ".join(cells) + " |")

    lines += [
        "",
        f"## Gap to `{REFERENCE}`",
        "",
        "Paired difference and its 95% interval. Both systems answered the same",
        "queries, so subtracting per query cancels out shows that are simply",
        "easier. An interval entirely above zero is a difference that survives",
        "resampling; one that straddles zero is not separated by this evidence.",
        "",
        "| System | " + " | ".join(f"@{k}" for k in K_VALUES) + " |",
        "|---|" + "---|" * len(K_VALUES),
    ]
    for name, entry in results.items():
        if not entry["gap"]:
            continue
        cells = []
        for k in K_VALUES:
            low, high = entry["gap"][k]
            gap = results[REFERENCE]["precision"][k] - entry["precision"][k]
            cells.append(f"{gap:+.3f} [{low:.3f}, {high:.3f}]")
        lines.append(f"| `{name}` | " + " | ".join(cells) + " |")

    lines += [
        "",
        "Query shows per cut-off (a show is only asked at k if TMDB names at",
        "least k related titles that this catalogue contains):",
        "",
    ] + [f"- precision@{k}: {len(query_sets[k])} shows" for k in K_VALUES] + [""]

    path = HERE / "retrieval_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWritten to {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
