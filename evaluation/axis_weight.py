"""
Should the 37 predicted axes join the ranking, and at what weight? (S9.5)

Everything else in the evaluation ranks on TMDB metadata. The 37 axes the model
predicts are shown in the interface but do not affect results, which leaves an
awkward gap: the axes are the machine-learning contribution, and nothing had
measured whether they help retrieve anything.

Ranking on the axes ALONE scores 0.033 against 0.134 for the metadata blocks
(see retrieval_accuracy.py), so they are much weaker on their own - though still
eleven times chance, so they are not noise. The question that actually matters
is whether they add anything to the blocks rather than replace them:

    score = (1 - w) * blocked_score  +  w * axis_similarity

WHY THIS SCRIPT EXISTS SEPARATELY

Because w has to be chosen, and choosing it on the same query shows the result
is reported on inflates that result. The first run of this did exactly that: it
found w = 0.15 lifting precision@5 from 0.134 to 0.145 with a confidence
interval clear of zero, which looked like a solid win and was not one.

So the query shows are split in half. The weight is picked on one half and
measured on the other, which it has never seen. That is the difference between
"the best weight I could find, measured where I found it" and "a weight that
works on shows it was not chosen for".

Run:  venv\\Scripts\\python evaluation/axis_weight.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.retrieval_accuracy import (  # noqa: E402
    DEFAULT_WEIGHTS,
    SEED,
    FeatureSpace,
    bootstrap_ci,
    ground_truth,
    load_predicted_axes,
    maturity_penalty,
    top_rows,
    unit_rows,
)

WEIGHTS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
K = 5


def main():
    space = FeatureSpace()
    truth = ground_truth(space, "tmdb_recommended")

    axes = load_predicted_axes(space)
    if axes is None:
        raise SystemExit("Train the model first - training/predictions.csv is missing.")

    # Centred, because every show scores somewhat on drama and tense and the
    # rest, so raw vectors all point the same way and cosine ends up describing
    # the average show rather than this one.
    centred = unit_rows(axes - axes.mean(axis=0))

    queries = sorted(row for row, related in truth.items() if len(related) >= K)
    rng = np.random.default_rng(SEED)
    shuffled = rng.permutation(queries)
    half = len(shuffled) // 2
    tune, test = sorted(shuffled[:half]), sorted(shuffled[half:])
    print(f"{len(tune)} query shows to choose the weight on, {len(test)} held back\n")

    def precision(rows, weight):
        out = np.empty(len(rows))
        for i, row in enumerate(rows):
            query = {name: matrix[row] for name, matrix in space.blocks.items()}
            combined, _ = space._combine(query, DEFAULT_WEIGHTS)
            combined = maturity_penalty(combined, space, row)
            if weight:
                combined = (1 - weight) * combined + weight * (centred @ centred[row])
            ranked = top_rows(combined, row, K)
            out[i] = sum(1 for r in ranked if r in truth[row]) / K
        return out

    print("weight   precision@5 on the tuning half")
    best, best_score = 0.0, -1.0
    for weight in WEIGHTS:
        score = precision(tune, weight).mean()
        if score > best_score:
            best, best_score = weight, score
        print(f"  {weight:.2f}   {score:.4f}")
    print(f"\nBest on the tuning half: w = {best:.2f}")

    plain = precision(test, 0.0)
    mixed = precision(test, best)
    low, high = bootstrap_ci(mixed - plain, np.random.default_rng(SEED))

    print(f"\nHeld-out half, {len(test)} query shows:")
    print(f"  blocked alone            {plain.mean():.4f}")
    print(f"  blocked + axes at {best:.2f}   {mixed.mean():.4f}")
    print(f"  difference               {mixed.mean() - plain.mean():+.4f}"
          f"  95% CI [{low:+.4f}, {high:+.4f}]")

    if low > 0:
        print("\n  The gain survives on shows the weight was not chosen for.")
    else:
        print("\n  The interval crosses zero, so this is not established. The direction")
        print("  is positive and the tuning curve is smooth, which is suggestive - but")
        print("  suggestive is not a result, and the honest move is to leave the axes")
        print("  out of ranking until more labels are in.")


if __name__ == "__main__":
    main()
