"""
Is the 37-axis schema actually 37 things? (report S9.4)

The schema was written by hand before any scoring happened, which is the right
way round - it stops a model inventing near-duplicate axes - but it also means
nothing guaranteed the axes are independent. If `creepy`, `unsettling` and
`tense` always move together, they are one axis wearing three hats: they cost
three dimensions, three vote targets and three lines of explanation, and give
the distance metric one dimension of information.

Two measurements, both over the predicted scores for all 3,542 shows:

  1. PCA. How many components are needed to keep most of the variance? If 37
     axes collapse into 12, the schema is carrying redundancy and should be
     pruned.
  2. The most correlated pairs, which names the specific axes to look at.

PCA is done on standardised axes (each centred and scaled to unit variance),
which is PCA on the correlation matrix rather than the covariance matrix. That
matters here: the axes have very different spreads - `documentary` is nearly
always near zero while `drama` is high everywhere - and without standardising,
the components would mostly describe which axes happen to have big numbers
rather than which axes move together.

Run:  venv\\Scripts\\python evaluation/feature_structure.py
"""

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from labelling.schema import FACT_AXES  # noqa: E402

PREDICTIONS = ROOT / "training" / "predictions.csv"

# Variance levels to report the component count for.
THRESHOLDS = (0.80, 0.90, 0.95)

# How many correlated pairs to name.
TOP_PAIRS = 12


def load_labels():
    """
    The hand/LLM-assigned labels, one row per labelled show.

    This is the control for the whole analysis. The predictions come from a
    ridge head over a single 384-dimensional embedding, so every axis is a
    linear function of the same features and the model can manufacture
    correlation that the schema does not have - especially at 78 labels, where
    shrinkage pulls the fitted directions towards each other. Running the same
    PCA on the labels themselves says which of the two is happening.
    """
    path = ROOT / "labelling" / "labels.jsonl"
    if not path.exists():
        return None, None

    import json

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    if not records:
        return None, None

    axes = list(records[0]["labels"])
    rows = [[float(r["labels"][axis]) for axis in axes] for r in records]
    return np.array(rows, dtype=np.float64), axes


def components_needed(matrix, thresholds=THRESHOLDS):
    """How many components hold each share of the variance, for one matrix."""
    _, singular, _ = np.linalg.svd(standardise(matrix), full_matrices=False)
    variance = singular ** 2
    cumulative = np.cumsum(variance / variance.sum())
    return {t: int(np.searchsorted(cumulative, t) + 1) for t in thresholds}


def load_scores():
    """The predicted score matrix: one row per show, one column per axis."""
    if not PREDICTIONS.exists():
        raise SystemExit(
            "training/predictions.csv is missing. Run training/train_model.py first."
        )
    with open(PREDICTIONS, encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        axes = header[2:]
        rows = [[float(v) for v in row[2:]] for row in reader]
    return np.array(rows, dtype=np.float64), axes


def standardise(matrix):
    """
    Centre each axis on zero and scale it to unit spread.

    Written out rather than reached for from a library because it is two lines
    and the whole result depends on it. An axis with no variance at all would
    divide by zero, so those are left as zeros - they carry no information for
    PCA either way.
    """
    centred = matrix - matrix.mean(axis=0)
    spread = matrix.std(axis=0)
    return np.divide(centred, spread, out=np.zeros_like(centred), where=spread > 0)


def main():
    scores, axes = load_scores()
    print(f"{scores.shape[0]} shows, {scores.shape[1]} axes\n")

    standardised = standardise(scores)

    # PCA via SVD on the standardised matrix. Eigenvalues of the correlation
    # matrix are the squared singular values, scaled by the sample count.
    _, singular, components = np.linalg.svd(standardised, full_matrices=False)
    variance = singular ** 2
    explained = variance / variance.sum()
    cumulative = np.cumsum(explained)

    print("Variance explained by the first components:")
    for i in range(min(10, len(explained))):
        bar = "#" * int(explained[i] * 120)
        print(f"  PC{i + 1:<3} {explained[i]:6.1%}  cumulative {cumulative[i]:6.1%}  {bar}")

    print()
    for threshold in THRESHOLDS:
        needed = int(np.searchsorted(cumulative, threshold) + 1)
        print(f"  {needed:2} of {len(axes)} components hold {threshold:.0%} of the variance")

    # Kaiser's rule: on standardised data every axis contributes 1 unit of
    # variance, so a component worth keeping should explain more than one axis
    # does on its own. Crude, and known to over-retain, but it is the standard
    # first cut and easy to defend.
    eigenvalues = variance / (scores.shape[0] - 1)
    kaiser = int((eigenvalues > 1).sum())
    print(f"  {kaiser:2} components have an eigenvalue above 1 (Kaiser's rule)")

    print("\nWhat the first three components are made of:")
    for i in range(min(3, len(components))):
        loading = components[i]
        order = np.argsort(-np.abs(loading))[:6]
        parts = [f"{axes[j]} {loading[j]:+.2f}" for j in order]
        print(f"  PC{i + 1}: " + ",  ".join(parts))

    # ----------------------------------------------------- redundant pairs
    correlation = np.corrcoef(standardised, rowvar=False)
    np.fill_diagonal(correlation, 0.0)

    pairs = []
    for a in range(len(axes)):
        for b in range(a + 1, len(axes)):
            pairs.append((abs(correlation[a, b]), correlation[a, b], axes[a], axes[b]))
    pairs.sort(reverse=True)

    print(f"\nMost correlated axis pairs (candidates for merging):")
    for _, value, first, second in pairs[:TOP_PAIRS]:
        note = "  <- both facts from TMDB" if first in FACT_AXES and second in FACT_AXES else ""
        print(f"  {value:+.2f}  {first:<20} {second}{note}")

    print("\nLeast correlated with everything (the axes doing their own work):")
    strongest = np.abs(correlation).max(axis=1)
    for j in np.argsort(strongest)[:6]:
        print(f"  {strongest[j]:.2f}  {axes[j]}")

    # ------------------------------------------------- schema or model?
    labels, label_axes = load_labels()
    if labels is None:
        print("\nNo labels.jsonl, so the schema-versus-model check is skipped.")
        return

    print(f"\nSame analysis on the {labels.shape[0]} labelled shows, for comparison.")
    print("If the labels need about as few components as the predictions do, the")
    print("redundancy is in the schema. If they need many more, the collapse is the")
    print("model's - one ridge head over one embedding, fitted on few examples.\n")

    predicted_needs = components_needed(scores)
    label_needs = components_needed(labels)
    print(f"  {'variance kept':<16} {'labels':>8} {'predictions':>12}")
    for threshold in THRESHOLDS:
        print(f"  {threshold:>13.0%}    {label_needs[threshold]:>8} {predicted_needs[threshold]:>12}")


if __name__ == "__main__":
    main()
