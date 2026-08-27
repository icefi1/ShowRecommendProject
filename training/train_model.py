"""
Train a multi-label regression model to predict feature scores from show text.

This is stage 2 of report S6.2, and it is the actual contribution. The LLM
labels are training data; this model is the artefact - frozen weights, free to
run, reproducible, and able to score a show released after any LLM's cutoff.

Reads   labelling/labels.jsonl, tmdb/shows_raw.json, tmdb/episodes.json
Writes  training/model.joblib, training/predictions.csv, training/report.md

Architecture: frozen sentence-transformer embeddings + ridge regression head.

Why not fine-tune DistilBERT end to end? With a few hundred labelled shows,
fine-tuning ~66M parameters overfits almost immediately - the model memorises
the training titles rather than learning what the words mean. A frozen encoder
plus a linear head fits roughly 384 x 37 parameters, which is the right
capacity for this much data. It also trains in seconds on CPU, which matters
when there is no GPU. If the labelled set later reaches a few thousand shows,
fine-tuning becomes worth revisiting - that is a scale question, not a
correctness one.

Run:
    python training/train_model.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent

# The C: drive on this machine runs at ~0.1GB free, and HuggingFace caches
# models under the user profile by default, which would fail the download.
os.environ.setdefault("HF_HOME", "D:/caches/hf")

sys.path.insert(0, str(ROOT))
from labelling.label_shows import build_show_text  # noqa: E402
from labelling.schema import AXES, AXIS_NAMES  # noqa: E402

import joblib  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402
from sklearn.metrics import mean_absolute_error, r2_score  # noqa: E402

# 384-dimensional, ~90MB, fast on CPU. The standard general-purpose sentence
# encoder and a defensible default; swapping it is a one-line ablation.
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"

# Ridge rather than plain least squares because 384 features against a few
# hundred examples is an underdetermined problem - the L2 penalty is what stops
# the head fitting noise. Alpha is the strength of that penalty.
ALPHA = 1.0


def load_corpus():
    """Show text for the whole catalogue, keyed by TMDB id."""
    shows = json.loads((ROOT / "tmdb" / "shows_raw.json").read_text(encoding="utf-8"))

    episodes_by_show = {}
    episodes_path = ROOT / "tmdb" / "episodes.json"
    if episodes_path.exists():
        for episode in json.loads(episodes_path.read_text(encoding="utf-8")):
            episodes_by_show.setdefault(episode["show_id"], []).append(episode)

    return shows, {s["id"]: build_show_text(s, episodes_by_show) for s in shows}


def load_labels():
    """The LLM-labelled training set."""
    path = ROOT / "labelling" / "labels.jsonl"
    if not path.exists():
        raise SystemExit(
            "No labelling/labels.jsonl found.\n"
            "Run: python labelling/label_shows.py --limit 20"
        )

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise SystemExit("labels.jsonl is empty.")
    return records


def evaluate(features, targets, n_splits=5):
    """
    Per-axis cross-validated scores.

    Held-out evaluation matters more than training fit here: a ridge head can
    always drive training error down, and the question is whether it generalises
    to a show it has not seen.
    """
    n_splits = min(n_splits, len(features))
    if n_splits < 2:
        return None

    folds = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    predictions = np.zeros_like(targets)

    for train_index, test_index in folds.split(features):
        model = Ridge(alpha=ALPHA)
        model.fit(features[train_index], targets[train_index])
        predictions[test_index] = model.predict(features[test_index])

    predictions = np.clip(predictions, 0.0, 1.0)

    rows = []
    for column, name in enumerate(AXIS_NAMES):
        truth, guess = targets[:, column], predictions[:, column]
        rows.append({
            "axis": name,
            "block": next(b for n, b, _ in AXES if n == name),
            "mae": mean_absolute_error(truth, guess),
            # R2 is undefined when the axis has no variance in the labels -
            # every show scored the same, so there is nothing to predict.
            "r2": r2_score(truth, guess) if truth.std() > 1e-9 else float("nan"),
            "label_mean": truth.mean(),
            "label_std": truth.std(),
        })
    return rows


def main():
    records = load_labels()
    shows, corpus = load_corpus()
    print(f"{len(records)} labelled shows, {len(corpus)} in catalogue")

    if len(records) < 20:
        print(f"\nWARNING: {len(records)} labels is very few. Cross-validated numbers")
        print("below will be noisy and should not be quoted as results yet.\n")

    print(f"Loading encoder {ENCODER} ...")
    encoder = SentenceTransformer(ENCODER)

    train_texts = [corpus[r["id"]] for r in records if r["id"] in corpus]
    kept = [r for r in records if r["id"] in corpus]
    targets = np.array([[r["labels"][name] for name in AXIS_NAMES] for r in kept])

    print(f"Encoding {len(train_texts)} training texts ...")
    features = encoder.encode(train_texts, show_progress_bar=False, batch_size=16)

    print("Cross-validating ...")
    rows = evaluate(np.asarray(features), targets)

    print("Fitting final model on all labels ...")
    model = Ridge(alpha=ALPHA)
    model.fit(features, targets)
    joblib.dump({"model": model, "encoder_name": ENCODER, "axes": AXIS_NAMES}, HERE / "model.joblib")

    # The point of the whole exercise: score every show, including the ones
    # never sent to the LLM.
    print(f"Scoring all {len(corpus)} shows ...")
    all_ids = [s["id"] for s in shows]
    all_features = encoder.encode([corpus[i] for i in all_ids], show_progress_bar=False, batch_size=16)
    all_predictions = np.clip(model.predict(all_features), 0.0, 1.0)

    with open(HERE / "predictions.csv", "w", encoding="utf-8", newline="") as handle:
        handle.write("id,name," + ",".join(AXIS_NAMES) + "\n")
        for row, show in enumerate(shows):
            name = (show.get("name") or "").replace('"', "'")
            scores = ",".join(f"{v:.4f}" for v in all_predictions[row])
            handle.write(f'{show["id"]},"{name}",{scores}\n')

    # ------------------------------------------------------------- report
    lines = [
        "# Model training report",
        "",
        f"- Encoder: `{ENCODER}` (frozen)",
        f"- Head: ridge regression, alpha={ALPHA}",
        f"- Training labels: {len(kept)} shows",
        f"- Predicted: {len(shows)} shows",
        "",
    ]

    if rows:
        ordered = sorted(rows, key=lambda r: (np.isnan(r["r2"]), -r["r2"]))
        lines += [
            "## Cross-validated performance (5-fold, held out)",
            "",
            "R2 above 0 means the model beats predicting the mean for that axis.",
            "",
            "| Axis | Block | MAE | R2 | label mean | label sd |",
            "|---|---|---|---|---|---|",
        ]
        for r in ordered:
            r2 = "n/a" if np.isnan(r["r2"]) else f"{r['r2']:.3f}"
            lines.append(
                f"| {r['axis']} | {r['block']} | {r['mae']:.3f} | {r2} | "
                f"{r['label_mean']:.2f} | {r['label_std']:.2f} |"
            )

        usable = [r for r in rows if not np.isnan(r["r2"])]
        if usable:
            beat = sum(1 for r in usable if r["r2"] > 0)
            lines += [
                "",
                f"- Mean MAE across axes: **{np.mean([r['mae'] for r in usable]):.3f}**",
                f"- Axes beating the mean baseline: **{beat}/{len(usable)}**",
            ]

    (HERE / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\nWrote model.joblib, predictions.csv and report.md")
    if rows:
        usable = [r for r in rows if not np.isnan(r["r2"])]
        if usable:
            print(f"Mean MAE {np.mean([r['mae'] for r in usable]):.3f}, "
                  f"{sum(1 for r in usable if r['r2'] > 0)}/{len(usable)} axes beat the mean baseline")


if __name__ == "__main__":
    main()
