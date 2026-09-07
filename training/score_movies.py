"""
Score the film catalogue with the model trained on television.

Reads   training/model.joblib, tmdb/movies_raw.json
Writes  training/predictions_movie.csv

No retraining happens here. The model is a ridge head over a frozen sentence
transformer, so it reads text and returns 37 numbers; nothing in it is specific
to television. Pointing it at films is a matter of assembling film text in the
same shape and running the encoder over it.

THE HONEST CAVEAT, WHICH BELONGS IN THE REPORT

This is transfer, not training. Every one of the 78 labelled examples is a
television series, and series text carries a block of sampled episode summaries
that film text has no equivalent of. So the model is being asked about a kind of
document it never saw while learning.

Two reasons to expect it to survive that better than it sounds:

  1. The encoder is frozen and general. It was trained on a very large corpus of
     ordinary English, not on television, so "a haunted house where something
     watches from the dark" embeds the same way whether it describes a series or
     a film.
  2. Both text shapes are truncated to the encoder's 256-token window anyway, so
     the series text the model learned from was mostly title, genres, overview
     and keywords - which is exactly what film text is.

What cannot be claimed is that the film scores are as good as the television
ones. Nothing has measured them. They are shown in the interface and, like the
television predictions, they do not drive ranking.

Run:  venv\\Scripts\\python training/score_movies.py
"""

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

MODEL = HERE / "model.joblib"
OUT = HERE / "predictions_movie.csv"


def build_film_text(movie):
    """
    Everything the model reads about one film.

    Deliberately the same layout as labelling/label_shows.build_show_text, minus
    the two parts a film does not have: episode summaries and TMDB reviews. The
    headings are kept identical because the model learned from text with those
    headings in it, and matching them costs nothing.
    """
    parts = [
        f"TITLE: {movie.get('name', '')} ({(movie.get('first_air_date') or '')[:4]})",
        f"TMDB GENRES: {', '.join(g['name'] for g in movie.get('genres', [])) or 'none listed'}",
        f"RUNTIME: {movie.get('runtime') or 'unknown'} minutes",
        f"\nOVERVIEW:\n{movie.get('overview') or '(none)'}",
    ]

    keywords = [k["name"] for k in movie.get("keywords", {}).get("results", [])]
    if keywords:
        parts.append(f"\nTMDB KEYWORDS: {', '.join(keywords)}")

    return "\n".join(parts)


def main():
    if not MODEL.exists():
        raise SystemExit("No training/model.joblib. Run training/train_model.py first.")

    movies_path = ROOT / "tmdb" / "movies_raw.json"
    if not movies_path.exists():
        raise SystemExit("No tmdb/movies_raw.json. Run tmdb/fetch_movies.py first.")

    bundle = joblib.load(MODEL)
    model, encoder_name, axes = bundle["model"], bundle["encoder_name"], bundle["axes"]
    print(f"Model: ridge head over {encoder_name}, {len(axes)} axes")

    movies = json.loads(movies_path.read_text(encoding="utf-8"))
    print(f"Films: {len(movies)}")

    # Imported here rather than at the top so the import error, if torch is
    # missing, arrives after the cheap checks above.
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(encoder_name)
    texts = [build_film_text(m) for m in movies]

    started = time.monotonic()
    print("Encoding ...")
    features = encoder.encode(texts, show_progress_bar=True, batch_size=16)

    # Clipped because ridge is a linear model and will happily predict 1.06 or
    # -0.03 for an axis defined on 0 to 1. The same clip is applied to the
    # television predictions in train_model.py.
    predictions = np.clip(model.predict(features), 0.0, 1.0)

    with open(OUT, "w", encoding="utf-8", newline="") as handle:
        handle.write("id,name," + ",".join(axes) + "\n")
        for row, movie in enumerate(movies):
            name = (movie.get("name") or "").replace('"', "'")
            scores = ",".join(f"{v:.4f}" for v in predictions[row])
            handle.write(f'{movie["id"]},"{name}",{scores}\n')

    elapsed = time.monotonic() - started
    print(f"\nWrote {OUT.name}: {len(movies)} films in {elapsed / 60:.1f} min")

    # A few named films as a sanity check. If a horror film does not score
    # highly on horror, something is wrong with the text assembly rather than
    # with the model.
    by_name = {m["name"]: i for i, m in enumerate(movies)}
    axis_index = {axis: i for i, axis in enumerate(axes)}
    for title in ("The Conjuring", "Hereditary", "The Dark Knight", "Forrest Gump"):
        if title not in by_name:
            continue
        row = predictions[by_name[title]]
        shown = {a: round(float(row[axis_index[a]]), 2)
                 for a in ("horror", "jumpscares", "tense", "warm", "action")
                 if a in axis_index}
        print(f"  {title:20} {shown}")


if __name__ == "__main__":
    main()
