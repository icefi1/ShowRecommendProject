"""
FastAPI service for the show recommender.

Run:
    venv\\Scripts\\python -m uvicorn app.main:app --reload --port 8000

Then open http://127.0.0.1:8000

Endpoints
---------
GET  /api/search?q=          find shows by title
GET  /api/similar/{show_id}  recommend against a chosen show
POST /api/preference         recommend against dial settings, no title needed
GET  /api/axes               the axes and genres the interface can offer

The feature space is loaded once at startup, not per request. Every query is a
matrix-vector product over the whole catalogue.
"""

import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.similarity import DEFAULT_WEIGHTS, FeatureSpace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

app = FastAPI(title="Show Recommender", version="0.1.0")

# Loaded once. A FeatureSpace is read-only after construction, so sharing one
# across requests is safe and avoids re-reading 500 rows per call.
space = FeatureSpace()


class Weights(BaseModel):
    """Block weights. Set at query time - this is what makes the space steerable."""

    genre: float = Field(DEFAULT_WEIGHTS["genre"], ge=0, le=1)
    keywords: float = Field(DEFAULT_WEIGHTS["keywords"], ge=0, le=1)
    structure: float = Field(DEFAULT_WEIGHTS["structure"], ge=0, le=1)


class PreferenceQuery(BaseModel):
    """A query built entirely from dials, with no show to anchor on."""

    structure: dict[str, float] = Field(default_factory=dict)
    genres: list[str] = Field(default_factory=list)
    weights: Weights = Field(default_factory=Weights)
    limit: int = Field(12, ge=1, le=50)


def load_predicted_axes():
    """
    Per-show predicted axis scores, if the model has been trained.

    These are the 37 schema axes the trained model produces. They are shown in
    the detail panel but are deliberately NOT used for ranking yet - the model
    is trained on 50 labels and its magnitudes are compressed, so it would make
    recommendations worse. Displaying them keeps the two separable: you can see
    what the model thinks without it silently steering results.
    """
    path = ROOT / "training" / "predictions.csv"
    if not path.exists():
        return {}, []

    import csv

    with open(path, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        return {}, []

    axes = [key for key in rows[0] if key not in ("id", "name")]
    return {
        int(row["id"]): {axis: float(row[axis]) for axis in axes} for row in rows
    }, axes


PREDICTED, PREDICTED_AXES = load_predicted_axes()


@app.get("/api/show/{show_id}")
def show_detail(show_id: int):
    """Everything the detail panel needs for one show."""
    index = space.index_by_id.get(show_id)
    if index is None:
        raise HTTPException(status_code=404, detail="Unknown show id")

    show = dict(space.catalogue[index])

    # Measured structure axes, as percentiles across the catalogue.
    structure_names = space.block_labels["structure"]
    show["structure"] = {
        name: round(float(space.blocks["structure"][index][column]), 3)
        for column, name in enumerate(structure_names)
    }

    # The show's most distinctive keywords - highest IDF weight, so the ones
    # that actually say something rather than "drama".
    keyword_row = space.blocks["keywords"][index]
    vocabulary = space.block_labels["keywords"]
    ranked = sorted(range(len(vocabulary)), key=lambda i: -keyword_row[i])
    show["top_keywords"] = [vocabulary[i] for i in ranked[:12] if keyword_row[i] > 0]

    predicted = PREDICTED.get(show_id, {})
    show["predicted"] = {k: round(v, 3) for k, v in predicted.items()}
    # Ranked so the panel can lead with what most describes the show.
    show["predicted_top"] = sorted(
        predicted.items(), key=lambda kv: -kv[1]
    )[:10]
    show["has_model"] = bool(predicted)

    return show


@app.get("/api/axes")
def axes():
    """Everything the interface needs to draw its controls."""
    return {
        "genres": space.block_labels["genre"],
        "structure": space.block_labels["structure"],
        "keyword_count": len(space.block_labels["keywords"]),
        "catalogue_size": len(space.catalogue),
        "default_weights": DEFAULT_WEIGHTS,
    }


@app.get("/api/search")
def search(q: str = "", limit: int = 10):
    results = space.search(q, limit=limit)
    return {"query": q, "results": results}


@app.get("/api/similar/{show_id}")
def similar(
    show_id: int,
    genre: float = DEFAULT_WEIGHTS["genre"],
    keywords: float = DEFAULT_WEIGHTS["keywords"],
    structure: float = DEFAULT_WEIGHTS["structure"],
    limit: int = 12,
):
    if show_id not in space.index_by_id:
        raise HTTPException(status_code=404, detail="Unknown show id")

    started = time.perf_counter()
    results = space.similar_to_show(
        show_id,
        weights={"genre": genre, "keywords": keywords, "structure": structure},
        limit=limit,
    )
    elapsed = (time.perf_counter() - started) * 1000

    return {
        "query_show": space.catalogue[space.index_by_id[show_id]],
        "results": results,
        # Surfaced in the interface so the efficiency claim is visible rather
        # than asserted.
        "query_ms": round(elapsed, 3),
    }


@app.post("/api/preference")
def preference(query: PreferenceQuery):
    started = time.perf_counter()
    results = space.by_preference(
        query.structure,
        query.genres,
        weights=query.weights.model_dump(),
        limit=query.limit,
    )
    elapsed = (time.perf_counter() - started) * 1000
    return {"results": results, "query_ms": round(elapsed, 3)}


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
