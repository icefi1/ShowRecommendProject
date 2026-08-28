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

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.accounts import SESSION_DAYS as ACCOUNT_SESSION_DAYS
from app.accounts import AccountStore
from app.crowd import CrowdStore
from app.crowd import fuse as crowd_fuse
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

# Vote storage. SQLite: one file, no server, standard library - the right size
# for this, and it keeps the raw votes as research data rather than only a cache.
crowd = CrowdStore()


accounts = AccountStore()

SESSION_COOKIE = "sr_session"


def current_user(request: Request):
    """The signed-in user, or None. Read from an httpOnly cookie."""
    return accounts.user_for_token(request.cookies.get(SESSION_COOKIE))


def require_user(request: Request):
    user = current_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Sign in to do that.")
    return user


def set_session_cookie(response: Response, token: str):
    """
    httpOnly so page scripts cannot read the token, which means an XSS bug
    cannot exfiltrate a session. SameSite=Lax blocks it being sent on
    cross-site form posts. secure=False only because this runs over plain
    http on localhost - it must be True behind HTTPS.
    """
    response.set_cookie(
        SESSION_COOKIE, token,
        httponly=True, samesite="lax", secure=False,
        max_age=60 * 60 * 24 * ACCOUNT_SESSION_DAYS, path="/",
    )


class Credentials(BaseModel):
    username: str = Field(min_length=1, max_length=32)
    password: str = Field(min_length=1, max_length=256)


@app.post("/api/signup")
def signup(creds: Credentials, response: Response):
    user_id, error = accounts.create_user(creds.username, creds.password)
    if error:
        raise HTTPException(status_code=400, detail=error)
    set_session_cookie(response, accounts.open_session(user_id))
    return {"username": creds.username}


@app.post("/api/login")
def login(creds: Credentials, response: Response):
    user_id = accounts.authenticate(creds.username, creds.password)
    if user_id is None:
        # Deliberately does not say which of the two was wrong - saying
        # "no such user" confirms which usernames exist.
        raise HTTPException(status_code=401, detail="Incorrect username or password.")
    set_session_cookie(response, accounts.open_session(user_id))
    return {"username": creds.username}


@app.post("/api/logout")
def logout(request: Request, response: Response):
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        accounts.close_session(token)
    response.delete_cookie(SESSION_COOKIE, path="/")
    return {"ok": True}


@app.get("/api/me")
def me(request: Request):
    user = current_user(request)
    return {"user": user}


class LibraryEntry(BaseModel):
    show_id: int
    # None removes the show from the library entirely.
    status: str | None = Field(default=None, pattern="^(watchlist|watching|watched)$")


@app.post("/api/library")
def set_library(entry: LibraryEntry, request: Request):
    user = require_user(request)
    if entry.show_id not in space.index_by_id:
        raise HTTPException(status_code=404, detail="Unknown show id")
    accounts.set_status(user["id"], entry.show_id, entry.status)
    return {"show_id": entry.show_id, "status": entry.status}


@app.get("/api/library")
def get_library(request: Request, status: str = ""):
    user = require_user(request)
    rows = accounts.library(user["id"], status or None)
    for row in rows:
        index = space.index_by_id.get(row["show_id"])
        if index is not None:
            row["show"] = space.catalogue[index]
    return {"rows": [r for r in rows if "show" in r]}


@app.get("/api/history")
def get_history(request: Request):
    user = require_user(request)
    rows = accounts.history(user["id"])
    for row in rows:
        index = space.index_by_id.get(row["show_id"])
        if index is not None:
            row["show"] = space.catalogue[index]
    return {"rows": [r for r in rows if "show" in r]}


class ReviewBody(BaseModel):
    show_id: int
    body: str = Field(min_length=1, max_length=8000)


@app.post("/api/review")
def post_review(review: ReviewBody, request: Request):
    """
    Write or update this user's review of a show.

    Report S4 note: this is the corpus TMDB could not provide. Reviews written
    here are experiential text tied to a known show id, and feed back into the
    labelling pipeline through accounts.review_text_for().
    """
    user = require_user(request)
    if review.show_id not in space.index_by_id:
        raise HTTPException(status_code=404, detail="Unknown show id")
    error = accounts.put_review(user["id"], review.show_id, review.body)
    if error:
        raise HTTPException(status_code=400, detail=error)
    return {"ok": True, "reviews": accounts.reviews_for(review.show_id)}


class Vote(BaseModel):
    """
    One descriptive vote on one axis of one show.

    Note what is NOT here: there is no whole-show vote and no rating field. The
    only thing a person can express is whether a named axis describes a show
    well, which is the separation report S6.3 requires - blend descriptive and
    evaluative votes and every popular show drifts high on every flattering
    axis.
    """

    show_id: int
    axis: str
    direction: str = Field(pattern="^(up|down|neutral)$")
    # What the voter had on screen. "Higher" is meaningless without it.
    score_shown: float = Field(ge=0.0, le=1.0)


@app.post("/api/vote")
def vote(v: Vote, request: Request):
    """
    Cast a descriptive vote. Requires an account.

    The identity comes from the session, never from the request body. The
    previous localStorage id could be reset at will, which made every tally
    meaningless; a vote only stands for a person if it is tied to one.
    """
    user = require_user(request)
    if v.show_id not in space.index_by_id:
        raise HTTPException(status_code=404, detail="Unknown show id")
    if v.axis not in PREDICTED_AXES:
        raise HTTPException(status_code=400, detail="Unknown axis")

    voter = f"user:{user['id']}"
    crowd.cast(v.show_id, v.axis, v.direction, v.score_shown, voter)
    priors = PREDICTED.get(v.show_id, {})
    return {"axis": v.axis, "state": crowd.state_for_show(v.show_id, priors, voter)[v.axis]}


@app.get("/api/disagreements")
def disagreements(min_votes: int = 3, limit: int = 40):
    """
    Where the crowd has moved the model most - the labelled error data of S6.3.

    Reported rather than silently applied: an axis the crowd consistently
    overrides is a measurement of where the model is weak, and that belongs in
    the evaluation chapter.
    """
    rows = []
    for row in crowd.disagreements(limit=limit, min_votes=min_votes):
        priors = PREDICTED.get(row["show_id"], {})
        prior = priors.get(row["axis"])
        if prior is None:
            continue
        posterior = crowd_fuse(prior, row["n_votes"], row["sum_targets"])
        index = space.index_by_id.get(row["show_id"])
        rows.append({
            "show": space.catalogue[index]["name"] if index is not None else row["show_id"],
            "axis": row["axis"],
            "model": round(prior, 3),
            "crowd": round(posterior, 3),
            "drift": round(posterior - prior, 3),
            "n_votes": row["n_votes"],
        })
    rows.sort(key=lambda r: -abs(r["drift"]))
    return {"rows": rows, "totals": crowd.totals()}


@app.get("/api/show/{show_id}")
def show_detail(show_id: int, request: Request):
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
    show["has_model"] = bool(predicted)

    # Every axis carries its model prior, its crowd-corrected score, the vote
    # tallies, and this voter's own standing vote. With no votes anywhere the
    # scores equal the priors exactly, so the page renders identically before
    # and after the crowd layer exists.
    user = current_user(request)
    voter = f"user:{user['id']}" if user else None
    show["axes"] = crowd.state_for_show(show_id, predicted, voter)

    # Per-user extras. A guest sees the show; a signed-in user also sees their
    # own library status and review, and the visit is recorded.
    show["signed_in"] = user is not None
    show["library_status"] = accounts.status_for(user["id"], show_id) if user else None
    show["my_review"] = accounts.my_review(user["id"], show_id) if user else None
    show["reviews"] = accounts.reviews_for(show_id)
    if user:
        accounts.record_view(user["id"], show_id)

    # Ranked by the corrected score, so the panel leads with what most
    # describes the show according to model and crowd together.
    show["axes_top"] = sorted(
        show["axes"].items(), key=lambda kv: -kv[1]["score"]
    )[:12]

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
def index(request: Request):
    """Guests get the landing page; signed-in users go straight to the app."""
    page = "index.html" if current_user(request) else "home.html"
    return FileResponse(HERE / "static" / page)


@app.get("/app")
def app_page():
    """The recommender itself. Reachable directly, e.g. after signing in."""
    return FileResponse(HERE / "static" / "index.html")


app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
