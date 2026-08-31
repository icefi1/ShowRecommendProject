# Show Recommender

An explainable television recommender. Every show is scored across 37 named
axes, so a recommendation can be steered — more of this, less of that — and
every result explains itself in plain language.

The request it exists to answer, which conventional recommenders cannot
express:

> "Another horror series with lots of jumpscares, but without the convoluted
> plot."

Final-year dissertation project, University of Leicester (project 25751,
*AI: Netflix Recommender System*).

---

## Why not just use genres

TMDB's television taxonomy has **15 genres, and none of them is Horror.**
Neither is Romance, Thriller, or History. Those exist in TMDB's *film* list;
the TV list omits them.

Over this catalogue that means:

| | |
|---|---|
| Shows carrying a `romance` keyword | **85** |
| Shows that can be labelled Romance | **0** |
| Stranger Things is labelled | Action & Adventure, Mystery, Sci-Fi & Fantasy |
| Black Mirror is labelled | Sci-Fi & Fantasy, Drama, Mystery |

The shows are all there. The vocabulary to describe them is not. A recommender
restricted to catalogue genres cannot accept the request "a romance" at all —
which is the gap this project fills, and the reason interpretable axes are the
point rather than a nicety.

---

## What it does

**Search a show** → get neighbours, each with a generated reason:

```
0.740  The Haunting of Bly Manor   shares haunted house, supernatural horror, ghost;
                                   but is a self-contained miniseries
0.586  The Midnight Club           shares haunted house, horror, based on novel
0.512  Archive 81                  shares supernatural horror, horror; but is shorter
```

**Or describe what you want** — genres and dials, no title needed.

**Steer it.** Block weights are adjustable live; each result shows its per-block
breakdown (`g0.67 · k0.15 · s0.97`), so a structural match is visibly distinct
from a subject-matter one.

**Correct it.** Signed-in users vote per axis — higher, lower, or *this is
right* — and votes are fused with the model's prediction as Bayesian evidence.

---

## Numbers

| | |
|---|---|
| Shows | 3,542 |
| Episodes analysed | 93,447 |
| Episode-overview corpus | 2.82M words |
| Feature dimensions | 1,446 (15 genre + 1,418 keyword + 13 structure) |
| Named axes | 37 — 29 votable, 8 catalogue-sourced |
| Hand-labelled shows | 50 |
| Query latency | 1.74 ms median, 2.11 ms p95 |

---

## How it works

```
TMDB ──► feature space ──► similarity ──► interface
             ▲                              │
             │                              ▼
      trained model ◄── labels ◄──── user reviews & votes
```

### The feature space

Three blocks, built by `app/build_space.py`:

- **genre** (15) — binary, from TMDB
- **keywords** (1,418) — TF-IDF, restricted to keywords on ≥3 shows
- **structure** (13) — measured pacing and form

Keywords appearing on a single show are dropped. 65% of the raw vocabulary is
singletons, and a singleton cannot create similarity with anything — it only
adds a dimension of noise.

Structure axes are **percentile-ranked, not min-max scaled**. This discards
absolute scale (0.9 means "more than 90% of the catalogue", not any particular
count) but is immune to outliers: Sesame Street's 3,551 episodes would
otherwise compress the entire catalogue into the bottom 2% of that axis.

### Similarity

**Blocked weighted distance, not flat cosine.** Cosine is computed per block and
the blocks are combined with weights supplied at query time. A single flat
cosine over all 1,446 dimensions would let a strong keyword match and a strong
structural match produce identical scores with no way to tell which happened.

Neighbours are always computed in the full-dimensional space.

Certificates are applied as a **multiplier on the finished score** rather than
as one of thirteen structure dimensions, where they carried about 1.5% of the
weight and could not stop a U-rated cartoon ranking against an 18.

### Explanations

Generated from the same vectors that produce the ranking: shared high-IDF
keywords, plus the largest gaps on named structure axes. There is deliberately
**no generative model in this path** — an explanation that can disagree with the
ranking would defeat the purpose.

### The trained model

An LLM scores shows against the schema; a multi-label ridge regression over
frozen sentence-transformer embeddings learns to predict those scores from
text. The LLM is annotation scaffolding — the trained model is the artefact,
with frozen weights, no API cost, and no drift between provider updates.

Current: **36/37 axes beat the mean baseline**, mean MAE 0.179, from 50 labels.

> **Predicted axes are displayed but do not drive ranking.** At 50 training
> labels the magnitudes are compressed by ridge shrinkage, so the *ordering* is
> trustworthy and the absolute values are not. Wiring them into the distance
> metric now would make recommendations worse.

### Facts vs judgements

Not every axis is an opinion. A show is animated or it is not.

| Kind | Axes | Votable |
|---|---|---|
| Fact, binary | `animation`, `documentary`, `reality` | no — written 0/1 from TMDB |
| Fact, degree | `comedy`, `drama`, `crime`, `mystery`, `action` | no — TMDB sets which side of 0.5 |
| Judgement | the other 29, incl. `horror`, `thriller`, `romance`, `historical` | **yes** |

The split lands where the argument does: TMDB has no Horror, Thriller, Romance
or History genre, so those four have no authority to defer to and are exactly
the axes worth voting on.

### Crowd correction

Votes are fused as Bayesian evidence, not counted as a ratio:

```
posterior = (κ · prior + Σ vote_targets) / (κ + n_votes)     κ = 8
```

From a prior of 0.30: 0 votes → 0.300; 3 up-votes → 0.345; 23 up-votes → 0.496.
A naive `up/(up+down)` would read three up-votes as a confident **1.00**, and
would do nothing at all before the system has users. This returns the model's
own prediction at zero votes, so it works on day one.

Every vote records `score_shown` — the value on screen when it was cast — which
makes the system self-correcting rather than runaway: once a score is right,
voters press *neutral*, which targets the current value and anchors it.

**There is no whole-show rating anywhere in the schema.** Blending descriptive
and evaluative votes turns descriptions into approval ratings, with every
popular show drifting high on every flattering axis.

---

## Running it

Requires Python 3.11 and a TMDB v4 read token.

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env` in the project root:

```
TMDB_TOKEN=your_v4_read_access_token
```

Fetch the data and build the space (about 8 minutes total):

```bash
python tmdb/fetch_shows.py
```
```bash
python tmdb/fetch_episodes.py
```
```bash
python tmdb/episode_features.py
```
```bash
python app/build_space.py
```

Optionally train the scoring model:

```bash
python training/train_model.py
```

Run it:

```bash
venv\Scripts\python -m uvicorn app.main:app --reload --port 8000
```

Then open <http://127.0.0.1:8000>.

### Rate limiting

TMDB permits 50 requests/second and 20 connections per IP. The fetchers run 12
workers against a shared limiter at **30 req/s** — 60% of the published limit.
`tmdb/rate_limit.py` spaces requests rather than using a token bucket, because a
bucket permits a burst up to its capacity after an idle moment, which is exactly
the shape CDN abuse heuristics look for.

---

## Layout

```
tmdb/           ingestion — fetch, episode features, rate limiter
labelling/      the 37-axis schema, labelling pipelines, labels.jsonl
training/       multi-label regression over frozen embeddings
app/            feature space, similarity, crowd fusion, accounts, FastAPI, UI
docs/           feature_schema.md — axis definitions and the running log
```

`docs/feature_schema.md` is the substantive companion to this file: every axis
with a written definition and scoring anchors, plus a versioned record of what
was measured and what failed.

Generated data (`shows_raw.json`, `episodes.json`, `feature_space.*`, the
SQLite databases) is gitignored and rebuilt by the commands above.

---

## Notes for a reader

**Two findings are recorded rather than hidden**, because both were informative:

*The horror blind spot.* The first labelling batch stratified by primary TMDB
genre and produced a training set containing no horror show at all — TMDB has
no Horror genre for the sampler to select. The model scored The Haunting of
Hill House at 0.10 on `horror`. Sampling on axis coverage instead took
`horror` R² from −0.081 to +0.343. A missing category in a source taxonomy
propagates into a blind spot in a learned model, and per-axis evaluation
catches it where an aggregate hides it.

*A failed hypothesis.* `guest_star_mean` was added expecting to separate
procedurals (fresh cast weekly) from serialised drama (standing cast). Measured,
it does the opposite — 13.2 vs 19.2 — because TMDB's `guest_stars` is the
per-episode supporting cast credit, so it tracks ensemble *size*. The axis was
relabelled to what it measures.

**Known limitations.** Review coverage is 19% of the catalogue. `jumpscares`
and `gore` remain weakly grounded. 6% of shows carry no certificate and sit
mid-scale. Preference-mode results carry no explanation, because the explainer
diffs against a query show and there isn't one. Accounts have no email
verification or rate limiting — fine for a prototype, not for a deployment.

**Security.** Passwords are stored as scrypt hashes (RFC 7914) with per-user
salts and stored cost parameters. Sessions are httpOnly, SameSite=Lax cookies.
`secure=False` on the cookie is correct only for plain-http localhost and
**must be `True` behind HTTPS**.

---

## Attribution

This product uses the TMDB API but is not endorsed or certified by TMDB.

Landing-page imagery was generated with Google Gemini; see
`app/static/img/ATTRIBUTION.md`. Labels in `labelling/labels.jsonl` were
generated by Claude Opus 5 and are marked with their annotator.
