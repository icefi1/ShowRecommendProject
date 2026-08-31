# Handoff — current state

Read `CLAUDE.md` first for project rules. This file is what a new session needs
on top of it: what exists, what was decided and why, and what is mid-flight.

Last commit: `d42005e` — "Add batch 3 labels and make the sampler choose its own targets"
Branch: `feature/model-crowd-accounts`. PR #3 open against `main`. PRs #1, #2 merged.

---

## 1. Run it

```
venv\Scripts\python -m uvicorn app.main:app --reload --port 8000
```

Guests get the landing page; signed-in users get the app. **No account exists —
sign up through the UI.** `accounts.db` and `crowd.db` are gitignored and empty.

Regenerate pipeline (all outputs gitignored):

```
python tmdb/fetch_shows.py          # ~3,542 shows, ~2.6 min
python tmdb/fetch_episodes.py       # ~93,447 episodes, ~6 min
python tmdb/episode_features.py     # per-show pacing features
python app/build_space.py           # feature_space.npz + .json
python training/train_model.py      # model.joblib, predictions.csv, report.md
python evaluation/explanation_audit.py   # explanation quality, fixed seed
```

---

## 2. Scale and performance

| | Value |
|---|---|
| Shows | 3,542 (full Netflix GB listing on TMDB) |
| Episodes | 93,447 |
| Episode-overview corpus | 2.82M words |
| Feature space | 1,446 dims — genre 15, keywords 1,418, structure 13 |
| Query latency | 1.74 ms median, 2.11 ms p95 |
| Labelled shows | 78 |
| Model | 37/37 axes beat mean baseline, mean MAE 0.174 |

TMDB rate limit is **50 req/s, 20 connections**. Fetchers run 12 workers at
30 req/s via `tmdb/rate_limit.py` (spacing, not a token bucket — a bucket bursts
after idle, which looks like an attack).

---

## 3. Architecture as built

```
tmdb/          ingestion + episode-derived pacing features
labelling/     schema (source of truth), batch export, labels.jsonl
training/      multi-label ridge over frozen sentence-transformer embeddings
app/           FastAPI + similarity engine + accounts + static frontend
evaluation/    rerunnable audits of what the engine outputs
docs/          feature_schema.md is the running research log
```

- `labelling/schema.py` is the **single definition** of the 37 axes. Code imports it.
- `docs/feature_schema.md` is the prose companion + every finding, versioned v0.1→v0.9.
- Similarity is **blocked weighted cosine**, weights set per query. Neighbours
  always computed in full dimensional space.
- Results are **paged**, 24 at a time: `_rank()` takes an `offset`, the API
  returns `total`, and the interface appends the next page as you reach the
  bottom (`IntersectionObserver`) or press the button. Ranking is always over
  the whole catalogue - paging only changes which slice comes back, so the
  ordering is identical whether you ask for 12 results or 3,541.
- Predicted axes are **displayed but do NOT drive ranking** — at 78 labels the
  magnitudes are compressed by ridge shrinkage, so ordering is trustworthy and
  absolute values are not. Same for crowd-corrected scores.

---

## 4. Findings that must not be lost

**The TMDB taxonomy gap is the project's central argument.** TMDB TV has 15
genres and no Horror, Thriller, Romance, History or Fantasy. 85+ shows carry a
`romance` keyword and none can be labelled Romance. Stranger Things is labelled
*Action & Adventure, Mystery, Sci-Fi & Fantasy*. A recommender restricted to
TMDB genres cannot accept "a romance" at all. This is a stronger justification
than competitive retrieval accuracy.

**TMDB reviews are unusable as a corpus.** 1,034 reviews across 3,542 shows,
19% coverage, median 0. This is why user-written reviews (`accounts.py` →
`review_text_for()`) matter: they feed back into labelling, so the corpus grows
with use.

**Two negative results, both deliberately kept:**

1. `guest_star_mean` failed its hypothesis. Predicted to separate procedurals
   (fresh cast weekly) from serialised drama (standing cast); measured, it does
   the opposite — 13.2 vs 19.2 — because TMDB's `guest_stars` is the per-episode
   supporting cast credit, so it tracks ensemble *size*. Axis relabelled to what
   it measures. Real churn needs guest star identities, which `fetch_episodes.py`
   discards.
2. Batch 1 stratified by primary TMDB genre and contained **no horror show**,
   because TMDB has no Horror genre to select. The model scored The Haunting of
   Hill House at 0.10 on `horror`. Fixed by `export_coverage.py`, which samples
   on keyword evidence and now picks its own targets from measured label
   variance.

**Facts are not opinions.** `labelling/schema.py` splits axes: if a catalogue
source asserts it, it is a FACT — not votable, written from TMDB. Binary facts
(`animation`, `documentary`, `reality`) are 0/1; degree facts (`comedy`, `drama`,
`crime`, `mystery`, `action`) take TMDB's side of 0.5 with the model supplying
degree. The 29 judgement axes are votable. A schema assertion fails the build if
`horror`/`thriller`/`romance`/`historical` ever move into the fact set.

**Crowd fusion** is Bayesian, κ=8: `(κ·prior + Σ vote_targets) / (κ + n)`. Zero
votes returns the model's prediction, so it works on day one. Every vote stores
`score_shown` so neutral votes anchor the current value. No whole-show rating
exists anywhere — descriptive votes only, or descriptions become approval ratings.

---

## 5. Model progression

| Labels | Axes beating baseline | Notable |
|---|---|---|
| 24 | 31/37 | horror R² −0.081 (blind) |
| 50 | 36/37 | horror +0.343 |
| 78 | **37/37** | jumpscares 0.245 → **0.556**, horror **0.570**, creepy **0.513** |

Held-out sanity check: Better Call Saul scores `jumpscares` 0.00, `horror` 0.00,
`dialogue_driven` 0.74 — never seen in training.

**Still weakest:** `sentimental` 0.053, `absurd` 0.084, `earnest` 0.091,
`slow_burn` 0.107. Diffuse tonal qualities with no keyword vocabulary to sample
against. Open question whether keyword-based sampling can reach them at all.
Ignore `documentary` 0.012 — it's a fact axis, its regression score is meaningless.

---

## 6. DONE — explanation ordering

Explanations now read **genre → keywords → structure**, which was the request.
Written up in full as `docs/feature_schema.md` v0.10; the short version:

    both crime dramas; shares drug cartels, outlaw; but has shorter episodes

- **Genre clause added** to `app/similarity.py` → `_genre_clause()`. There was
  none before, which is why genre-driven matches were being explained by
  structural differences. Commonest shared genre supplies the noun, rarest
  supplies the modifier ("crime dramas", not "drama crimes").
- **Provenance keywords suppressed** in the explanation only — the 24 `based
  on ...`/`remake` terms. They still carry IDF weight and still affect ranking;
  removing them from the vocabulary is left as an ablation.
- **Structure capped at one trailing difference** (was two), and each phrase now
  carries its own verb, fixing "but is shorter episodes".
- **No shared genre and no shared keyword** returns "no shared genre or subject
  - matched on form alone" rather than a structure-only sentence.

Measured with `evaluation/explanation_audit.py` (fixed seed, 120 shows × 5
results = 600 explanations, rerunnable):

| | Before | After |
|---|---|---|
| Names a shared genre | 0.0% | **95.3%** |
| **No shared clause at all** | **25.0%** | **4.3%** |
| Cites a provenance keyword | 9.7% | **0.0%** |

The residual 4.3% is real missing data, not a phrasing bug: **87 shows (2.5%)
have neither a TMDB genre nor a surviving keyword**, and 19.3% have no keyword.

## 7. Other open items

- **Preference mode has no explanations.** Now the only open half of S6.6:
  `explain()` needs two catalogue rows to diff, and a dial query has no query
  show. The same three-clause shape would work diffed against the dial settings
  themselves (chosen genres, target structure values).
- **The green number on result cards** is raw cosine × 100 and moves with the
  weight sliders. Honest but confusing; worth relabelling.
- **Wire predicted axes into ranking** — blocked on label count. Probably needs
  ~150 labels before magnitudes decompress enough.
- **Accounts have no email verification or rate limiting.** Fine for a prototype,
  needed before vote counts mean anything live.
- **`secure=False` on the session cookie** is correct for localhost only. Must be
  `True` behind HTTPS. Flagged in code.
- **6% of catalogue is unrated** and sits mid-scale on `maturity`. `NR` is
  deliberately unmapped — "not rated" is missing data, not a rating.
- **Landing page images** are AI-generated (Gemini), originals in `homepage images/`,
  WebP in `app/static/img/`. **Declare this in the report** if they appear in the
  submission.

---

## 8. Working preferences established

- The one-line uni brief is a floor, not a target. Never advise scoping down to
  meet it — marks come from ambition. (Also saved to memory.)
- No budget for API keys. Labels are produced by the assistant in-session at no
  marginal cost, written to `labelling/labels.jsonl` in the same format the API
  path would produce.
- Negative results are kept and written up, not hidden.
- Explain code; it has to be defended in a viva. Hand-written over library calls
  where the hand-written version is short and clearer (`least_squares_slope`,
  `entity_density`).
