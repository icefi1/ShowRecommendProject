# CLAUDE.md

Project context for Claude Code. Read this before doing anything.

---

## 1. What this project is

An **explainable television show recommender** built around an interpretable feature space.

Every show is scored 0–1 across roughly 40 named axes (genre, mood, experiential features). Recommendation is a distance query in that space. Because the axes have names, every recommendation carries a plain-language explanation, and the user can steer results by adjusting individual axes.

The target user request — the one conventional recommenders cannot express:

> "Another horror series with lots of jumpscares, but without the convoluted plot."

**The core claim:** interpretable features enable steering and explanation. The research question is whether they can do that while remaining competitive on retrieval accuracy against black-box sentence embeddings.

## 2. Academic context

This is a **final year undergraduate dissertation** at the University of Leicester (project 25751, *AI: Netflix Recommender System*, proposer W.O.C. Ward). It is assessed work.

Consequences that affect how you should help:

- **I must understand every line I submit.** Explain what code does and why. Do not produce large opaque blocks. If there is a simpler approach that I will actually understand, prefer it over a clever one.
- **Prefer standard, citable techniques** over novel tricks. I need to justify choices in a report with references.
- **Evaluation matters as much as implementation.** A feature without measurement is worth less than a measured one.
- Learning outcomes centre on machine learning, feature extraction and classification, dimensionality reduction, NLP, and ML deployment. Work that demonstrates these is worth more than work that doesn't.

## 3. My environment

- **Windows**, PowerShell terminal, VS Code
- Project root: `D:\University\ShowRecommender\ShowRecommendProject`
- **Python 3.11**, virtual environment at `venv/` (activate with `venv\Scripts\activate`)
- Git repo: `https://github.com/icefi1/ShowRecommendProject` (public)
- `C:` drive has limited space — avoid installing large packages there where avoidable
- PowerShell `echo` writes UTF-16 and breaks git config files; always create text files via VS Code

**Secrets:** TMDB v4 read token lives in `.env` as `TMDB_TOKEN`, loaded with `python-dotenv`. `.gitignore` covers `.env`, `venv/`, `__pycache__/`, `*.pyc`, `shows_raw.json`. Never write a key into source.

## 4. Data

**Catalogue: TMDB.** Netflix retired its public API in 2014. TMDB `/discover/tv` with `with_watch_providers=8` and `watch_region=GB` gives Netflix availability. Use `append_to_response=reviews,keywords,content_ratings` to bundle requests. Developer key, non-commercial academic use. The app must display: *"This product uses the TMDB API but is not endorsed or certified by TMDB."*

**Review text — the critical dependency.** Scores must reflect the actual viewing experience, not the synopsis. A synopsis is marketing copy written before anyone watched; it can say a show is set in a haunted house but cannot say whether it made people jump. Reviews are the only text where viewers report what happened to them.

- TMDB reviews are available via API but **sparse** — viability is currently being tested
- **Do not scrape IMDb** — their terms prohibit it, and this is submitted work with an ethics section
- Fallback: **Penha & Hauff (2020)**, a research dataset pairing IMDb reviews with MovieLens titles
- Also available: Amazon Reviews (McAuley, UCSD), Movies & TV category

**Existing files:** `fetch_shows.py` (TMDB ingestion), outputs `shows_raw.json` and `shows.csv`.

## 5. Prior art I must cite

**The MovieLens Tag Genome** (Vig, Sen & Riedl, 2012) is the closest existing work — continuous 0–1 tag relevance scores over ~9,700 movies and ~1,080 tags, computed by ML over user tags, ratings and reviews.

My differentiators:

- **TV shows**, not movies
- **Experiential density** (how many jumpscares) rather than topical relevance (is this "horror")
- **Cold start** — tag genome relevance degrades for obscure titles because crowd tags need a large user base per title. A model reading review text scores a show released last week.

Where titles overlap, tag genome scores serve as external validation for mine.

## 6. Architecture

```
REVIEWS  →  MODEL  →  PEOPLE  →  FEATURE SPACE  →  INTERFACE
```

### 6.1 Feature schema

~40 axes, **fixed and defined in advance**, each with a written definition. Grouped into blocks:

| Block | Examples |
|---|---|
| Genre | horror, comedy, crime, documentary |
| Mood | creepy, bleak, warm, campy, tense |
| Features | jumpscares, plot twists, gore, slow burn, episodic vs serialised |

**Scores, not booleans** — a binary tag can't express "lots of jumpscares, few twists". **Fixed schema, not free generation** — letting a model invent tags produces near-duplicate axes ("creepy" and "spooky") which pushes identical shows apart and destroys distance.

### 6.2 Scoring — two stages

1. **Label generation.** An LLM scores a subset of shows against the fixed schema from review text, producing a labelled training set.
2. **Model training.** A multi-label regression model (fine-tuned sentence transformer or DistilBERT head) trained on those labels to predict scores from text.

The LLM is a labelling tool, like paid annotators. The trained model is my contribution. It also gives cost, speed and reproducibility that API calls can't — LLM outputs drift between provider updates; frozen weights don't.

Note terminology: outputs are continuous, so this is **multi-label regression**, not classification.

**Selection bias to design around:** reviewers are genre fans, and "wasn't even scary" is their most common complaint. Asking a model "how scary is this?" over such reviews systematically deflates horror. Prefer **evidence counting** — what proportion of reviewers describe being startled — over vibe estimation. It yields a defensible quantity with a confidence interval.

### 6.3 Crowd correction layer

Users can disagree with any score.

**Critical design rule: descriptive votes must be separate from evaluative ones.** The question is "does *funny* describe this show?" — never "did you like it?". Blend them and every popular show drifts high on every flattering axis, turning descriptions into approval ratings.

**Fusion:** the model's score is a **Bayesian prior**; votes are evidence updating it. Zero votes → the model's prediction. Few votes → small shift. Sustained agreement → large shift. This avoids the naive-ratio trap where 3 up / 0 down reads as 1.0, and means the system works on day one with no users.

Disagreements between crowd and model are **labelled error data** — report which axes the model gets wrong.

### 6.4 Similarity

**Blocked weighted distance**, not flat cosine. The vector is partitioned into blocks; distance is computed per block; blocks are weighted at query time.

Flat cosine produces mush — two shows can be equidistant for unrelated reasons and the user can't tell which. Blocking enables *same vibe, different subject* (mood high, genre low) and its inverse.

**Neighbours are always computed in the full-dimensional space.**

### 6.5 Learned weights (metric learning)

Rather than hand-setting block weights, learn them from pairwise human preference: "which of B or C is more similar to A?" Fit ~40 parameters with a RankNet-style pairwise objective.

Frame this as **metric learning**, citing the learning-to-rank literature as methodological ancestor. Full neural LTR is explicitly **rejected** — it needs click logs I don't have, and a black-box ranker would destroy both explanation and steering, which are the point of the project.

Learned weights become defaults; users still adjust from there. Learned vs hand-set weights is an ablation.

### 6.6 Explanation

Compare query vector to result vector, surface largest agreements and divergences in natural language: *"same dread, 60% less gore, shorter episodes."* Falls out of the representation for free. This is the central contribution.

## 7. Interface — two tiers

**Tier 1, standard GUI (the front door).** Search a show, adjust a few feature dials, get a ranked list where each result carries its one-line explanation. Usable in ten seconds, no spatial metaphor.

**Tier 2, 3D explorer (opt-in).** UMAP projects the feature space to 3D; the catalogue renders as a navigable point cloud.

- Projection is **display only**; true neighbours drawn as **explicit edges**, never inferred from visual proximity
- Also a **development tool** — if clusters look wrong to someone who's watched the shows, the tags are wrong

**Navigation follows Blender's viewport conventions** (transfer of learned skill; users of Blender/Maya/Unity arrive with muscle memory):

| Input | Action |
|---|---|
| MMB drag | Orbit (turntable) |
| Shift + MMB drag | Pan |
| Scroll | Dolly / zoom |
| Left click | Select |
| `.` | Frame selected |
| `Home` | Frame all |
| `1`/`3`/`7` | Front/side/top |
| `5` | Perspective ↔ orthographic |

Non-negotiable decisions:

- **Turntable, not trackball** — fixed up-axis; a stable horizon is the only orientation anchor in a featureless cloud
- **Orbit around selection, default** — orbiting world-origin after flying into a cluster throws the user across the map
- **Zoom toward cursor**
- **Middle-mouse emulation from day one** (`Alt`+drag orbit, `Alt`+`Shift`+drag pan, trackpad gestures, visible bindings helper) — trackpad users have no MMB; this is an accessibility requirement, not polish
- **Input layer as a remappable lookup table**, not hardcoded key codes

## 8. Stack

| Layer | Technology |
|---|---|
| Ingestion, tagging, training | Python |
| Projection | Python (`umap-learn`) |
| Similarity engine | Python (numpy) — one matrix op over ~5k shows |
| API | FastAPI (demonstrates ML deployment, a learning outcome) |
| Frontend | TypeScript + three.js |

Rendering: single `Points` geometry or instanced meshes, **never one mesh per show**. Positions pre-computed server-side. Default `OrbitControls` will not give Blender behaviour — use custom controls or `camera-controls`.

## 9. Evaluation (designed up front, not retrofitted)

1. **Baseline** — sentence-transformer embeddings + plain cosine. All accuracy claims are relative to this. It may win on accuracy; that's an acceptable finding, because it can't steer or explain.
2. **Retrieval accuracy** — precision@k against TMDB similar-titles as proxy ground truth, plus a human-judged set.
3. **Label validity** — 100 hand-tagged shows, Cohen's kappa against model output.
4. **Feature space structure** — PCA over the tag matrix. If 40 axes collapse to 12 components, several are redundant and the schema gets pruned. Directly demonstrates the dimensionality-reduction outcome.
5. **Ablations** — blocked weighted vs flat cosine; trained model vs the LLM labels it learned from; learned vs hand-set weights.
6. **Usability** — 5 participants, two tasks (find a show similar to X via standard GUI, then via 3D explorer), time-to-completion plus SUS. Participants also generate real tag/vote data.

## 10. Build order

1. TMDB ingestion, ~500 titles ✔ *(script written)*
2. Verify review availability — **current step, highest-risk dependency**
3. Schema definition — 40 axes with written definitions
4. **Hand-score 20 shows against the schema.** This is a gate. If I can't score consistently by hand, the definitions are ambiguous and no model will fix that. Expect to rewrite axes.
5. Baseline implementation
6. LLM labelling pipeline
7. Model training, scale to ~5,000 shows
8. Similarity engine + explanation generation
9. Evaluation (§9)
10. Standard GUI
11. 3D explorer
12. Blender navigation

**Sequencing rationale:** 3D work comes after evaluation deliberately. An incomplete evaluation chapter undermines the project's core claim; incomplete navigation degrades gracefully into "orbit and pan implemented, rest documented as further work."

## 11. Rules

- **Never** hardcode API keys; always `.env`
- **Never** suggest scraping IMDb or Netflix
- Explain code; I have to defend it
- Prefer simple and citable over clever
- Don't build ahead of the build order — flag it if I ask for something out of sequence
- When a design choice has a trade-off, say so; I need trade-offs for the report
- Flag anything that would make the system less explainable — explainability is the contribution, not a feature
