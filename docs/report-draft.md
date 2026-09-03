# An explainable television recommender built on interpretable features

**Draft.** Numbers in here are real and come from the scripts in `evaluation/`.
Anything not measured yet is marked `[TODO]` so I don't accidentally write a
claim I can't back up.

---

## Abstract

[TODO: write last, once the evaluation chapter is finished. Should be about 200
words: the problem, what I built, the headline numbers, and the one honest
caveat.]

---

## 1. Introduction

Netflix can tell you that people who watched *Breaking Bad* also watched *Ozark*.
It can't tell you why, and it can't take a request like:

> "Another horror series with lots of jumpscares, but without the convoluted plot."

That sentence has three parts. A genre, an experiential feature I want more of,
and one I want less of. Every recommender I can actually use will take the first
part and ignore the other two, because underneath it is a black box — a matrix
factorisation or a neural embedding where the dimensions mean nothing to anyone,
including the people who built it.

This project is an attempt at the opposite. Every show gets scored 0–1 on 37
named axes: genre things like `horror` and `crime`, mood things like `bleak` and
`campy`, and experiential things like `jumpscares` and `plot_complexity`.
Recommendation is then just a distance query in that space. Because the axes
have names, two things fall out for free:

1. **Steering.** More jumpscares, less plot complexity, is a query you can
   actually type, because those are dimensions the system has.
2. **Explanation.** Comparing the query vector to the result vector tells you
   what they share and how they differ, in words: *both crime dramas; shares
   mexican cartel, drug lord; but has longer episodes.*

The research question is whether that costs anything. Interpretable features are
obviously more useful — the question is whether they're less accurate than the
black box they replace. My answer, measured two different ways, is no.

### What I'm claiming

- A blocked, weighted feature space over interpretable axes **beats** a
  sentence-transformer baseline on retrieval accuracy, on both an automatic
  answer key (+0.047 precision@5, 95% CI [0.038, 0.056]) and a human-judged one
  (+0.204, CI [0.031, 0.385]).
- Every recommendation carries a one-line explanation that comes out of the same
  representation used to rank, so it cannot disagree with the ranking. 95.3% of
  explanations name a shared genre and 70.7% name a shared keyword.
- The design decisions are measured rather than asserted, including two that
  measured badly and one that failed its hypothesis outright. Those are in here
  too, because a design chapter with no failures in it is not a real one.

---

## 2. Background

### 2.1 The closest existing work

The **MovieLens Tag Genome** (Vig, Sen and Riedl, 2012) is the thing this is most
like. It scores ~9,700 films against ~1,080 tags on a continuous 0–1 relevance
scale, computed by a model over user tags, ratings and reviews. It's the proof
that a continuous, named feature space over a catalogue is a workable idea.

Three things make mine different:

- **Television, not film.** Series have structure films don't — episode counts,
  season counts, pacing across a run, whether it's a miniseries. Several of my
  axes only exist because the unit is a series.
- **Experiential density, not topical relevance.** The tag genome answers "is
  this horror?". I want "how many jumpscares?". That's a different question and
  it's the one the motivating query needs.
- **Cold start.** Tag genome relevance degrades for obscure titles, because it
  needs a crowd per title. A model reading text can score a show released last
  week with no users at all.

[TODO: where titles overlap between MovieLens and my catalogue, correlate my
scores against tag genome relevance. That's free external validation and I
should do it.]

### 2.2 Why I couldn't use the reviews

The original plan was to score shows from review text, on the argument that a
synopsis is marketing written before anyone watched — it can tell you a show is
set in a haunted house but not whether it made anyone jump. Reviews are the only
text where viewers say what happened to them.

TMDB has a reviews endpoint. I measured what it actually returns across my
catalogue: **1,034 reviews across 3,542 shows, 19% coverage, median zero.** Most
shows have no reviews at all. As a corpus it's unusable, and I'm not scraping
IMDb — their terms forbid it and this is submitted work with an ethics section.

So the model reads what I do have: TMDB synopses plus the episode-overview
corpus, 2.82 million words across 93,447 episodes. That's a real limitation and
I'd rather state it plainly than bury it: my scores describe what shows are
*about* more reliably than what watching them *feels like*.

[TODO: mention Penha and Hauff (2020) and the Amazon Reviews dataset as the
fallbacks I considered, and say why I didn't use them — mainly title alignment
cost against a TV catalogue.]

### 2.3 The taxonomy gap, which is the real argument

This is the finding I'd lead with if I could only keep one.

**TMDB's television genre list has 15 entries and none of them is Horror.**
Neither is Thriller, Romance, History, or Fantasy. Those exist in TMDB's *film*
list; the TV list omits them.

Over my catalogue that means:

| | |
|---|---|
| Shows carrying a `romance` keyword | 85 |
| Shows that can be labelled Romance | **0** |
| *Stranger Things* is labelled | Action & Adventure, Mystery, Sci-Fi & Fantasy |

The shows exist. The vocabulary to ask for them doesn't. A recommender
restricted to catalogue genres cannot accept "a romance" as a request at all.
That's a stronger justification for building an interpretable feature space than
any accuracy number I could produce, because it isn't a question of doing the
same job better — it's a job the catalogue's own taxonomy cannot do.

---

## 3. Design

### 3.1 The feature space

37 axes, fixed and defined in writing before any scoring happened. Fixing the
schema up front matters: if you let a model invent tags it produces near
duplicates — `creepy` and `spooky` — and near duplicates push identical shows
apart, which destroys the distance metric that the whole system is.

Scores, not booleans, because a binary tag can't say "lots of jumpscares, few
twists", and that distinction is the entire point.

### 3.2 Facts are not opinions

The axes are split in two, and this is enforced by a build-time assertion.

If a catalogue source asserts it, it's a **fact**: `animation`, `documentary`,
`reality` are 0/1 straight from TMDB. Degree facts like `comedy` and `crime` take
TMDB's side at 0.5 and let the model supply the degree. Facts are not open to
voting — there's no point crowdsourcing whether a documentary is a documentary.

The other 29 are **judgements**, and those are votable. The assertion fails the
build if `horror`, `thriller`, `romance` or `historical` ever migrate into the
fact set, because TMDB cannot supply them (see 2.3) and pretending otherwise
would quietly reintroduce the gap the project exists to close.

### 3.3 Blocked similarity

The vector is partitioned into three blocks — genre (15 dims), keywords (1,418),
structure (13) — and distance is computed per block, then combined with weights
set at query time.

Flat cosine over one concatenated vector produces mush. Two shows come out
equidistant from the query for completely unrelated reasons and the user can't
tell which, and can't ask for more of one. Blocking is what makes "same vibe,
different subject" expressible: turn the mood weight up and the genre weight
down.

Neighbours are always computed in the full-dimensional space. Nothing is ever
ranked in a projection.

I measured whether blocking earns its place rather than assuming it. It does —
see 5.3.

### 3.4 The certificate rule

Age rating is one of 13 structure dimensions, and the structure block carries
0.20 of the weight, so maturity contributes about 1.5% of a score on its own —
nowhere near enough to stop a U-rated cartoon ranking against an 18-rated drama
when their keywords happen to line up. *Teen Titans Go!* was pulling 15-rated
shows into its results.

So it's applied separately as a multiplier: `score *= 1 - 0.5 * |maturity gap|`.
The widest possible gap halves a score; one step costs about 15%. It demotes
rather than excludes, which is the intent.

That 0.5 was reasoned into existence and never measured. When I finally measured
it, it lost — see 5.5. I kept it anyway, and I explain why there.

### 3.5 The crowd layer

Users can disagree with any judgement axis. Two rules make this work rather than
turn into a popularity contest:

**Descriptive, never evaluative.** The question is "does *funny* describe this
show?" — never "did you like it?". If you blend those, every popular show drifts
high on every flattering axis and your descriptions quietly become approval
ratings. There is no whole-show rating anywhere in the system, deliberately.

**The model is a prior, votes are evidence.** Fusion is Bayesian with κ=8:

```
score = (κ · model_prediction + Σ vote_targets) / (κ + n)
```

Zero votes returns the model's prediction, so the system works on day one with no
users. A handful of votes nudges it. Sustained agreement moves it a long way.
This avoids the naive-ratio trap where 3 up-votes and 0 down reads as 1.0.

Every vote stores the score that was on screen when it was cast, so a neutral
vote anchors the current value rather than drifting it.

Disagreement between crowd and model is labelled error data — it tells me which
axes the model is worst at, for free, from use.

### 3.6 Explanation

Compare the query vector to the result vector, take the largest agreements and
divergences, and write them as a sentence. It falls out of the representation, so
there's no separate explanation model that could disagree with the ranking — a
property most post-hoc explanation methods can't claim.

The ordering is genre, then keywords, then structure, and that ordering came from
a measurement rather than taste (5.6).

---

## 4. Implementation

| Layer | What I used |
|---|---|
| Ingestion, features, training | Python |
| Similarity engine | numpy — one matrix-vector product over the catalogue |
| API | FastAPI |
| Interface | HTML/CSS/JS, server-rendered data |

| | |
|---|---|
| Shows | 3,542 (the full Netflix GB listing on TMDB) |
| Episodes | 93,447 |
| Episode-overview corpus | 2.82M words |
| Feature space | 1,446 dimensions |
| Query latency | 1.74 ms median, 2.11 ms p95 |
| Labelled shows | 78 |

### 4.1 Things worth defending in the viva

**Precomputed row norms.** Computing vector norms per query dominated query
time. Precomputing them at startup took a query from ~54 ms to under 1 ms, which
is the difference between a visible stall and an instant response while dragging
a weight slider.

**argpartition, not sort.** The top k is found with `np.argpartition`, which is
O(n) rather than O(n log n), because I only need the top handful out of 3,542.
When the interface pages deeper, the partition point moves down the ranking
rather than the whole list being sorted.

**Rate limiting by spacing, not a token bucket.** TMDB allows 50 requests/second.
I run 12 workers at 30 req/s with enforced spacing between requests. A token
bucket would have been easier, but a bucket bursts after any idle period, and a
burst against a public API looks like an attack.

**Percentile normalisation on structure axes.** Structure values are mapped to
0–1 by rank, not magnitude. This throws away absolute scale — 0.9 means "longer
than 90% of the catalogue", not any particular number of minutes — but it's
immune to outliers. *Sesame Street* has 3,551 episodes and would otherwise
compress every other show into the bottom 2% of that axis.

---

## 5. Evaluation

Everything here is reproducible: fixed seeds, scripts in `evaluation/`.

### 5.1 What I compared against

Seven systems, all ranking the same catalogue:

| | |
|---|---|
| `blocked` | my engine as deployed |
| `blocked-cert` | same, certificate rule off |
| `flat` | same numbers, one vector, one cosine |
| `embeddings` | sentence-transformer over show text, plain cosine — **the baseline** |
| `embeddings+m` | the baseline with my certificate rule bolted on |
| `popular` | most popular shows, query ignored |
| `random` | chance |

The baseline reads exactly the same text my model does, so it isn't a straw man.

### 5.2 Automatic answer key

TMDB stores its own "recommendations" list per show, which my fetcher already
saved. Restricted to titles inside my catalogue that gives an answer key for 94%
of the catalogue: 1,561 query shows with at least 5 related titles, 395 with at
least 10. Intervals are 95% bootstrap over 2,000 resamples.

| System | precision@5 | precision@10 |
|---|---|---|
| `blocked` | 0.134 [0.126, 0.142] | 0.138 [0.125, 0.152] |
| `blocked-cert` | 0.137 [0.128, 0.145] | **0.149** [0.136, 0.163] |
| `flat` | 0.082 [0.075, 0.089] | 0.095 [0.085, 0.105] |
| `embeddings` | 0.087 [0.079, 0.094] | 0.089 [0.078, 0.100] |
| `embeddings+m` | 0.084 | 0.079 |
| `popular` | 0.003 | 0.001 |
| `random` | 0.003 | 0.004 |

**My space beats the baseline by +0.047 at k=5, paired CI [0.038, 0.056]** —
about 54% relative, nowhere near zero.

The controls are the check that the key itself is sane. If `popular` had scored
well it would mean the key mostly rewards recommending famous shows and every
other row would be suspect. It scores 0.003.

**What this number is not.** TMDB's list is another recommender's opinion, partly
built from user behaviour. Scoring well means agreeing with the kind of system I
argue is insufficient. It's a floor — evidence I'm not returning noise — and the
claim rests on 5.4 instead. There's also a shared-provenance problem: my keyword
block and TMDB's recommender both derive from TMDB metadata, so some of the lead
may be common ancestry rather than better modelling.

### 5.3 Does blocking earn its place?

`flat` is the same numbers concatenated into one vector with the certificate rule
held constant, so blocking is the only difference: **+0.052 at k=5, CI [0.044,
0.060]**.

The sharper observation is that `flat` (0.082) is no better than the sentence
embeddings (0.087). Concatenating the blocks destroys exactly what makes the
representation work. The "flat cosine produces mush" claim in 3.3 now has a
number attached instead of being an assertion.

### 5.4 Human judgement

The claim needs people, so I built a judging tool and used it.

**Method.** Pooling, from the Cranfield paradigm and standard TREC practice since
1992: for each query show take the top 5 from every system, merge them, judge the
merged set once, reuse each judgement across all systems. The pool is shuffled so
I can't tell which system produced a candidate, and fixed so a second judge sees
identical pairs.

**Two design corrections I had to make, which are findings in themselves.**

My first attempt sampled query shows from the 400 most popular titles, on the
assumption that popularity approximates familiarity. It came back **80–90% "don't
know it"** and was worthless — 14 of 331 pairs marked relevant. So I added a
screening step: I tick which shows I've actually seen, and the pool is built only
from those. The asymmetry is the point — only the *query* show needs to be known,
because the candidate can be judged from its poster and description. Screening
one side of the pair rather than both is what keeps this to an hour.

Second, I was originally counting "don't know it" as a miss, which punishes a
system for surfacing an obscure show exactly as hard as for surfacing a bad one.
Those are different failures. The standard treatment of incomplete judgements is
the condensed list (Sakai, 2007; the same reasoning behind bpref, Buckley and
Voorhees, 2004): drop unassessable results and score what's left.

**Results.** 752 verdicts over 20 query shows I've seen:

| System | counting unknowns as misses | assessable results only | assessable per 5 |
|---|---|---|---|
| `blocked` | 0.190 [0.110, 0.290] | **0.551** [0.383, 0.710] | 3.1 |
| `blocked-cert` | 0.190 | 0.529 [0.333, 0.711] | 2.8 |
| `flat` | 0.150 | 0.436 [0.275, 0.617] | 2.8 |
| `embeddings` | 0.150 | 0.347 [0.125, 0.583] | 2.1 |
| `embeddings+m` | 0.120 | 0.375 [0.139, 0.625] | 1.8 |

Counting unknowns, nothing separates. Counting only what a person could actually
assess, **blocked beats the baseline by 0.204, CI [0.031, 0.385], entirely above
zero** — and it agrees with the direction the automatic key already showed.

The coverage column is its own result: the embedding baseline returns markedly
more shows I couldn't assess (2.1 of 5 against 3.1). The strict measure was
penalising it for that and it still lost.

**Caveats, stated up front.** One judge, and that judge built the system. 12 of 20
queries were usable for the condensed measure. The intervals are wide.

[TODO: second judge on the same pool. Cohen's kappa between us is what turns this
from my opinion into a measurement, and it's the single highest-value thing left.]

By accident I got a version of this already: I judged 79 identical pairs twice,
about forty minutes apart, under two spellings of my name that the scorer treated
as two people. Test–retest agreement was **kappa 0.661** — substantial on the
Landis and Koch scale — and 11 of the 12 answers I changed were pairs I'd first
marked unknown. It's a real reliability measure and worth reporting.

### 5.5 A design decision that measured badly

`MATURITY_PENALTY = 0.5` (3.4) had never been tested. Switching it off improves
precision@10 by 0.011, CI [0.002, 0.021], entirely below zero. The same direction
shows on the baseline.

I kept it. TMDB's answer key has no opinion about mixing certificates, so this
measurement scores the rule against a criterion it wasn't built for: it exists to
stop a U-rated cartoon ranking beside an 18-rated drama, which is a user-facing
property the proxy can't see. What's changed is that the cost is now quantified —
about one percentage point of proxy precision — instead of assumed to be zero.

### 5.6 Explanations

600 explanations from a 120-show sample, before and after reordering:

| | Before | After |
|---|---|---|
| Names a shared genre | 0.0% | **95.3%** |
| Names a shared keyword | 75.0% | 70.7% |
| **No shared clause at all** | **25.0%** | **4.3%** |
| Cites a provenance keyword | 9.7% | **0.0%** |

A quarter of explanations used to open with "but is…" and list only structural
differences — telling you how two shows diverge before telling you they had
anything in common. The cause was that `explain()` read the keyword and structure
blocks and never touched the genre block at all, so a genre-driven match got
explained structurally.

Which genres to name is decided by catalogue frequency: the commonest shared
genre becomes the noun, the rarest the modifier. Drama is on 1,707 shows and
Crime on 621, so a pair sharing both reads *both crime dramas*, not *both drama
crimes*. That's the same rarity-is-informative principle behind IDF, reused on a
block that's binary and has no IDF of its own.

The residual 4.3% is missing data, not phrasing: 87 shows carry neither a TMDB
genre nor a surviving keyword.

### 5.7 The one that failed its hypothesis

`guest_star_mean` was built to separate procedurals (fresh cast every week) from
serialised drama (standing cast). Measured, it does the exact opposite — 13.2
against 19.2 — because TMDB's `guest_stars` field is the per-episode supporting
cast credit, so it tracks ensemble *size*, not turnover. I relabelled the axis to
what it actually measures and kept it. Measuring real churn needs guest star
identities across episodes, which my fetcher discards.

[TODO: 5.8 label validity — Cohen's kappa between my hand scores and model
output on 100 shows. Not done.]

[TODO: 5.9 PCA over the tag matrix. If 37 axes collapse to 12 components several
are redundant and the schema should be pruned. Not done, and it's the clearest
demonstration of the dimensionality-reduction learning outcome.]

[TODO: 5.10 usability study, 5 participants, SUS plus time-to-completion.]

---

## 6. What I'd do differently

**The review corpus was the plan and it didn't survive contact with the data.**
I should have measured TMDB review coverage in week one instead of assuming an
endpoint's existence meant it was populated. Everything downstream inherited that.

**I guessed at familiarity instead of asking.** The first two judging sessions
produced almost nothing because I assumed popular meant known. Two hours of my
own judging went into finding that out, when a screening step would have cost
twenty minutes to build.

**I set constants by reasoning and measured them far too late.** The certificate
penalty is the clearest case. It sounded obviously right, so it went unmeasured
for months, and when I finally tested it it was costing accuracy.

---

## 7. Further work

- **Second judge**, then Cohen's kappa. Highest value, lowest effort.
- **Wire the predicted axes into ranking.** They're displayed but don't drive
  results yet: at 78 labels ridge shrinkage compresses the magnitudes, so the
  ordering is trustworthy but the absolute values aren't. Probably needs ~150.
- **3D explorer.** UMAP to 3D, catalogue as a navigable point cloud with true
  neighbours drawn as explicit edges rather than inferred from visual proximity,
  Blender viewport conventions for navigation. Designed, not built.
- **Preference-mode explanations.** Dial-built queries return results with no
  explanation, because `explain()` needs two catalogue rows to diff.
- **Drop provenance keywords from the vocabulary**, not just from explanations.
  Currently `based on novel or book` still influences ranking; removing it would
  change results, so it's an ablation I haven't run.

---

## 8. Notes to self

Not part of the report.

- Declare the AI-generated landing page imagery (Gemini) if those screenshots
  appear in the submission. Attribution file is in `app/static/img/`.
- The TMDB attribution line must appear in the app: it does, in the footer.
- Ethics: no scraping, no personal data beyond locally-stored accounts,
  participants for the usability study need consent forms. [TODO: get the form.]
- Every number in section 5 regenerates from `evaluation/`. Re-run before
  submission in case the catalogue is refetched.
