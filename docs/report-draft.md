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

That sentence has three parts: a genre, a thing I want more of, and a thing I
want less of. Every recommender I can actually use handles the first part and
ignores the other two. The reason is that underneath they are all some kind of
black box, usually matrix factorisation or a neural embedding. The dimensions in
those don't mean anything, so there is nothing for the user to grab hold of.

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
- I measured my design decisions instead of just asserting them. Two of them
  came out badly and one failed completely. I have written those up as well,
  since leaving them out would make the rest of it less believable.

---

## 2. Background

### 2.1 The closest existing work

The **MovieLens Tag Genome** (Vig, Sen and Riedl, 2012) is the closest thing to
what I built. It scores about 9,700 films against about 1,080 tags on a
continuous 0–1 scale, worked out by a model over user tags, ratings and reviews.
It is the main evidence that this kind of feature space works at all, so most of
my design starts from it.

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

So the model reads what I actually have: TMDB synopses plus every episode
overview, 2.82 million words across 93,447 episodes. This is a genuine
limitation and it affects the whole project, so I would rather say it here than
hide it in the conclusion. My scores are better at describing what a show is
*about* than what watching it *feels like*, which is not what I set out to
build.

[TODO: mention Penha and Hauff (2020) and the Amazon Reviews dataset as the
fallbacks I considered, and say why I didn't use them — mainly title alignment
cost against a TV catalogue.]

### 2.3 The taxonomy gap

If I could only keep one finding from this project it would be this one.

**TMDB's television genre list has 15 entries and none of them is Horror.**
Neither is Thriller, Romance, History, or Fantasy. Those exist in TMDB's *film*
list; the TV list omits them.

Over my catalogue that means:

| | |
|---|---|
| Shows carrying a `romance` keyword | 85 |
| Shows that can be labelled Romance | **0** |
| *Stranger Things* is labelled | Action & Adventure, Mystery, Sci-Fi & Fantasy |

The shows are all there. The words to ask for them are not. A recommender that
only knows catalogue genres cannot accept "a romance" as a request, full stop.

I think this justifies the project better than any accuracy number does. If my
system were only a slightly better version of what already exists, it would be
hard to argue for. This is a request the catalogue's own taxonomy cannot
represent at all.

---

## 3. Design

### 3.1 The feature space

37 axes, written down and defined before any scoring happened. Fixing the schema
first matters more than it looks. If you let a model invent its own tags it
produces near-duplicates like `creepy` and `spooky`, and then two shows that are
basically identical end up far apart because one got tagged with each. Since the
entire system is a distance calculation, that breaks it.

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

I picked 0.5 because it seemed about right, and then never checked it. When I
did eventually measure it, it turned out to be making the results slightly
worse (5.5). I kept it, and I explain why there.

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

Compare the query vector to the result vector, take the biggest agreements and
differences, and turn them into a sentence.

The useful part is that this comes out of the same numbers used to do the
ranking. There is no second model producing explanations, so the explanation
cannot contradict the result. Post-hoc methods like LIME cannot promise that,
because they are approximating a model from the outside.

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

### 4.1 Getting the scores: annotation, then a model

Two stages, and the split matters.

**Stage one, labelling.** An LLM scores a subset of shows against the fixed
schema from their text. It is being used as an annotator — the same job you'd
pay people to do — not as the recommender. Batches are sampled deliberately
rather than at random, which I had to learn the hard way (5.7).

**Stage two, the model.** A multi-label ridge regression head on top of frozen
sentence-transformer embeddings (`all-MiniLM-L6-v2`, 384 dimensions), trained on
those labels to predict all 37 axes from text.

The trained model is the contribution, not the LLM. It gives three things API
calls can't: it costs nothing to run over the full catalogue, it scores a show
released after any model's cutoff, and it's frozen — LLM outputs drift between
provider updates, weights on disk don't. Reproducibility matters for a result
someone else is supposed to be able to check.

**Why not fine-tune DistilBERT end to end?** With a few hundred labelled shows,
fine-tuning ~66M parameters overfits almost immediately: it memorises the
training titles instead of learning what the words mean. A frozen encoder plus a
linear head fits about 384 × 37 parameters, which is the right capacity for this
much data, and it trains in seconds on a CPU with no GPU available. If the
labelled set ever reaches a few thousand shows, fine-tuning becomes worth
revisiting — that's a scale question, not a correctness one.

**Results at 78 labels**, five-fold cross-validated, R² above 0 meaning the model
beats predicting that axis's mean:

| | |
|---|---|
| Axes beating the mean baseline | **37 of 37** |
| Mean absolute error | 0.174 |
| Best axes | `horror` 0.570, `jumpscares` 0.556, `fantasy` 0.538, `crime` 0.519, `creepy` 0.513 |
| Worst | `sentimental` 0.053, `absurd` 0.084, `earnest` 0.091, `slow_burn` 0.107 |

`jumpscares` is the one I care about: it's the axis the motivating query is built
on, it had near-zero variance two batches earlier, and it more than doubled
(0.245 → 0.556) once the sampler started deliberately pairing horror shows that
rely on shocks against horror shows whose own descriptions say they don't. You
cannot learn that axis from horror shows alone, only from horror shows that
disagree on it.

The weak axes are all diffuse tonal qualities with no keyword vocabulary to
sample against. It's an open question whether keyword-based sampling can reach
them at all.

Held-out sanity check on shows never seen in training: *Better Call Saul* scores
0.00 on `jumpscares`, 0.00 on `horror`, and 0.74 on `dialogue_driven`. Nothing in
the training set resembles it.

### 4.2 Things worth defending in the viva

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

**I got this wrong twice before it worked, and both mistakes are worth writing
up.**

My first attempt picked query shows from the 400 most popular titles, assuming
popular meant I would know them. It came back **80–90% "don't know it"**. Out of
331 pairs I marked 14 as relevant, so there was almost nothing there to measure
with.

The fix was to add a screening step where I tick the shows I have actually
watched, and build the pool only from those. The thing I had missed is that the
two sides of a pair are not the same. I have to know the *query* show, because
the question is what I would recommend to someone who liked it. The candidate I
only need to be able to look at, since a poster and a description are enough to
say whether it looks like a sensible suggestion. Screening one side instead of
both is the difference between an hour of judging and a week of it.

The second mistake was in my scoring. I was counting "don't know it" as a miss,
which meant a system got punished for showing me an obscure show exactly as hard
as for showing me a bad one. Those are not the same failure and only one of them
is the system's fault.

This is a known problem in information retrieval, and the standard fix is the
condensed list (Sakai, 2007, following the same reasoning as bpref in Buckley
and Voorhees, 2004): throw away the results nobody could assess and score what
is left.

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

**Caveats, before anyone else points them out.** There is one judge and it is
me, the person who built the thing being measured. Only 12 of the 20 queries
were usable for the condensed measure. The intervals are wide.

[TODO: second judge on the same pool. Cohen's kappa between us is what turns this
from my opinion into a measurement, and it's the single highest-value thing left.]

I did get a version of this by accident. I judged the same 79 pairs twice about
forty minutes apart, once from the terminal and once in the browser, and because
I typed my name with different capitalisation the scorer treated me as two
different judges. Agreement between the two of me was **kappa 0.661**, which
Landis and Koch call substantial, and 11 of the 12 answers I changed were pairs
I had first marked as unknown.

It was a bug, but it is also a test–retest reliability measure, which is
something I would not otherwise have had.

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

### 5.8 Is the schema actually 37 things?

I wrote the 37 axes by hand before any scoring happened. That's the right order —
it stops a model inventing near-duplicates — but nothing in the process
guaranteed the axes are independent. If `creepy`, `unsettling` and `tense` always
move together they're one axis wearing three hats: three dimensions, three vote
targets and three lines of explanation, for one dimension of information.

PCA on the standardised score matrix over all 3,542 shows (correlation matrix,
not covariance, because the axes have very different spreads — `documentary` sits
near zero almost everywhere while `drama` is high across the catalogue):

| Variance kept | Components needed, of 37 |
|---|---|
| 80% | 4 |
| 90% | 7 |
| 95% | 10 |

Six components have an eigenvalue above 1 (Kaiser's rule). The first alone holds
43.2%.

That looks damning, so before concluding the schema is bloated I checked whether
the collapse belongs to the schema or to the model. Every predicted axis is a
linear function of the same 384-dimensional embedding, fitted by ridge on 78
examples — shrinkage pulls those directions towards each other, so a model can
manufacture correlation the schema doesn't have. Running the identical analysis
on the labels themselves separates the two:

| Variance kept | Labels (78 shows) | Predictions (3,542) |
|---|---|---|
| 80% | 7 | 4 |
| 90% | 11 | 7 |
| 95% | 15 | 10 |

**Both effects are real.** The labels need 11 components for 90%, so 37 axes are
carrying roughly 11 independent dimensions of human judgement — the schema *is*
redundant. But the predictions need only 7, so the model is compressing it
further, which is what ridge on 78 examples would be expected to do.

The most correlated pairs name exactly which axes to look at:

| | |
|---|---|
| `thriller` / `tense` | +0.97 |
| `horror` / `jumpscares` | +0.96 |
| `bleak` / `unsettling` | +0.94 |
| `creepy` / `jumpscares` | +0.94 |
| `thriller` / `plot_twists` | +0.94 |
| `warm` / `cosy` | +0.93 |

And the axes doing genuinely independent work, by their strongest correlation
with anything else: `historical` (0.54), `sci_fi` (0.62), `romance` (0.66),
`documentary` (0.70), `reality` (0.70), `ensemble` (0.72).

**What I'd actually do about it.** Not a straight prune. `horror` correlating
0.96 with `jumpscares` is partly a real property of the catalogue — most horror
on Netflix GB does use shocks — and partly the model being unable to tell them
apart at 78 labels. Merging them would destroy the exact distinction the
motivating query depends on ("horror, lots of jumpscares, simple plot"). The
honest conclusion is that the schema has room to lose maybe six to eight axes
among the mood cluster, and that the right test is to re-run this once the
labelled set is large enough for the two matrices to converge.

**Caveat.** 78 shows against 37 variables is a thin basis for PCA — the usual
rule of thumb wants five to ten observations per variable, so 185 to 370. The
label-side numbers should be treated as indicative.

[TODO: 5.9 label validity — Cohen's kappa between my own hand scores and model
output on 100 shows. Not done, and distinct from 5.8: this measures whether the
scores are *right*, not whether the axes are *distinct*.]

[TODO: 5.10 usability study, 5 participants, SUS plus time-to-completion.]

---

## 6. What I'd do differently

**The review corpus was the whole plan and the data did not support it.** I
assumed that because TMDB has a reviews endpoint, it would have reviews in it. I
should have spent an hour checking coverage in week one. Instead I found out
later, and everything downstream had to change because of it.

**I guessed instead of asking.** The first two judging sessions produced almost
nothing usable because I assumed popular shows would be shows I knew. I spent
about two hours judging pairs to discover that, and the screening step that
fixed it took twenty minutes to build.

**I set constants by reasoning and left them untested for months.** The
certificate penalty is the obvious one. It sounded so sensible that I never
questioned it, and when I finally measured it, it was quietly costing me
accuracy. If a number in my code was chosen because it felt right, that is a
reason to test it sooner, not later.

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
- **Prune the mood cluster.** 5.8 says the labels carry about 11 independent
  dimensions across 37 axes, and names the candidates. The test is to re-run that
  analysis once the labelled set is large enough for the label and prediction
  matrices to agree — merging axes on the current evidence risks destroying the
  `horror` / `jumpscares` distinction the whole project is built around.
- **Drop provenance keywords from the vocabulary**, not just from explanations.
  Currently `based on novel or book` still influences ranking; removing it would
  change results, so it's an ablation I haven't run.

---

## 8. References

Working list. **[TODO: check every one against the actual paper and reformat to
the school's required style — these are from memory and the details need
verifying before submission.]**

**The closest prior work**
- Vig, J., Sen, S. and Riedl, J. (2012) 'The Tag Genome: Encoding Community
  Knowledge to Support Novel Interaction', *ACM Transactions on Interactive
  Intelligent Systems*.
- Herlocker, J. et al. (2004) 'Evaluating Collaborative Filtering Recommender
  Systems', *ACM Transactions on Information Systems*.

**Evaluation methodology** — this is where most of my citations are, because it's
where I made the most decisions.
- Cleverdon, C. (1967) 'The Cranfield tests on index language devices',
  *Aslib Proceedings*. — pooling, and the whole test-collection idea.
- Buckley, C. and Voorhees, E. (2004) 'Retrieval Evaluation with Incomplete
  Information', *SIGIR*. — bpref; why unjudged results must not be counted as
  irrelevant (5.4).
- Sakai, T. (2007) 'Alternatives to Bpref', *SIGIR*. — condensed lists, which is
  what I actually implemented.
- Voorhees, E. (2000) 'Variations in relevance judgments and the measurement of
  retrieval effectiveness', *Information Processing and Management*. — judges
  disagree, and it matters less than you'd think for *comparing* systems. Backs
  my argument for a second judge.
- Cohen, J. (1960) 'A Coefficient of Agreement for Nominal Scales', *Educational
  and Psychological Measurement*.
- Landis, J. and Koch, G. (1977) 'The Measurement of Observer Agreement for
  Categorical Data', *Biometrics*. — the 0.41–0.60 / 0.61–0.80 bands I quote.
- Efron, B. (1979) 'Bootstrap Methods: Another Look at the Jackknife',
  *Annals of Statistics*. — every confidence interval in section 5.

**Models and representation**
- Reimers, N. and Gurevych, I. (2019) 'Sentence-BERT: Sentence Embeddings using
  Siamese BERT-Networks', *EMNLP*. — the encoder, and the baseline.
- Spärck Jones, K. (1972) 'A statistical interpretation of term specificity and
  its application in retrieval', *Journal of Documentation*. — IDF, which the
  keyword block and the explanation ordering both rest on.
- Burges, C. et al. (2005) 'Learning to Rank using Gradient Descent', *ICML*. —
  RankNet, cited as the methodological ancestor of the learned block weights I
  designed but haven't built.
- Ribeiro, M., Singh, S. and Guestrin, C. (2016) '"Why Should I Trust You?":
  Explaining the Predictions of Any Classifier', *KDD*. — LIME, cited for
  contrast: post-hoc explanation approximates a model it can disagree with,
  whereas mine is computed from the same vectors used to rank.
- McInnes, L., Healy, J. and Melville, J. (2018) 'UMAP: Uniform Manifold
  Approximation and Projection', *arXiv*. — for the 3D explorer in further work.

**Data**
- The Movie Database (TMDB) API. This product uses the TMDB API but is not
  endorsed or certified by TMDB.

---

## 9. Notes to self

Not part of the report.

- Declare the AI-generated landing page imagery (Gemini) if those screenshots
  appear in the submission. Attribution file is in `app/static/img/`.
- The TMDB attribution line must appear in the app: it does, in the footer.
- Ethics: no scraping, no personal data beyond locally-stored accounts,
  participants for the usability study need consent forms. [TODO: get the form.]
- Every number in section 5 regenerates from `evaluation/`. Re-run before
  submission in case the catalogue is refetched.
