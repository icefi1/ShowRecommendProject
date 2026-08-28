# Feature schema v0.1

40 axes. Every show is scored 0.0–1.0 on each. Fixed in advance: the model
never invents an axis (near-duplicate axes such as "creepy"/"spooky" push
identical shows apart and destroy the distance metric).

Scores are **descriptive, not evaluative**. The question is always "how much
does this describe the show?", never "is the show good?".

## Grounding tags

Each axis carries a tag saying where its score can actually come from. This
was determined by inventorying all 500 shows in `tmdb/shows_raw.json`.

| Tag | Meaning | Confidence |
|---|---|---|
| `META` | Computable from TMDB structured fields | High — deterministic |
| `KW` | Supported by TMDB keywords/genres at usable frequency | Medium |
| `TEXT` | Needs the LLM to read overview + keywords + any reviews | Medium-low |
| `WEAK` | **TMDB cannot evidence this.** Flagged as a known limitation | Low |

`WEAK` axes are kept in the schema deliberately. They are the axes the
project argues for, and reporting that the available data cannot support
them is a finding, not a gap to hide. See "Known limitation" below.

## Scoring convention

- `0.0` — absent. The word does not apply.
- `0.25` — present in one or two moments, incidental.
- `0.5` — a recurring element, but not what the show is about.
- `0.75` — a defining characteristic; most episodes.
- `1.0` — the show's central organising principle.

Score what the show *is*, averaged across its run — not its best episode.

---

## Block 1 — Genre (14 axes)

Topical. What the show is *about*. Weighted low when the user wants
"same vibe, different subject".

| Axis | Definition | Tag |
|---|---|---|
| `comedy` | Intends to make the viewer laugh; jokes are structural, not incidental. | `KW` |
| `drama` | Character conflict and interior stakes carry the story. | `KW` |
| `crime` | Crimes, their commission, or their investigation drive the plot. | `KW` |
| `mystery` | Withheld information the audience is invited to work out. | `KW` |
| `thriller` | Sustained forward tension built on threat and time pressure. | `KW` |
| `horror` | Aims to frighten. Distinct from `creepy`, which is a mood. | `KW` |
| `sci_fi` | Speculative technology or futurity is load-bearing. | `KW` |
| `fantasy` | Magic or invented worlds are load-bearing. | `KW` |
| `action` | Physical conflict, chases, fights are a primary draw. | `KW` |
| `romance` | A romantic relationship is a main plotline. | `KW` |
| `documentary` | Presents itself as non-fiction. | `META` |
| `reality` | Unscripted participants, constructed situations. | `META` |
| `animation` | Animated rather than live action. | `META` |
| `historical` | Set meaningfully in the past; period is load-bearing. | `KW` |

## Block 2 — Mood (13 axes)

How it feels to watch. The block the user raises for "something with the
same atmosphere".

| Axis | Definition | Tag |
|---|---|---|
| `creepy` | Unease and wrongness that lingers. Dread, not shock. | `TEXT` |
| `bleak` | Outcomes are bad and the show does not soften them. | `TEXT` |
| `warm` | Affection between characters is the point; kindness recurs. | `TEXT` |
| `tense` | The viewer is kept anxious about what happens next. | `TEXT` |
| `campy` | Knowingly excessive; invites you to enjoy the artifice. | `TEXT` |
| `melancholy` | Sad in a reflective register rather than a shocking one. | `TEXT` |
| `whimsical` | Playful strangeness, lightly worn. | `TEXT` |
| `cynical` | Assumes bad motives; institutions and people disappoint. | `TEXT` |
| `earnest` | Sincere, unironic, means what it says. | `TEXT` |
| `cosy` | Low-stakes comfort; the world is fundamentally safe. | `TEXT` |
| `absurd` | Logic of the world is deliberately nonsensical. | `TEXT` |
| `sentimental` | Actively reaches for the viewer's emotions. | `TEXT` |
| `unsettling` | Disturbing without being frightening; moral discomfort. | `TEXT` |

## Block 3 — Features (13 axes)

Experiential density and form. **The project's differentiator** — the block
the Tag Genome does not have, and the block that answers "lots of
jumpscares, but without the convoluted plot".

| Axis | Definition | Tag |
|---|---|---|
| `serialised` | Story continues across episodes; order matters. Inverse of episodic. | `META` |
| `episode_length` | Normalised runtime. 0.0 ≈ 10 min, 0.5 ≈ 30 min, 1.0 ≈ 70+ min. | `META` |
| `commitment` | Total episode count normalised. How much of your life this asks for. | `META` |
| `maturity` | Certificate-derived. U=0.0, PG=0.2, 12=0.4, 15=0.7, 18=1.0. | `META` |
| `jumpscares` | Density of sudden sensory shocks intended to startle. | `WEAK` |
| `gore` | On-screen graphic bodily harm. | `WEAK` |
| `plot_twists` | Frequency of revelations that reframe what came before. | `WEAK` |
| `plot_complexity` | How much the viewer must track to follow it. The "convoluted" axis. | `WEAK` |
| `slow_burn` | Deliberate pacing; payoff deferred. | `WEAK` |
| `ensemble` | Attention distributed across many characters vs one lead. | `TEXT` |
| `dialogue_driven` | Talk carries the show more than incident does. | `WEAK` |
| `visual_spectacle` | Draws on scale, effects, cinematography as an attraction. | `WEAK` |
| `emotional_intensity` | How hard the show pushes on feeling, regardless of valence. | `WEAK` |

---

## Known limitation

7 of the 13 Features axes are `WEAK`, and they include the four the project
was designed around: `jumpscares`, `plot_twists`, `plot_complexity`,
`slow_burn`.

Evidence, from the 500-show TMDB sample:

- TMDB TV has **no Horror genre** (15 genres, none is Horror or Thriller)
- `horror` keyword: 19/500 shows. `gore`: 4. `slasher`: 1
- **Zero of 2,184 distinct keywords describe an experiential quality** —
  nothing about being startled, confused, or made to wait
- 65% of the keyword vocabulary appears on exactly one show, so it carries
  no similarity signal; the usable vocabulary is ~82 terms
- TMDB reviews: 499 total across 500 shows, median 0 per show

TMDB describes *subject matter*. The Features block asks about *viewing
experience*. These are different things, and the gap is the finding.

Three ways forward, to be decided:

1. **Score `WEAK` axes from LLM world knowledge** rather than from supplied
   text. Cheap, and probably accurate for well-known shows. Costs the
   cold-start claim (§5) — a show released last week is not in the model's
   training data — and costs reproducibility, since the scores become a
   property of the model rather than of evidence.
2. **Drop the `WEAK` axes.** Schema shrinks to ~33 grounded axes. Honest and
   defensible, but the system becomes a Tag Genome reimplementation on TV,
   without the experiential differentiator.
3. **Keep them and measure the failure.** Score them anyway, then use §9.3
   (100 hand-tagged shows, Cohen's kappa) to report *per-axis* agreement.
   The prediction is that `META` axes score near-perfect, `KW` axes score
   well, and `WEAK` axes score poorly. That is a real result about what
   metadata can and cannot support, and it is the most defensible option.

Option 3 is recommended: it keeps the research question intact and turns the
data limitation into a measured contribution rather than a retreat.

---

## Next: hand-scoring gate (build order step 4)

Score 20 shows by hand against these definitions. If the same show cannot be
scored consistently twice, the definition is ambiguous and no model will fix
it. Expect to rewrite axes. Candidates from the sample spanning the space:
Breaking Bad, The Office, Rick and Morty, Stranger Things, Black Mirror,
Bridgerton, Peaky Blinders, Demon Slayer, Queer Eye, Chernobyl.

---

# v0.2 update — measured grounding

The episode pipeline has since run: 46,264 episodes across 499 shows, carrying
~1.66M words of episode overview text and 29,670 independently rated episodes.
That moves several axes off `WEAK`. It also produced one clear negative result,
recorded here because it is the kind of thing a report should not quietly drop.

## Axes now grounded by measurement

| Axis | Was | Now | How it is measured |
|---|---|---|---|
| `slow_burn` | `WEAK` | `META` | Least-squares slope of `vote_average` across season 1. Positive slope = opens weak, climbs. |
| `plot_complexity` | `WEAK` | `TEXT` | Proper-noun density across episode overviews — how many names and factions a viewer must track. |
| `emotional_intensity` | `WEAK` | `META` | `vote_count` peak-to-median ratio: how far the standout episode stands above the typical one. |
| `ensemble` | `TEXT` | `META` | Mean per-episode credited cast size (see failed prediction below). |

`serialised` is partly grounded via per-episode rating variance and
finale-versus-mean delta, but neither separates cleanly on inspection, so it
stays `WEAK` pending proper evaluation.

`jumpscares`, `gore` and `plot_twists` remain `WEAK`. Nothing in TMDB
evidences them, at either the series or the episode level.

## Failed prediction: guest star churn does not measure serialisation

`guest_star_mean` was added expecting it to separate procedurals (fresh guest
cast every week) from serialised drama (standing cast). **It does the
opposite.** Measured over the catalogue:

| Group | Mean credited guest stars per episode |
|---|---|
| Procedurals (SVU, The Mentalist, House, The Rookie, Bones) | 13.2 |
| Serialised (Breaking Bad, Stranger Things, Money Heist, Dark, Squid Game) | 19.2 |

The cause is a misreading of the field. TMDB's `guest_stars` is the per-episode
supporting cast credit, not a marker of one-off guest appearances. It therefore
tracks ensemble *size*, and large-ensemble serialised shows (Stranger Things
25.9, Dark 24.4) score highest.

Two consequences:

1. The axis is retained, but re-labelled as `ensemble`, which is what it
   actually measures, and the interface phrasing was corrected — an explanation
   that says "fresh cast weekly" about Stranger Things is worse than no
   explanation at all, because explainability is the contribution.
2. A genuine churn measure requires guest star *identities* compared across
   episodes, computing turnover rather than count. `fetch_episodes.py`
   currently keeps only `guest_star_count` and discards the identities, so this
   needs a re-fetch before it can be tested.

The general lesson, worth stating in the report: a derived feature is a
hypothesis, and hypotheses need checking against known cases before they are
wired into a distance metric. This one was checked and failed. Others in the
`META` column above are checked only in the weak sense that they behave
plausibly on inspection — §9 evaluation is what will actually settle them.

---

# v0.3 — first labelled batch, and a sampling failure

24 shows labelled (`labelling/labels.jsonl`), model trained. Headline: **31 of
37 axes beat the mean baseline**, mean MAE 0.176 on a 0–1 scale, from only 24
training examples. The mood block scores particularly well — `campy` 0.37,
`bleak` 0.27, `melancholy` 0.26, `whimsical` 0.23, `cosy` 0.23 — which matters,
because mood is the block TMDB metadata could not ground at all. The model is
reading mood out of text.

## The failure: the sample cannot see horror

The six axes with negative R² are `cynical`, `creepy`, `horror`, `earnest`,
`historical`, `sentimental`. Every one is an axis where the training set has
almost no variance. `horror` has label mean 0.07 and sd 0.09 across 24 shows.

Consequence on held-out titles:

| Show | predicted `horror` |
|---|---|
| The Haunting of Hill House | 0.10 |
| Stranger Things | 0.10 |
| Sweet Home | 0.09 |
| Wednesday | 0.10 |

The model has never seen a horror show, so it predicts the mean for all of them.

**The cause is the sampling method, not the model.** `export_batch.py`
stratifies by primary TMDB genre. TMDB TV has no Horror genre (see the v0.1
analysis above), so horror can never be selected as a stratum and none arrived
by chance. The stratification is systematically blind to exactly the axes this
project exists to capture.

This is a useful result rather than an embarrassing one: it is a concrete
demonstration that a missing category in the source taxonomy propagates all the
way through to a blind spot in a learned model, and that per-axis evaluation
catches it while an aggregate score would have hidden it. Mean MAE of 0.176
looks fine; it conceals a model that cannot recognise horror at all.

## Fix for batch 2

Sample to cover **label space**, not genre space. Deliberately include titles
likely to score high on the under-covered axes — horror and creepy (The
Haunting of Hill House, Sweet Home, Black Mirror), historical (The Crown,
Peaky Blinders, Chernobyl), cynical and campy. Then re-check per-axis variance
in the training set *before* training, so the gap is caught automatically
rather than by inspection.

---

# v0.4 — coverage sampling fixes the blind spots

Batch 2 selected 26 shows by keyword evidence for the axes batch 1 could not
teach (`labelling/export_coverage.py`), rather than by TMDB primary genre.
50 labelled shows total.

| | Batch 1 (24) | Batch 2 (50) |
|---|---|---|
| Axes beating mean baseline | 31/37 | **36/37** |
| `horror` R² | −0.081 | **+0.343** |
| `historical` R² | −0.167 | **+0.311** |
| `creepy` R² | −0.066 | **+0.299** |
| `jumpscares` R² | no signal | **+0.245** |
| `horror` label sd | 0.092 | 0.274 |

Held-out shows the model has never seen now order correctly:

| Show | horror | romance | historical |
|---|---|---|---|
| The Sandman | 0.35 | 0.18 | 0.15 |
| Sweet Home | 0.34 | 0.37 | 0.08 |
| Bridgerton | 0.08 | 0.50 | 0.28 |
| Emily in Paris | 0.10 | 0.44 | 0.21 |
| Vikings | 0.15 | 0.30 | 0.27 |

Magnitudes stay compressed toward the mean — ridge shrinkage at n=50 — so the
ordering is trustworthy and the absolute values are not. For a distance-based
recommender that is the right way round, but it is why predicted axes are
**displayed and not yet used for ranking**.

## Why the taxonomy gap is the project's central argument

TMDB TV offers 15 genres and includes **no Horror, Romance, Thriller, History
or Fantasy**. Those exist in TMDB's movie taxonomy; the TV list omits them.
The consequence over this catalogue:

- 85 shows carry a `romance` keyword. **Zero** can be labelled Romance.
- Stranger Things is labelled *Action & Adventure, Mystery, Sci-Fi & Fantasy*.
- Black Mirror is labelled *Sci-Fi & Fantasy, Drama, Mystery*.

The shows exist; the vocabulary to describe them does not. A recommender
restricted to TMDB's genres cannot accept the request "a romance" at all.
An interpretable axis space can, because the axes are defined by the schema
rather than inherited from the source. This is a stronger justification for the
approach than competitive retrieval accuracy would be, and it is measurable.

---

# v0.6 — accounts, and reviews as the missing corpus

## Voting now requires an account

The previous localStorage voter id meant one person could vote unlimited times
by clearing browser storage, which made every tally meaningless. Identity now
comes from an httpOnly session cookie, resolved server-side; the vote request
body carries no identity field at all.

## Reviews close the loop S4 could not close

This is the important part. Report S4 established that experiential axes need
review text, and that TMDB provides 499 reviews across 500 shows with a median
of zero — unusable. Reviews written by members here are:

- text about the viewing experience, not marketing copy
- attached to a known TMDB show id, so no fuzzy title matching
- free of the licensing problem that rules out scraping IMDb

`AccountStore.review_text_for(show_id)` returns them for the labelling pipeline,
so the corpus grows with use rather than being fixed at ingestion time. The
system does not depend on this to work — the trained model already produces
scores — but every review measurably improves the text available for the axes
that TMDB metadata cannot evidence.

## Security decisions worth defending

| Decision | Reason |
|---|---|
| scrypt (RFC 7914), per-user salt | Memory-hard KDF, in the standard library, no dependency. Parameters stored per hash so they can be raised later without invalidating existing passwords. |
| Hash even when the username does not exist | Otherwise response time reveals which usernames are registered. Measured: 51 ms vs 50 ms. |
| `secrets.compare_digest` | Constant-time; `==` leaks how many leading bytes matched. |
| httpOnly, SameSite=Lax cookie | Page scripts cannot read the token, so an XSS bug cannot exfiltrate a session. |
| "Incorrect username or password" | A distinct "no such user" confirms which accounts exist. |
| Reviews escaped before insertion | Reviews are written by other people; inserting them as HTML would be stored XSS. |

`secure=False` on the cookie is correct only for plain-http localhost and must
be `True` behind HTTPS.

## Landing page

Guests get `home.html`, signed-in users get the app. The page reveals a section
at a time on scroll, with two pinned panels — a sticky child inside a tall
wrapper, covered by a following section with a solid background and a higher
stacking order.

One robustness note: the reveal animation initially left every section at
`opacity: 0` if IntersectionObserver did not fire, which would have shown a
blank page whenever scripting was slow, blocked, or the tab was not
compositing. Content is now visible by default, and hidden only once an inline
script confirms scripting is alive to put it back.

---

# v0.7 — facts are not opinions

## The problem

Every predicted axis was votable, including `animation`. A show is animated or
it is not; polling that invites disagreement about a settled question, and the
model was *predicting* it as a continuous score, so an obviously animated show
could come back at 0.6.

## The rule

**If a catalogue source asserts it, it is a fact and the source wins. If no
source asserts it, it is a judgement — the model predicts and the crowd
corrects.**

| Kind | Axes | Behaviour |
|---|---|---|
| Fact, binary | `animation`, `documentary`, `reality` | Written as 0 or 1 from TMDB. "40% animated" is not a thing. |
| Fact, degree | `comedy`, `drama`, `crime`, `mystery`, `action` | TMDB decides which side of 0.5; the model supplies the degree within that half. Membership is a fact, emphasis is not. |
| Judgement | the other 29, including `horror`, `thriller`, `romance`, `historical` | Model predicts, crowd votes. |

The rule lands exactly where the project's argument does. TMDB has no Horror,
Thriller, Romance or History genre, so those four have no authority to defer to
— they are precisely the axes worth voting on. `sci_fi` and `fantasy` stay
judgements too: TMDB conflates them into one tag, so it says at least one
applies but never which.

Votes on fact axes are refused with a 400 rather than silently dropped, and a
schema assertion fails the build if any of the four unsourced genres is ever
moved into the fact set.

**Trade-off:** if TMDB is wrong about a genre, users cannot correct it here.
That is the price of treating the source as authoritative; the remedy is to add
or fix sources, not to let votes overrule facts.

## Age certificates now shape ranking

`maturity` was already one of 13 dimensions in the structure block — but that
block carries 0.20 of the total weight, so it contributed about **1.5%** of a
score. Nowhere near enough to stop a U-rated cartoon ranking against an 18.

It is now applied separately, as a multiplier on the finished score:

    score *= 1 - 0.5 * |maturity_query - maturity_result|

The widest gap (U against 18) halves a score; one step (15 against 12) costs
about 15%. That demotes rather than excludes, which is the intent.

**Coverage matters more than the formula.** GB certificates cover only 69% of
the catalogue, and an unrated show defaults to mid-scale and then ranks against
everything — Teen Titans Go! had no GB rating and was pulling 15-rated shows
into its results. Falling back to US TV ratings takes coverage to **94%**.

| Query | Before | After |
|---|---|---|
| Teen Titans Go! (TV-PG) | 15-rated *Titans* at #3 | PG, TV-G, PG — *Titans* demoted |
| Sesame Street (U) | mixed | TV-Y, TV-Y, U |
| Breaking Bad (18) | unrated *El Chapo* penalised to #7 | TV-MA *El Chapo* back at #1 |

That last row is the fallback proving itself: El Chapo was being punished for
missing data rather than for being unsuitable.

`NR` is deliberately unmapped — "not rated" is missing data, not a rating, and
giving it a number would invent information. 6% of the catalogue remains
unrated and sits mid-scale.
