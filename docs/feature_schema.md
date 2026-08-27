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
