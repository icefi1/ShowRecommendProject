"""
The feature schema, machine-readable.

docs/feature_schema.md is the prose companion to this file - it carries the
scoring anchors, the grounding analysis and the reasoning. This module is what
the code imports, so the axis list has exactly one definition.

Three axes from the Features block are deliberately NOT scored by the LLM:

    episode_length, commitment, maturity

They are arithmetic on TMDB fields - runtime, episode count, and the UK
certificate - so a measured value is strictly better than a judgement, and
asking for them would spend tokens to get a worse answer. They are filled in
from metadata in app/build_space.py instead.

That leaves 37 axes for the labeller.
"""

# (name, block, definition shown to the labeller)
AXES = [
    # ---------------------------------------------------------------- genre
    ("comedy", "genre", "Intends to make the viewer laugh; jokes are structural, not incidental."),
    ("drama", "genre", "Character conflict and interior stakes carry the story."),
    ("crime", "genre", "Crimes, their commission, or their investigation drive the plot."),
    ("mystery", "genre", "Withheld information the audience is invited to work out."),
    ("thriller", "genre", "Sustained forward tension built on threat and time pressure."),
    ("horror", "genre", "Aims to frighten. Distinct from the creepy mood axis."),
    ("sci_fi", "genre", "Speculative technology or futurity is load-bearing."),
    ("fantasy", "genre", "Magic or invented worlds are load-bearing."),
    ("action", "genre", "Physical conflict, chases and fights are a primary draw."),
    ("romance", "genre", "A romantic relationship is a main plotline."),
    ("documentary", "genre", "Presents itself as non-fiction."),
    ("reality", "genre", "Unscripted participants in constructed situations."),
    ("animation", "genre", "Animated rather than live action."),
    ("historical", "genre", "Set meaningfully in the past; the period is load-bearing."),

    # ----------------------------------------------------------------- mood
    ("creepy", "mood", "Unease and wrongness that lingers. Dread, not shock."),
    ("bleak", "mood", "Outcomes are bad and the show does not soften them."),
    ("warm", "mood", "Affection between characters is the point; kindness recurs."),
    ("tense", "mood", "The viewer is kept anxious about what happens next."),
    ("campy", "mood", "Knowingly excessive; invites you to enjoy the artifice."),
    ("melancholy", "mood", "Sad in a reflective register rather than a shocking one."),
    ("whimsical", "mood", "Playful strangeness, lightly worn."),
    ("cynical", "mood", "Assumes bad motives; institutions and people disappoint."),
    ("earnest", "mood", "Sincere, unironic, means what it says."),
    ("cosy", "mood", "Low-stakes comfort; the world is fundamentally safe."),
    ("absurd", "mood", "The logic of the world is deliberately nonsensical."),
    ("sentimental", "mood", "Actively reaches for the viewer's emotions."),
    ("unsettling", "mood", "Disturbing without being frightening; moral discomfort."),

    # ------------------------------------------------------------- features
    ("serialised", "features", "Story continues across episodes; watching in order matters."),
    ("jumpscares", "features", "Density of sudden sensory shocks intended to startle."),
    ("gore", "features", "On-screen graphic bodily harm."),
    ("plot_twists", "features", "Frequency of revelations that reframe what came before."),
    ("plot_complexity", "features", "How much the viewer must actively track to follow it."),
    ("slow_burn", "features", "Deliberate pacing; payoff deferred rather than delivered early."),
    ("ensemble", "features", "Attention distributed across many characters rather than one lead."),
    ("dialogue_driven", "features", "Talk carries the show more than incident does."),
    ("visual_spectacle", "features", "Draws on scale, effects and cinematography as an attraction."),
    ("emotional_intensity", "features", "How hard the show pushes on feeling, regardless of valence."),
]

# Computed from TMDB metadata rather than judged. See module docstring.
METADATA_AXES = ["episode_length", "commitment", "maturity"]

AXIS_NAMES = [name for name, _, _ in AXES]
BLOCKS = {block for _, block, _ in AXES}

assert len(AXIS_NAMES) == len(set(AXIS_NAMES)), "duplicate axis name in schema"
assert len(AXES) + len(METADATA_AXES) == 40, "schema should total 40 axes"


def definitions_block():
    """The axis list as it appears in the labelling prompt, grouped by block."""
    lines = []
    for block in ("genre", "mood", "features"):
        lines.append(f"\n## {block.upper()}")
        for name, axis_block, definition in AXES:
            if axis_block == block:
                lines.append(f"- {name}: {definition}")
    return "\n".join(lines)
