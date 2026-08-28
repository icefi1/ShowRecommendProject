"""
Blocked weighted similarity over the feature space, plus explanation generation.

This is report S6.4 and S6.6. Two design commitments matter here:

1. Distance is computed per block, then blocks are combined with weights set at
   query time. A single flat cosine over all 461 dimensions would let a strong
   keyword match and a strong structural match produce the same score, with no
   way for the user to tell which happened or to ask for more of one.

2. Neighbours are always computed in the full-dimensional space. Nothing is
   ranked in a projection.
"""

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# What each block contributes when the user has not touched the weights.
# Keywords carry the most specific signal about subject matter, so they lead.
DEFAULT_WEIGHTS = {"genre": 0.25, "keywords": 0.55, "structure": 0.20}

# How hard a certificate mismatch is punished.
#
# Age rating is already one of 13 dimensions in the structure block, but that
# block carries 0.20 of the total weight, so maturity contributes barely 1.5%
# of a score - nowhere near enough to stop a U-rated cartoon ranking against an
# 18-rated drama when their keywords happen to align.
#
# So it is applied separately, as a multiplier on the finished score:
#
#     score *= 1 - MATURITY_PENALTY * |maturity_query - maturity_result|
#
# At 0.5, the widest possible gap (U against 18) halves a result's score, and a
# one-step gap such as 15 against 12 costs about 15%. That demotes rather than
# excludes, which is the intent: a mismatched certificate should push a show
# down the list, not remove it from consideration.
#
# The value is immutable - it comes from the classification body via TMDB, is
# never predicted by the model and never open to voting.
MATURITY_PENALTY = 0.5

# Human phrasing for structure axes, used when explaining a result.
# (axis, wording when the result is higher, wording when lower)
STRUCTURE_PHRASING = {
    "episode_count": ("far more episodes", "far fewer episodes"),
    "season_count": ("many more seasons", "many fewer seasons"),
    "episode_length": ("longer episodes", "shorter episodes"),
    "maturity": ("more adult", "less adult"),
    "audience_rating": ("better reviewed", "less well reviewed"),
    "is_miniseries": ("a self-contained miniseries", "an ongoing series"),
    "rating_stdev": ("much more variable episode to episode", "much more consistent"),
    "slow_burn_slope": ("more of a slow burn", "stronger from the start"),
    "finale_delta": ("more built around its finales", "less finale-driven"),
    "vote_peak_ratio": ("more centred on standout episodes", "more evenly watched"),
    # Measured, not assumed: this was built expecting it to separate procedurals
    # (fresh cast weekly) from serialised drama (standing cast). It does not.
    # TMDB's guest_stars is the per-episode supporting cast credit, so it tracks
    # ensemble SIZE, and serialised shows scored higher than procedurals
    # (19.2 vs 13.2 average). Phrased here as what it actually measures. A real
    # churn measure needs guest star identities across episodes, which
    # fetch_episodes.py currently discards - see docs/feature_schema.md.
    "guest_star_mean": ("a larger ensemble cast", "a tighter core cast"),
    "entity_density": ("more names and factions to track", "a simpler cast of characters"),
    "runtime_stdev": ("more varied episode lengths", "more uniform episode lengths"),
}


class FeatureSpace:
    """The catalogue, its blocked vectors, and queries over them."""

    def __init__(self):
        arrays = np.load(HERE / "feature_space.npz")
        self.blocks = {name: arrays[name] for name in ("genre", "keywords", "structure")}

        meta = json.loads((HERE / "feature_space.json").read_text(encoding="utf-8"))
        self.catalogue = meta["catalogue"]
        self.block_labels = meta["blocks"]

        self.index_by_id = {show["id"]: i for i, show in enumerate(self.catalogue)}
        # Lowercased once at startup so search does not re-lower 500 strings
        # on every keystroke.
        self.search_names = [show["name"].lower() for show in self.catalogue]

        # Row norms never change once the space is built, so computing them per
        # query was pure waste - it dominated query time. Precomputing here cut
        # a similarity query from ~54ms to under 1ms, which is the difference
        # between a visible pause and an instant response while dragging a dial.
        self.block_norms = {
            name: np.linalg.norm(matrix, axis=1) for name, matrix in self.blocks.items()
        }

        # Certificate position per show, pulled out of the structure block so
        # the penalty is one vector op rather than a lookup per candidate.
        maturity_col = self.block_labels["structure"].index("maturity")
        self.maturity = self.blocks["structure"][:, maturity_col].astype(np.float32)

    # ------------------------------------------------------------ searching

    def search(self, term, limit=10):
        """
        Find shows by title. Exact prefix matches rank above substring matches,
        which is what a user typing a title expects.
        """
        term = term.strip().lower()
        if not term:
            return []

        starts, contains = [], []
        for index, name in enumerate(self.search_names):
            if name.startswith(term):
                starts.append(index)
            elif term in name:
                contains.append(index)

        # Within each tier, more popular shows first - the catalogue arrives
        # from TMDB in popularity order, so the index itself is the tiebreak.
        return [self.catalogue[i] for i in (starts + contains)[:limit]]

    # ---------------------------------------------------------- similarity

    def _block_similarity(self, block, query_row):
        """
        Cosine similarity between one query vector and every show, for one block.

        Both sides are L2-normalised, so this is a single matrix-vector product:
        one numpy operation over the whole catalogue rather than a Python loop.
        That is where the "efficient" in the project brief is actually earned.
        """
        matrix = self.blocks[block]

        matrix_norms = self.block_norms[block]
        query_norm = np.linalg.norm(query_row)
        if query_norm == 0:
            return np.zeros(len(matrix), dtype=np.float32)

        scores = matrix @ query_row
        denominator = matrix_norms * query_norm
        # Shows with an all-zero block (no keywords at all) score 0, not NaN.
        return np.divide(
            scores, denominator, out=np.zeros_like(scores), where=denominator > 0
        )

    def similar_to_show(self, show_id, weights=None, limit=10):
        """Rank the catalogue against one show the user picked."""
        index = self.index_by_id.get(show_id)
        if index is None:
            return []

        weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        query = {name: matrix[index] for name, matrix in self.blocks.items()}

        combined, per_block = self._combine(query, weights)
        combined = self._apply_maturity(combined, self.maturity[index])
        combined[index] = -1.0  # never recommend the query back to itself

        return self._rank(combined, per_block, limit, compare_to=index)

    def by_preference(self, structure_targets, genre_targets, weights=None, limit=10):
        """
        Rank against a query the user built from dials rather than a show.

        This is the "or preference" half of the project brief: the user never
        has to name a title, they can describe what they want directly.
        """
        weights = dict(weights or DEFAULT_WEIGHTS)

        structure_names = self.block_labels["structure"]
        structure_query = np.full(len(structure_names), 0.5, dtype=np.float32)
        for axis, value in (structure_targets or {}).items():
            if axis in structure_names:
                structure_query[structure_names.index(axis)] = float(value)

        genre_names = self.block_labels["genre"]
        genre_query = np.zeros(len(genre_names), dtype=np.float32)
        for genre in genre_targets or []:
            if genre in genre_names:
                genre_query[genre_names.index(genre)] = 1.0

        query = {
            "genre": genre_query,
            "keywords": np.zeros(self.blocks["keywords"].shape[1], dtype=np.float32),
            "structure": structure_query,
        }

        # With no title to work from there are no keywords to match on, so that
        # block would contribute nothing but would still dilute the weighting.
        if not genre_targets:
            weights = {**weights, "genre": 0.0}
        weights = {**weights, "keywords": 0.0}

        combined, per_block = self._combine(query, weights)
        # If the user moved the maturity dial, treat it as the target rating.
        combined = self._apply_maturity(combined, (structure_targets or {}).get("maturity"))
        return self._rank(combined, per_block, limit, compare_to=None)

    # ------------------------------------------------------------- internals

    def _combine(self, query, weights):
        """Weighted sum of per-block similarities, plus the parts, for display."""
        per_block, total = {}, 0.0
        combined = np.zeros(len(self.catalogue), dtype=np.float32)

        for name, weight in weights.items():
            if weight <= 0:
                continue
            scores = self._block_similarity(name, query[name])
            per_block[name] = scores
            combined += weight * scores
            total += weight

        if total > 0:
            combined /= total
        return combined, per_block

    def _apply_maturity(self, combined, query_maturity):
        """
        Demote results whose age rating is far from the query's.

        Multiplicative rather than subtractive, so it scales a score instead of
        flattening weak matches to zero, and a strong match with a mild
        certificate gap still outranks a poor match with none.
        """
        if query_maturity is None:
            return combined
        gap = np.abs(self.maturity - float(query_maturity))
        return combined * (1.0 - MATURITY_PENALTY * gap)

    def _rank(self, combined, per_block, limit, compare_to):
        """Take the top `limit` scores and attach an explanation to each."""
        # argpartition finds the top k without sorting all 500 - the standard
        # trick when k is small relative to n.
        top = np.argpartition(-combined, min(limit, len(combined) - 1))[:limit]
        top = top[np.argsort(-combined[top])]

        results = []
        for index in top:
            show = dict(self.catalogue[index])
            show["score"] = round(float(combined[index]), 4)
            show["maturity_gap"] = None
            show["block_scores"] = {
                name: round(float(scores[index]), 3) for name, scores in per_block.items()
            }
            show["explanation"] = (
                self.explain(compare_to, int(index)) if compare_to is not None else ""
            )
            results.append(show)

        return results

    # ----------------------------------------------------------- explanation

    def explain(self, query_index, result_index, max_points=3):
        """
        Say in words why this result came back (report S6.6).

        Two ingredients: what the two shows share (the highest-IDF keywords
        present in both, which are the most specific things they have in
        common) and how they differ structurally (the largest gaps on named
        axes). Both fall directly out of the representation - there is no
        separate explanation model that could disagree with the ranking.
        """
        keyword_matrix = self.blocks["keywords"]
        vocabulary = self.block_labels["keywords"]

        shared = np.minimum(keyword_matrix[query_index], keyword_matrix[result_index])
        # Highest weight = rarest shared keyword = most informative.
        top_shared = [vocabulary[i] for i in np.argsort(-shared)[:max_points] if shared[i] > 0]

        structure_matrix = self.blocks["structure"]
        structure_names = self.block_labels["structure"]
        deltas = structure_matrix[result_index] - structure_matrix[query_index]

        differences = []
        for position in np.argsort(-np.abs(deltas)):
            axis = structure_names[position]
            delta = float(deltas[position])
            # Below this the two shows are effectively the same on that axis and
            # saying so out loud would be noise.
            if abs(delta) < 0.25 or axis not in STRUCTURE_PHRASING:
                continue
            higher, lower = STRUCTURE_PHRASING[axis]
            differences.append(higher if delta > 0 else lower)
            if len(differences) >= 2:
                break

        parts = []
        if top_shared:
            parts.append("shares " + ", ".join(top_shared))
        if differences:
            parts.append("but is " + " and ".join(differences))

        return "; ".join(parts) if parts else "similar overall profile"
