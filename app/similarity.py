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
# (axis: wording when the result is higher, wording when lower)
#
# Each phrase carries its own verb. The old table stored bare noun phrases and
# the sentence was built as "but is " + phrase, which produced "but is shorter
# episodes" and "but is more names and factions to track" - ungrammatical for
# every count-noun axis. Owning the verb per axis fixes that with no extra logic.
STRUCTURE_PHRASING = {
    "episode_count": ("has far more episodes", "has far fewer episodes"),
    "season_count": ("has many more seasons", "has many fewer seasons"),
    "episode_length": ("has longer episodes", "has shorter episodes"),
    "maturity": ("is more adult", "is less adult"),
    "audience_rating": ("is better reviewed", "is less well reviewed"),
    "is_miniseries": ("is a self-contained miniseries", "is an ongoing series"),
    "rating_stdev": ("is much more variable episode to episode", "is much more consistent"),
    "slow_burn_slope": ("is more of a slow burn", "is stronger from the start"),
    "finale_delta": ("is more built around its finales", "is less finale-driven"),
    "vote_peak_ratio": ("is more centred on standout episodes", "is more evenly watched"),
    # Measured, not assumed: this was built expecting it to separate procedurals
    # (fresh cast weekly) from serialised drama (standing cast). It does not.
    # TMDB's guest_stars is the per-episode supporting cast credit, so it tracks
    # ensemble SIZE, and serialised shows scored higher than procedurals
    # (19.2 vs 13.2 average). Phrased here as what it actually measures. A real
    # churn measure needs guest star identities across episodes, which
    # fetch_episodes.py currently discards - see docs/feature_schema.md.
    "guest_star_mean": ("has a larger ensemble cast", "has a tighter core cast"),
    "entity_density": ("has more names and factions to track", "has a simpler cast of characters"),
    "runtime_stdev": ("has more varied episode lengths", "has more uniform episode lengths"),
}


# Human phrasing for TMDB's 15 television genres.
# (genre: form used as a modifier, form used as the noun)
#
# Two forms because an explanation reads better as "both crime dramas" than as
# "both crime and drama". The broadest shared genre supplies the noun and the
# most specific supplies the modifier - see _genre_clause.
#
# The slashes on the conflated genres are deliberate: TMDB really does file
# science fiction and fantasy under one heading, and war alongside politics.
# Writing them out that way keeps the interface honest about the taxonomy it
# inherited, which is the same limitation the interpretable axes exist to fix.
GENRE_PHRASING = {
    "Action & Adventure": ("action-adventure", "action-adventure shows"),
    "Animation": ("animated", "animated shows"),
    "Comedy": ("comedy", "comedies"),
    "Crime": ("crime", "crime shows"),
    "Documentary": ("documentary", "documentaries"),
    "Drama": ("drama", "dramas"),
    "Family": ("family", "family shows"),
    "Kids": ("kids'", "kids' shows"),
    "Mystery": ("mystery", "mysteries"),
    "Reality": ("reality", "reality shows"),
    "Sci-Fi & Fantasy": ("sci-fi/fantasy", "sci-fi/fantasy shows"),
    "Soap": ("soap", "soaps"),
    "Talk": ("talk", "talk shows"),
    "War & Politics": ("war/politics", "war and politics shows"),
    "Western": ("western", "westerns"),
}

# How many structural differences an explanation may end with.
#
# It was two. One is a trade-off made on purpose: the structure block is the
# least interesting thing about a recommendation - nobody chooses a show for its
# episode-length percentile - and every structural clause pushes the shared
# genre and subject further from the start of the sentence, which is where the
# eye lands. Losing a second difference costs some information; burying the
# reason the show was recommended costs more.
MAX_STRUCTURE_POINTS = 1


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

        # How many shows carry each genre. Used only when writing explanations:
        # the commonest shared genre is the broadest description of the pair
        # ("dramas") and the rarest is the most specific ("crime"), so the pair
        # reads as one noun phrase. Same reasoning as the IDF weighting on
        # keywords - rarer means more informative - reused on a block that is
        # binary and therefore has no IDF of its own.
        self.genre_counts = self.blocks["genre"].sum(axis=0)

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

    def _is_provenance(self, term):
        """
        True for keywords describing where a show came from rather than what it
        is like to watch: "based on novel or book", "based on true story",
        "remake", and the 22 other variants in the vocabulary.

        They are the highest-frequency keywords in the catalogue, so they match
        constantly and say nothing a viewer chose the show for. Note this hides
        them from the EXPLANATION only - they still carry their IDF weight in
        the vector and still influence ranking. Dropping them from the
        vocabulary outright is the stronger option and would change results;
        that is an ablation for the evaluation chapter, not a silent change
        here.
        """
        return term.startswith("based on") or "remake" in term

    def _genre_clause(self, query_index, result_index):
        """
        "both crime dramas" - the genres the two shows have in common.

        The genre block is binary, so shared genres are an element-wise minimum
        exactly as with keywords. Which two to name is decided by catalogue
        frequency: the commonest shared genre becomes the noun because it is the
        broadest true statement about the pair, and the rarest becomes the
        modifier because it is the most specific. With Drama on 1,707 shows and
        Crime on 621, a pair sharing both reads "both crime dramas" rather than
        the less informative "both drama crimes".
        """
        names = self.block_labels["genre"]
        shared = np.minimum(self.blocks["genre"][query_index], self.blocks["genre"][result_index])
        positions = [i for i in range(len(names)) if shared[i] > 0]
        if not positions:
            return ""

        # Commonest first, so positions[0] is the noun and positions[-1] the modifier.
        positions.sort(key=lambda i: -self.genre_counts[i])
        head_genre = names[positions[0]]
        # A genre TMDB added after this table was written still gets a sentence.
        _, noun = GENRE_PHRASING.get(head_genre, (head_genre.lower(), head_genre.lower() + " shows"))

        if len(positions) == 1:
            return f"both {noun}"

        modifier_genre = names[positions[-1]]
        modifier, _ = GENRE_PHRASING.get(modifier_genre, (modifier_genre.lower(), ""))
        return f"both {modifier} {noun}"

    def _keyword_clause(self, query_index, result_index, max_keywords):
        """"shares drug cartels, outlaw" - the rarest keywords present in both."""
        matrix = self.blocks["keywords"]
        vocabulary = self.block_labels["keywords"]

        shared = np.minimum(matrix[query_index], matrix[result_index])
        terms = []
        # Highest weight = rarest shared keyword = most informative.
        for position in np.argsort(-shared):
            if shared[position] <= 0:
                break
            term = vocabulary[position]
            if self._is_provenance(term):
                continue
            terms.append(term)
            if len(terms) >= max_keywords:
                break

        return "shares " + ", ".join(terms) if terms else ""

    def _structure_clause(self, query_index, result_index):
        """"but has shorter episodes" - the largest difference in form."""
        matrix = self.blocks["structure"]
        names = self.block_labels["structure"]
        deltas = matrix[result_index] - matrix[query_index]

        phrases = []
        for position in np.argsort(-np.abs(deltas)):
            axis = names[position]
            delta = float(deltas[position])
            # Below this the two shows are effectively the same on that axis and
            # saying so out loud would be noise.
            if abs(delta) < 0.25 or axis not in STRUCTURE_PHRASING:
                continue
            higher, lower = STRUCTURE_PHRASING[axis]
            phrases.append(higher if delta > 0 else lower)
            if len(phrases) >= MAX_STRUCTURE_POINTS:
                break

        return "but " + " and ".join(phrases) if phrases else ""

    def explain(self, query_index, result_index, max_keywords=3):
        """
        Say in words why this result came back (report S6.6).

        Three clauses, always in this order, because that is the order a viewer
        cares about them in:

            both crime dramas; shares drug cartels, outlaw; but has shorter episodes
            \_____ genre _____/  \______ keywords _______/  \____ structure ____/

        Ordering is the whole point of this method. Measured over 600
        explanations, a quarter of them opened with "but is ..." and listed only
        structural differences - the reader was told how two shows differ before
        being told they had anything in common, or in those cases instead of it.
        Episode-length percentiles are not why anyone watches a television
        programme, so structure is now a trailing qualifier and never the lead.

        Every clause is read straight out of the vectors that did the ranking.
        There is no separate explanation model that could disagree with the
        result it is explaining, which is what makes this an explanation rather
        than a plausible-sounding caption.
        """
        parts = [
            self._genre_clause(query_index, result_index),
            self._keyword_clause(query_index, result_index, max_keywords),
        ]
        parts = [clause for clause in parts if clause]

        if not parts:
            # Nothing shared in either block, so the result is here on form
            # alone. Listing its structural differences would be the failure
            # this ordering exists to remove: an explanation that never says why
            # the show was recommended. Say what actually happened instead.
            return "no shared genre or subject - matched on form alone"

        structure = self._structure_clause(query_index, result_index)
        if structure:
            parts.append(structure)

        return "; ".join(parts)
