"""
Crowd correction layer: votes as evidence updating the model's prediction.

This is report S6.3. Three design commitments, each of which has a failure mode
it exists to avoid.

1. VOTES ARE DESCRIPTIVE, NEVER EVALUATIVE
   The question is always "does `creepy` describe this show?" and never "did you
   like this show?". If the two are blended, every popular show drifts high on
   every flattering axis and the descriptions quietly become approval ratings.
   The interface wording enforces this; so does the fact that there is no
   whole-show vote at all, only per-axis votes.

2. THE MODEL IS A PRIOR, VOTES ARE EVIDENCE
   The naive approach - score = upvotes / (upvotes + downvotes) - reads 3 up and
   0 down as a confident 1.0. That is wrong, and it also means the system does
   nothing at all until it has users. Instead the model's prediction acts as a
   prior with a pseudo-count, and votes update it:

       posterior = (kappa * prior + sum_of_vote_targets) / (kappa + n_votes)

   With no votes the posterior IS the model's prediction, so the system works on
   day one. A few votes shift it slightly. Sustained agreement moves it a long
   way. `kappa` is how many votes it takes to pull the score halfway to the
   crowd's opinion.

3. A VOTE IS RELATIVE TO WHAT THE VOTER SAW
   "Higher" only means something against a displayed number, so every vote
   stores the score that was on screen when it was cast. Up means "higher than
   that", down means "lower than that", and neutral means "that value is
   already right".

   This makes the system self-correcting rather than runaway. As up-votes push a
   score higher, later voters see the higher number, and once it is right they
   start voting neutral - which targets the current value and anchors it. The
   score converges instead of drifting to the extreme.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DB_PATH = HERE / "crowd.db"

# How far one up/down vote asks the score to move. A design parameter, not a
# discovered constant: larger converges faster but overshoots more.
VOTE_STEP = 0.15

# Prior strength, in pseudo-votes. At kappa=8 it takes 8 agreeing votes to pull
# a score halfway from the model's prediction to the crowd's opinion. Chosen so
# that a handful of votes cannot overturn a trained prediction, but a sustained
# consensus can.
PRIOR_STRENGTH = 8.0

DIRECTIONS = ("up", "down", "neutral")

SCHEMA = """
CREATE TABLE IF NOT EXISTS votes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    show_id     INTEGER NOT NULL,
    axis        TEXT    NOT NULL,
    direction   TEXT    NOT NULL CHECK (direction IN ('up', 'down', 'neutral')),
    -- The value displayed when this vote was cast. Without it, "higher" has no
    -- referent and the vote cannot be turned into a target.
    score_shown REAL    NOT NULL,
    -- The value this vote argues for, derived from direction and score_shown.
    target      REAL    NOT NULL,
    voter       TEXT    NOT NULL,
    created_at  TEXT    NOT NULL,
    -- One standing vote per person per axis. Voting again replaces it, the way
    -- changing your mind on Reddit replaces rather than stacks.
    UNIQUE (show_id, axis, voter)
);

CREATE INDEX IF NOT EXISTS idx_votes_show_axis ON votes (show_id, axis);

-- Rolling aggregate per (show, axis). Kept alongside the raw votes rather than
-- replacing them: the raw votes are the research data, this is only a cache so
-- rendering a page is one indexed lookup rather than a scan.
CREATE TABLE IF NOT EXISTS axis_state (
    show_id     INTEGER NOT NULL,
    axis        TEXT    NOT NULL,
    n_up        INTEGER NOT NULL DEFAULT 0,
    n_down      INTEGER NOT NULL DEFAULT 0,
    n_neutral   INTEGER NOT NULL DEFAULT 0,
    sum_targets REAL    NOT NULL DEFAULT 0.0,
    updated_at  TEXT,
    PRIMARY KEY (show_id, axis)
);
"""


def connect():
    """One connection. check_same_thread=False because FastAPI serves from a pool."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # WAL lets reads continue during a write, which matters as soon as more than
    # one person is voting.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    return conn


def target_for(direction, score_shown):
    """
    Turn a vote into the value it argues for.

    Clamped to [0, 1] because an up-vote on something already at 1.0 cannot ask
    for more than the axis allows, and an unclamped target would drag the mean
    outside the valid range.
    """
    if direction == "up":
        return min(1.0, score_shown + VOTE_STEP)
    if direction == "down":
        return max(0.0, score_shown - VOTE_STEP)
    return score_shown  # neutral: "this value is already right"


def fuse(prior, n_votes, sum_targets, kappa=PRIOR_STRENGTH):
    """
    Combine the model's prediction with the crowd's opinion.

        posterior = (kappa * prior + sum_targets) / (kappa + n)

    This is a conjugate-style update of a mean with a pseudo-count prior. Three
    properties matter and all three fall out of the formula:

      n = 0        -> posterior == prior, so no votes means the model's answer
      n small      -> a small shift, proportional to how much evidence exists
      n >> kappa   -> converges on the crowd's mean

    It never produces the naive-ratio result where 3 up and 0 down reads as 1.0.
    """
    if n_votes <= 0:
        return prior
    return (kappa * prior + sum_targets) / (kappa + n_votes)


class CrowdStore:
    """Vote storage and the fused scores that come out of it."""

    def __init__(self):
        self.conn = connect()

    # ------------------------------------------------------------------ write

    def cast(self, show_id, axis, direction, score_shown, voter):
        """
        Record one vote, replacing this voter's previous vote on this axis.

        The aggregate is recomputed from the raw votes for this (show, axis)
        rather than incremented, because a replaced vote has to have its old
        contribution removed. Recomputing is a handful of rows and removes a
        whole class of drift bug where the cache and the votes disagree.
        """
        if direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}")
        score_shown = max(0.0, min(1.0, float(score_shown)))
        target = target_for(direction, score_shown)
        now = datetime.now(timezone.utc).isoformat()

        with self.conn:
            self.conn.execute(
                """
                INSERT INTO votes (show_id, axis, direction, score_shown, target,
                                   voter, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (show_id, axis, voter) DO UPDATE SET
                    direction   = excluded.direction,
                    score_shown = excluded.score_shown,
                    target      = excluded.target,
                    created_at  = excluded.created_at
                """,
                (show_id, axis, direction, score_shown, target, voter, now),
            )
            self._recompute(show_id, axis, now)

    def _recompute(self, show_id, axis, now):
        row = self.conn.execute(
            """
            SELECT
                SUM(direction = 'up')      AS n_up,
                SUM(direction = 'down')    AS n_down,
                SUM(direction = 'neutral') AS n_neutral,
                COALESCE(SUM(target), 0.0) AS sum_targets
            FROM votes WHERE show_id = ? AND axis = ?
            """,
            (show_id, axis),
        ).fetchone()

        self.conn.execute(
            """
            INSERT INTO axis_state (show_id, axis, n_up, n_down, n_neutral,
                                    sum_targets, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (show_id, axis) DO UPDATE SET
                n_up = excluded.n_up, n_down = excluded.n_down,
                n_neutral = excluded.n_neutral, sum_targets = excluded.sum_targets,
                updated_at = excluded.updated_at
            """,
            (show_id, axis, row["n_up"] or 0, row["n_down"] or 0,
             row["n_neutral"] or 0, row["sum_targets"], now),
        )

    # ------------------------------------------------------------------- read

    def state_for_show(self, show_id, priors, voter=None):
        """
        Fused scores for every axis of one show, plus the vote tallies.

        `priors` is the model's prediction per axis. Axes with no votes come
        back with the prior unchanged and n_votes 0, which is what lets the
        interface render identically before and after the first vote.
        """
        rows = {
            r["axis"]: r
            for r in self.conn.execute(
                "SELECT * FROM axis_state WHERE show_id = ?", (show_id,)
            )
        }

        mine = {}
        if voter:
            mine = {
                r["axis"]: r["direction"]
                for r in self.conn.execute(
                    "SELECT axis, direction FROM votes WHERE show_id = ? AND voter = ?",
                    (show_id, voter),
                )
            }

        out = {}
        for axis, prior in priors.items():
            row = rows.get(axis)
            n_up = row["n_up"] if row else 0
            n_down = row["n_down"] if row else 0
            n_neutral = row["n_neutral"] if row else 0
            n = n_up + n_down + n_neutral
            sum_targets = row["sum_targets"] if row else 0.0

            posterior = fuse(prior, n, sum_targets)
            out[axis] = {
                "prior": round(prior, 3),
                "score": round(posterior, 3),
                "n_up": n_up,
                "n_down": n_down,
                "n_neutral": n_neutral,
                "n_votes": n,
                # How far the crowd has moved the model. This is the labelled
                # error data of S6.3 - the axes with the largest drift are the
                # ones the model gets most wrong.
                "drift": round(posterior - prior, 3),
                "your_vote": mine.get(axis),
            }
        return out

    def disagreements(self, limit=40, min_votes=3):
        """
        Where the crowd most disagrees with the model.

        Report S6.3 calls these labelled error data. Ranked by absolute drift,
        with a minimum vote count so a single contrarian does not top the table.
        """
        return [
            dict(r)
            for r in self.conn.execute(
                """
                SELECT show_id, axis, n_up, n_down, n_neutral, sum_targets,
                       (n_up + n_down + n_neutral) AS n_votes
                FROM axis_state
                WHERE (n_up + n_down + n_neutral) >= ?
                ORDER BY n_votes DESC
                LIMIT ?
                """,
                (min_votes, limit),
            )
        ]

    def totals(self):
        row = self.conn.execute(
            """
            SELECT COUNT(*) AS votes,
                   COUNT(DISTINCT voter) AS voters,
                   COUNT(DISTINCT show_id) AS shows
            FROM votes
            """
        ).fetchone()
        return dict(row)
