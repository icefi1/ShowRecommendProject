"""
The judging session (report S9.2) as a page, with posters.

Same pool, same judgements file and same rules as `judge.py` - this is the
interface, not a second experiment. Recognising a show from its title alone is
the slowest part of judging, and a poster answers "have I seen this?" instantly,
so this is the version worth actually using. The terminal version stays for
machines where opening a browser is inconvenient.

It runs its own little server rather than joining app/main.py on purpose: the
recommender is the thing being measured, and measurement apparatus does not
belong inside it.

Run:
    venv\\Scripts\\python evaluation/judge_web.py

then open http://127.0.0.1:8020 and put your name in.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from app.similarity import FeatureSpace  # noqa: E402
from evaluation.judge import (  # noqa: E402
    JUDGEMENTS_FILE,
    VERDICTS,
    load_judgements,
    load_pool,
    record,
)

PORT = 8020

app = FastAPI(title="Relevance judging")
space = FeatureSpace()
by_id = {show["id"]: show for show in space.catalogue}


def show_card(show_id):
    """Everything the page needs to draw one show."""
    show = by_id[show_id]
    return {
        "id": show["id"],
        "name": show["name"],
        "year": show.get("year"),
        "genres": show.get("genres") or [],
        "episodes": show.get("episodes"),
        "certificate": show.get("certificate"),
        "rating": show.get("rating"),
        "overview": show.get("overview") or "",
        # Same image source the recommender itself uses, one size up: these are
        # being looked at rather than skimmed in a results list.
        "poster": f"https://image.tmdb.org/t/p/w185{show['poster']}" if show.get("poster") else None,
    }


def pairs_in_pool():
    pool = load_pool()
    if pool is None:
        raise HTTPException(
            status_code=503,
            detail="No judging pool yet. Run: python evaluation/judge.py --build-pool",
        )
    return [(e["query_id"], c) for e in pool["entries"] for c in e["candidates"]]


@app.get("/api/next")
def next_pair(judge: str, revisit: bool = False):
    """
    The next pair this judge has not answered, plus how far through they are.

    `revisit` serves the pairs they marked "don't know it" instead. The first
    terminal session came back 86% unfamiliar, because a title alone is not
    enough to recognise a show - with a poster in front of them many of those
    become answerable, and a second pass is far cheaper than a second pool.
    """
    judge = judge.strip()
    if not judge:
        raise HTTPException(status_code=400, detail="Tell me who is judging.")

    pairs = pairs_in_pool()
    verdicts = {(j["query_id"], j["candidate_id"]): j["verdict"]
                for j in load_judgements() if j["judge"] == judge}

    fresh = [p for p in pairs if p not in verdicts]
    unknown = [p for p in pairs if verdicts.get(p) == "unfamiliar"]
    queue = unknown if revisit else fresh

    progress = {
        "done": len(verdicts), "total": len(pairs),
        "fresh_left": len(fresh), "unknown": len(unknown),
    }
    if not queue:
        return {"finished": True, "progress": progress, "mode": "revisit" if revisit else "fresh"}

    query_id, candidate_id = queue[0]
    return {
        "finished": False,
        "mode": "revisit" if revisit else "fresh",
        "progress": progress,
        "query": show_card(query_id),
        "candidate": show_card(candidate_id),
    }


class Verdict(BaseModel):
    judge: str = Field(min_length=1, max_length=40)
    query_id: int
    candidate_id: int
    # The same four answers the terminal version records, no more: the scorer
    # only understands these.
    verdict: str = Field(pattern="^(yes|no|maybe|unfamiliar)$")


@app.post("/api/verdict")
def post_verdict(verdict: Verdict):
    """Append one judgement. Written immediately, so quitting loses nothing."""
    if verdict.verdict not in VERDICTS.values():
        raise HTTPException(status_code=400, detail="Unknown verdict")

    record({
        "judge": verdict.judge.strip(),
        "query_id": verdict.query_id,
        "candidate_id": verdict.candidate_id,
        "verdict": verdict.verdict,
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    return {"ok": True}


@app.post("/api/undo")
def undo(judge: str):
    """
    Take back this judge's most recent answer.

    A misclick is certain somewhere in 331 questions, and an answer nobody can
    correct is worse data than one they can. The file is append-only during a
    session, so undo is the one operation that rewrites it.
    """
    records = load_judgements()
    for position in range(len(records) - 1, -1, -1):
        if records[position]["judge"] == judge.strip():
            removed = records.pop(position)
            with open(JUDGEMENTS_FILE, "w", encoding="utf-8") as handle:
                for row in records:
                    handle.write(json.dumps(row) + "\n")
            return {"ok": True, "removed": removed}
    return {"ok": False, "detail": "Nothing to undo."}


@app.get("/")
def page():
    return FileResponse(HERE / "judge.html")


if __name__ == "__main__":
    if load_pool() is None:
        raise SystemExit(
            "No judging pool yet. Build one first:\n"
            r"  venv\Scripts\python evaluation/judge.py --build-pool"
        )
    print(f"\n  Judging session: http://127.0.0.1:{PORT}\n")
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
