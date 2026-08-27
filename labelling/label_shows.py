"""
Label shows against the feature schema using Claude, producing training data.

This is stage 1 of report S6.2. The LLM is a labelling tool here - the same job
paid annotators would do. It is not the deliverable. The model trained on these
labels (training/train_model.py) is the deliverable, because it is reproducible,
free to run, and scores a show the LLM has never heard of.

Run:
    python labelling/label_shows.py --limit 20        # start small
    python labelling/label_shows.py                   # whole catalogue

Requires ANTHROPIC_API_KEY in .env.

Output: labelling/labels.jsonl - one JSON object per show, appended as it goes.
Re-running skips shows already present, so an interrupted run resumes.
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import anthropic
from dotenv import load_dotenv
from pydantic import create_model

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from labelling.schema import AXES, AXIS_NAMES, definitions_block  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

MODEL = "claude-opus-5"

# How many episode overviews to show the labeller. The whole run would be tens
# of thousands of words for a long series; a spread sample across the whole run
# captures how the show changes without paying for every episode.
EPISODE_SAMPLE = 12

# Pricing per million tokens, for the cost estimate only (Claude Opus 5).
INPUT_PER_MTOK = 5.00
OUTPUT_PER_MTOK = 25.00


# Built from the schema so the two can never drift apart. Every axis becomes a
# float field constrained to 0-1, which the API enforces rather than us
# validating after the fact.
ShowLabels = create_model(
    "ShowLabels",
    **{name: (float, ...) for name in AXIS_NAMES},
    evidence_quality=(float, ...),
    notes=(str, ...),
)

SYSTEM_PROMPT = f"""You are annotating television shows for a research dataset that
describes what watching a show is like.

Score the show on every axis from 0.0 to 1.0:

  0.0   absent - the word does not apply
  0.25  present in one or two moments, incidental
  0.5   a recurring element, but not what the show is about
  0.75  a defining characteristic, present in most episodes
  1.0   the show's central organising principle

Rules that matter:

1. These are DESCRIPTIVE judgements, not evaluative ones. The question is
   always "how much does this word describe the show?", never "is the show
   good?". A brilliant show and a terrible show can both score 0.9 on `bleak`.

2. Score the show as a whole, averaged across its run - not its best episode
   and not its first season.

3. Base your scores on the supplied text. Where the text is thin, say so via
   evidence_quality rather than inventing detail. Do not let general reputation
   override what the text actually shows.

4. Axes are independent. A show can be high on both `comedy` and `bleak`.
   Do not spread a fixed budget across axes.

Also return:
  evidence_quality - 0.0 to 1.0, how much the supplied text actually supported
                     these judgements. Low when you had little to work with.
  notes            - one short sentence on anything ambiguous.

The axes:
{definitions_block()}"""


def build_show_text(show, episodes_by_show):
    """Assemble everything the labeller reads about one show."""
    parts = [
        f"TITLE: {show.get('name', '')} ({(show.get('first_air_date') or '')[:4]})",
        f"TMDB GENRES: {', '.join(g['name'] for g in show.get('genres', [])) or 'none listed'}",
        f"EPISODES: {show.get('number_of_episodes')} across {show.get('number_of_seasons')} seasons",
        f"\nOVERVIEW:\n{show.get('overview') or '(none)'}",
    ]

    keywords = [k["name"] for k in show.get("keywords", {}).get("results", [])]
    if keywords:
        parts.append(f"\nTMDB KEYWORDS: {', '.join(keywords)}")

    # Episode overviews are the richest text available - roughly 1.66M words
    # across the catalogue, against 22k words of series overviews.
    episodes = episodes_by_show.get(show["id"], [])
    with_text = [e for e in episodes if e.get("overview")]
    if with_text:
        step = max(1, len(with_text) // EPISODE_SAMPLE)
        sample = with_text[::step][:EPISODE_SAMPLE]
        lines = [
            f"  S{e['season_number']}E{e['episode_number']} {e['name']}: {e['overview']}"
            for e in sample
        ]
        parts.append(
            f"\nEPISODE SUMMARIES ({len(sample)} sampled across {len(with_text)}):\n"
            + "\n".join(lines)
        )

    reviews = show.get("reviews", {}).get("results", [])
    if reviews:
        joined = "\n\n".join(r["content"][:1500] for r in reviews[:4])
        parts.append(f"\nVIEWER REVIEWS:\n{joined}")

    return "\n".join(parts)


def label_one(client, show, episodes_by_show):
    """One API call. Returns (record, usage) or raises."""
    response = client.messages.parse(
        model=MODEL,
        max_tokens=4000,
        system=[{
            "type": "text",
            "text": SYSTEM_PROMPT,
            # The schema is identical on every one of the 500 calls, so caching
            # it turns ~1,500 repeated input tokens into a cache read at ~10%
            # of the price.
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{
            "role": "user",
            "content": build_show_text(show, episodes_by_show),
        }],
        output_format=ShowLabels,
        thinking={"type": "adaptive"},
        # Labelling is a judgement task run hundreds of times. Low effort keeps
        # cost sane; raise it if agreement against the hand-scored set is poor.
        output_config={"effort": "low"},
    )

    labels = response.parsed_output
    record = {
        "id": show["id"],
        "name": show.get("name", ""),
        "labels": {name: getattr(labels, name) for name in AXIS_NAMES},
        "evidence_quality": labels.evidence_quality,
        "notes": labels.notes,
    }
    return record, response.usage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="label only the first N unlabelled shows")
    parser.add_argument("--workers", type=int, default=4, help="concurrent requests")
    parser.add_argument("--out", default="labels.jsonl")
    args = parser.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "No ANTHROPIC_API_KEY found.\n"
            "Add it to .env as ANTHROPIC_API_KEY=sk-ant-..., then re-run."
        )

    shows = json.loads((ROOT / "tmdb" / "shows_raw.json").read_text(encoding="utf-8"))

    episodes_by_show = {}
    episodes_path = ROOT / "tmdb" / "episodes.json"
    if episodes_path.exists():
        for episode in json.loads(episodes_path.read_text(encoding="utf-8")):
            episodes_by_show.setdefault(episode["show_id"], []).append(episode)

    # Resume: anything already in the output file is skipped.
    out_path = HERE / args.out
    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])

    todo = [s for s in shows if s["id"] not in done]
    if args.limit:
        todo = todo[:args.limit]

    print(f"{len(shows)} shows, {len(done)} already labelled, {len(todo)} to do")
    if not todo:
        return

    client = anthropic.Anthropic()
    write_lock = Lock()
    totals = {"in": 0, "out": 0, "cache_read": 0, "cache_write": 0, "failed": 0}

    with open(out_path, "a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(label_one, client, show, episodes_by_show): show
                for show in todo
            }

            for count, future in enumerate(as_completed(futures), start=1):
                show = futures[future]
                try:
                    record, usage = future.result()
                except Exception as error:
                    totals["failed"] += 1
                    print(f"  FAILED {show.get('name')}: {type(error).__name__}: {error}")
                    continue

                # Serialise writes - several threads finish at once and would
                # otherwise interleave partial lines.
                with write_lock:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()

                totals["in"] += usage.input_tokens
                totals["out"] += usage.output_tokens
                totals["cache_read"] += getattr(usage, "cache_read_input_tokens", 0) or 0
                totals["cache_write"] += getattr(usage, "cache_creation_input_tokens", 0) or 0

                if count % 10 == 0 or count == len(todo):
                    spent = (
                        totals["in"] / 1e6 * INPUT_PER_MTOK
                        + totals["out"] / 1e6 * OUTPUT_PER_MTOK
                        + totals["cache_read"] / 1e6 * INPUT_PER_MTOK * 0.1
                        + totals["cache_write"] / 1e6 * INPUT_PER_MTOK * 1.25
                    )
                    print(f"  {count}/{len(todo)}  ~${spent:.2f} so far")

    print(f"\nWrote to {out_path.name}. Failed: {totals['failed']}")
    print(f"Tokens - input {totals['in']:,}, output {totals['out']:,}, "
          f"cache read {totals['cache_read']:,}, cache write {totals['cache_write']:,}")
    if totals["cache_read"]:
        saved = totals["cache_read"] / 1e6 * INPUT_PER_MTOK * 0.9
        print(f"Prompt caching saved roughly ${saved:.2f}")


if __name__ == "__main__":
    main()
