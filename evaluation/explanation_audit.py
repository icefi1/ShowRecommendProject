"""
Audit the explanations the similarity engine produces (report S6.6).

An explanation is the project's central contribution, so it needs a measurement
rather than a vibe check. This samples the catalogue, generates explanations for
the top results of each query, and counts the two failure modes that matter:

  1. No shared clause. The explanation says only how the two shows DIFFER, which
     tells the user nothing about why the show was recommended.
  2. Provenance keywords cited ("based on novel or book", "remake"). These say
     where a show came from, not what watching it is like.

Run:  venv\Scripts\python evaluation/explanation_audit.py
"""

import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.similarity import FeatureSpace  # noqa: E402

SAMPLE_SHOWS = 120
RESULTS_PER_SHOW = 5
SEED = 20250831  # fixed so two runs of this script are comparable


def clauses(explanation):
    """Split an explanation into its clauses and label each one by its opener."""
    labelled = []
    for clause in explanation.split("; "):
        if clause.startswith("both "):
            labelled.append(("genre", clause))
        elif clause.startswith("shares "):
            labelled.append(("keywords", clause))
        elif clause.startswith("but "):
            labelled.append(("structure", clause))
        else:
            labelled.append(("other", clause))
    return labelled


def is_provenance(term):
    """Same rule the engine uses, restated here so the audit is independent."""
    return term.startswith("based on") or "remake" in term


def main():
    space = FeatureSpace()
    vocabulary = space.block_labels["keywords"]
    provenance_terms = [t for t in vocabulary if is_provenance(t)]

    random.seed(SEED)
    sample = random.sample(space.catalogue, SAMPLE_SHOWS)

    counts = Counter()
    examples = {"no_shared": [], "provenance": []}
    total = 0

    for show in sample:
        for result in space.similar_to_show(show["id"], limit=RESULTS_PER_SHOW):
            explanation = result["explanation"]
            total += 1
            kinds = {kind for kind, _ in clauses(explanation)}

            for kind in kinds:
                counts[kind] += 1

            if not (kinds & {"genre", "keywords"}):
                counts["no_shared"] += 1
                if len(examples["no_shared"]) < 6:
                    examples["no_shared"].append(f"{show['name']} -> {result['name']}: {explanation}")

            if any(term in explanation for term in provenance_terms):
                counts["provenance"] += 1
                if len(examples["provenance"]) < 6:
                    examples["provenance"].append(f"{show['name']} -> {result['name']}: {explanation}")

    print(f"{total} explanations from {SAMPLE_SHOWS} shows x {RESULTS_PER_SHOW} results\n")
    for label, key in [
        ("genre clause", "genre"),
        ("keyword clause", "keywords"),
        ("structure clause", "structure"),
        ("NO shared clause", "no_shared"),
        ("provenance keyword cited", "provenance"),
    ]:
        print(f"  {label:28} {counts[key]:5}  {counts[key] / total:6.1%}")

    for title, key in [("no shared clause", "no_shared"), ("provenance cited", "provenance")]:
        print(f"\n--- examples: {title} ---")
        for line in examples[key] or ["(none)"]:
            print("  " + line)


if __name__ == "__main__":
    main()
