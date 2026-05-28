#!/usr/bin/env python3
"""Quality filter for extracted open medical questions.

Removes low-quality entries from the extracted dataset and regenerates
all export files (JSONL, CSV, MCP benchmark, statistics).

Usage:
    python filter_quality.py [--input FILE] [--dry-run]

Filtering pipeline (applied sequentially):

  1. SHORT QUESTIONS (<30 chars)
     Extraction failures that produced sentence fragments instead of
     proper questions. Examples: "unclear?", "Grand Challenge Project.",
     "The certainty of evidence".

  2. FDA ADVERSE EVENT (source_id starts with 'fda_')
     Auto-generated from OpenFDA adverse event profiles. All 106 entries
     follow a formulaic pattern ("Why do X drugs cause Y adverse effects?")
     and lack genuine research context. Not real open research questions.

  3. KEGG PATHWAY TEMPLATES (source_id starts with 'hsa')
     KEGG pathway entries produced 3 repeated template patterns across
     all disease pathways. 84 of 133 KEGG questions were identical
     templates applied mechanically to different pathways:
       - "rate-limiting or bottleneck nodes in the X pathway"
       - "response heterogeneity to drugs targeting X pathways"
       - "functional interactions between X pathway and other axes"
     The remaining 49 non-template KEGG questions are kept.

  4. NCT LOW QUALITY (source_id starts with 'NCT')
     Clinical trial entries have two quality issues:
     a) Difficulty ≤ 2: Trial endpoint questions like "What is the safety
        and efficacy of Restylane Defyne for chin augmentation?" — these
        are trial objectives, not unsolved research questions. (143 removed)
     b) Pasted protocol text: Raw trial descriptions copied verbatim,
        e.g. "The investigators plan to conduct...", "subjects will be
        recruited..." — not questions at all. (2 removed)
     Remaining 200 NCT entries (difficulty ≥ 3, proper question format)
     are kept — these tend to ask about novel mechanisms or treatment-
     resistant conditions.

  5. STACKEXCHANGE (URL contains 'stackexchange')
     412 entries from medicalsciences.stackexchange.com. These are casual
     user-generated Q&A, not research-level open problems:
       - "Why does the BRAT diet recommend toast instead of plain bread?"
       - "How can one choose the proper pressure setting on a water flosser?"
       - "Could rectal prolapse and bloody masturbation in 4 year-old happen absent abuse?"
     Even high-difficulty-rated ones are often raw post text ("A Google
     search for... returned no relevant answers") or personal medical
     questions. Not suitable for a research open-questions dataset.

  6. NON-QUESTION FRAGMENTS (no '?' and length < 80 chars)
     Extraction failures that grabbed conclusion sentences from abstracts
     instead of reformulating into questions:
       - "However, the etiology remains unclear"
       - "There is no cure, and its prevalence will double by 2030"
       - "grand challenge to academic medicine: speak out on gay rights"
     144 entries caught by this filter. True open problems behind these
     statements are captured from better sources elsewhere in the dataset.

  7. MISSING QUESTION MARK (no '?' in self_contained_question)
     After filters 1-6, 1,017 remaining entries lack a question mark.
     Manual inspection of 200+ samples found zero properly formulated
     questions — all are pasted abstract fragments:
       - "Huntington's disease has no cure and patients rely only in
          symptomatic treatment."
       - "More research is warranted, mainly focusing on personalizing
          current treatments to optimize response and remission rates"
       - "Complete miscarriage Based on the relative effects from the
          network meta-analysis of 59 trials..."
     These are NOT self-contained questions as required by the schema.

Summary of total filtering:
    Before: 4,271  →  After: 2,345  (1,926 removed, 45% reduction)

    Removal breakdown:
        20   short (<30 chars)
       106   FDA adverse event
        84   KEGG template
       145   NCT low quality
       410   StackExchange
       144   non-question fragment (<80 chars, no ?)
     1,017   missing question mark
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

BASE_DIR = Path("/data/project/private/minstar/workspace/healthcare-research")
DEFAULT_INPUT = BASE_DIR / "data" / "extracted" / "all_questions_final.jsonl"
EXPORT_DIR = BASE_DIR / "data" / "export"

KEGG_TEMPLATES = [
    "rate-limiting or bottleneck",
    "response heterogeneity to drugs targeting",
    "functional interactions between",
]

NCT_PROTOCOL_PHRASES = [
    "the investigators plan",
    "subjects will be recruited",
    "the study will take place",
    "this study is",
    "this trial is",
]


def classify_removal(q: dict) -> str | None:
    """Return removal reason or None if the question should be kept."""
    sid = q.get("source_id", "")
    url = q.get("source_url", "")
    text = q.get("self_contained_question", "")
    text_lower = text.lower()
    difficulty = q.get("difficulty", 0)

    if len(text) < 30:
        return "short (<30 chars)"

    if sid.startswith("fda_"):
        return "FDA adverse event"

    if sid.startswith("hsa"):
        if any(t in text_lower for t in KEGG_TEMPLATES):
            return "KEGG template"

    if sid.startswith("NCT"):
        if difficulty <= 2:
            return "NCT difficulty ≤ 2"
        if any(t in text_lower for t in NCT_PROTOCOL_PHRASES):
            return "NCT protocol text"

    if "stackexchange" in url:
        return "StackExchange"

    if "?" not in text and len(text) < 80:
        return "non-question fragment"

    if "?" not in text:
        return "missing question mark"

    return None


def regenerate_exports(questions: list[dict]) -> None:
    """Regenerate statistics.json, mcp_benchmark.jsonl, and CSV."""
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    qtypes = Counter(q["question_type"] for q in questions)
    domains = Counter(q["clinical_domain"] for q in questions)
    difficulty_dist = Counter(q.get("difficulty", 0) for q in questions)

    stats = {
        "total_questions": len(questions),
        "question_types": dict(qtypes.most_common()),
        "top_30_clinical_domains": dict(domains.most_common(30)),
        "difficulty_distribution": {
            f"level_{k}": v for k, v in sorted(difficulty_dist.items())
        },
        "unique_clinical_domains": len(domains),
    }
    with open(EXPORT_DIR / "statistics.json", "w") as f:
        json.dump(stats, f, indent=4, ensure_ascii=False)

    with open(EXPORT_DIR / "mcp_benchmark.jsonl", "w") as f:
        for i, q in enumerate(questions):
            entry = {
                "id": f"med_open_q_{i+1:05d}",
                "question": q["self_contained_question"],
                "source": {
                    "source_id": q.get("source_id", ""),
                    "url": q.get("source_url", ""),
                    "title": q.get("source_title", ""),
                },
                "metadata": {
                    "original_question": q.get("original_question", ""),
                    "question_type": q.get("question_type", ""),
                    "clinical_domain": q.get("clinical_domain", ""),
                    "why_open": q.get("why_open", ""),
                    "difficulty": q.get("difficulty", 0),
                },
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    fields = [
        "source_id", "source_url", "source_title", "original_question",
        "self_contained_question", "question_type", "clinical_domain",
        "why_open", "difficulty",
    ]
    with open(EXPORT_DIR / "open_medical_questions.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for q in questions:
            writer.writerow({k: q.get(k, "") for k in fields})


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be removed without modifying files")
    args = parser.parse_args()

    questions = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    kept = []
    removed = Counter()
    removed_samples: dict[str, list[str]] = {}

    for q in questions:
        reason = classify_removal(q)
        if reason:
            removed[reason] += 1
            if reason not in removed_samples:
                removed_samples[reason] = []
            if len(removed_samples[reason]) < 3:
                removed_samples[reason].append(
                    q["self_contained_question"][:120]
                )
        else:
            kept.append(q)

    total_removed = sum(removed.values())
    print(f"{'='*60}")
    print(f"Quality Filter Results")
    print(f"{'='*60}")
    print(f"Input:   {len(questions):,} questions")
    print(f"Removed: {total_removed:,} ({total_removed/len(questions)*100:.1f}%)")
    print(f"Kept:    {len(kept):,} ({len(kept)/len(questions)*100:.1f}%)")
    print()
    print(f"{'Reason':<30} {'Count':>6}")
    print(f"{'-'*36}")
    for reason, count in removed.most_common():
        print(f"  {reason:<28} {count:>6}")
        if reason in removed_samples:
            for s in removed_samples[reason]:
                print(f"    e.g. \"{s}...\"")
    print()

    if args.dry_run:
        print("[DRY RUN] No files modified.")
        return

    with open(args.input, "w") as f:
        for q in kept:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"Wrote {len(kept):,} questions to {args.input}")

    regenerate_exports(kept)
    print(f"Regenerated exports in {EXPORT_DIR}")

    domains = Counter(q["clinical_domain"] for q in kept)
    diff = Counter(q.get("difficulty", 0) for q in kept)
    print(f"\nFinal stats: {len(kept)} questions, {len(domains)} domains")
    print(f"Difficulty: " + ", ".join(f"L{d}:{diff[d]}" for d in sorted(diff)))


if __name__ == "__main__":
    main()
