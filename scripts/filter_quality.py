#!/usr/bin/env python3
"""Quality filter for extracted open medical questions.

Removes low-quality entries from the extracted dataset and regenerates
all export files (JSONL, CSV, MCP benchmark, statistics).

Usage:
    python filter_quality.py [--input FILE] [--dry-run]

Filtering pipeline (applied sequentially):

  1. SHORT QUESTIONS (<30 chars)
     Extraction failures producing sentence fragments.

  2. FDA ADVERSE EVENT (source_id starts with 'fda_')
     All 106 entries are formulaic auto-generated questions from OpenFDA
     adverse event profiles. Not real research questions.

  3. KEGG PATHWAY TEMPLATES (source_id starts with 'hsa')
     84 of 133 KEGG questions used 3 mechanical templates across pathways:
       - "rate-limiting or bottleneck nodes in the X pathway"
       - "response heterogeneity to drugs targeting X pathways"
       - "functional interactions between X pathway and other axes"

  4. NCT LOW QUALITY (source_id starts with 'NCT')
     a) Difficulty ≤ 2: trial endpoint questions, not open research questions.
     b) Pasted protocol text ("The investigators plan to conduct...").

  5. STACKEXCHANGE (URL contains 'stackexchange')
     Casual user Q&A, not research-level open problems.

  6. NON-QUESTION FRAGMENTS (no '?' and length < 80 chars)
     Conclusion sentences grabbed from abstracts instead of questions.

  7. MISSING QUESTION MARK (no '?' at all)
     1,017 pasted abstract fragments lacking question formulation.

  8. "HOW CAN WE BETTER UNDERSTAND" PREFIX
     31 arXiv extraction failures wrapping random abstract text in a generic
     prefix. Topics include lithium batteries, galaxy clusters, quasicrystals.

  9. KEGG D1 FRAGMENTS
     Remaining KEGG description fragments with difficulty=1.

 10. META-QUESTIONS & LAZY EXTRACTION
     Questions where the LLM wrapped a paper title into a generic template:
       - "What are the key unresolved questions regarding [PAPER TITLE]?"
       - "What aspects of [PAPER TITLE] remain unsolved or unclear?"
     Identified by lazy why_open: "Research article addressing medical topic",
     "Literature explicitly identifies unsolved aspects".
     104 entries removed.

 11. COCHRANE REVIEW FRAGMENTS
     Sentences from systematic reviews pasted as questions:
       - "insufficient evidence to reach conclusions regarding adverse effects?"
       - "no evidence that Reiki is either beneficial or harmful?"
       - "unclear or high risk of bias in most of the domains assessed?"

 12. GENERIC TEMPLATE QUESTIONS
     Extraction failures producing vague, content-free questions:
       - "What treatments are needed to address for patients and families?"
       - "Why treatment resistance a leading?"
       - "What remains remains in the field?"

 13. LOWERCASE START (fragments)
     Entries starting with lowercase — all are sentence fragments from
     abstracts, e.g. "hallmark of EPEC/EHEC infections is induction of..."

 14. TITLE-IN-QUESTION (ends with '.' before '?')
     Questions containing the verbatim paper title, detectable by a period
     preceding the question mark.

 15. GARBLED PATTERNS
     Remaining broken patterns: "In the context of [title]", "What are the
     barriers to effective [specialty] management: [abstract text]".

Summary (from 4,271 raw extraction):
    4,271 → 2,023 (2,248 removed, 53% reduction)
    96% of remaining questions are difficulty 3-5.
    All entries have '?' and start with uppercase.
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

LAZY_WHY_STARTERS = [
    "Research article",
    "Literature explicitly",
    "Explicitly identified",
    "Listed as unsolved",
]

META_PATTERNS = [
    "what are the key unresolved questions regarding",
    "what aspects of",
    "what are the unresolved questions",
    "what are the open questions",
    "remain unsolved or unclear",
    "this manuscript provides",
]

COCHRANE_FRAGMENTS = [
    "insufficient evidence",
    "no evidence that",
    "no evidence on the potential",
    "uncertain whether",
    "unclear or high",
    "findings of this review",
    "need to be evaluated in further",
    "scientific study design was poor",
    "quality of studies evaluating",
    "there is evidence that topical",
    "to assess the evidence for",
    "to evaluate acupuncture",
    "to examine the efficacy",
    "to assess the effects of",
]

GENERIC_TEMPLATES = [
    "what treatments are needed to address",
    "why significant work to identify",
    "why treatment resistance a leading",
    "why outcomes are generally poor",
    "why although",
    "why the treatment of",
    "what remains remains",
    "what are the main challenges in this area",
    "what needs to be resolved",
    "can this intervention prevent disease",
    "how can we improve diagnostic accuracy and speed for this condition",
    "what are the comparative safety and efficacy profiles of the treatment groups",
    "is the intervention safe and efficacious in the target population",
    "what are the specific mechanisms of action and biological pathways involved",
    "what is the diagnostic accuracy and clinical utility of the diagnostic approach",
    "what is the optimal treatment protocol, dosage, or intervention approach",
    "what is the true effectiveness and is it clinically meaningful",
    "which approach is superior for patient outcomes",
    "are short-term benefits sustained over months or years",
    "what is the frequency, severity, and clinical significance of harms",
    "what dosing regimen and treatment duration produce best outcomes",
    "what are the biological or physiological pathways involved",
    "what is the complete safety profile and adverse event incidence",
    "which intervention or approach is most effective",
    "what is the optimal dose, intensity, duration, and frequency",
    "what are standardized outcome measures for assessing effectiveness",
    "what are the long-term clinical outcomes and durability",
    "which patient populations benefit most from available interventions",
    "what is the impact on patient quality of life and functional outcomes",
    "what are the biological and pharmacological mechanisms of action",
    "can oral stimulation interventions shorten hospital stays",
    "can exercise interventions enhance quality of life outcomes",
]

GARBLE_PATTERNS = [
    "in the context of",
    "what are the barriers to effective",
    "what is the clinical effectiveness and efficacy of treatments for",
    "cochrane reviews",
    "an overview of cochrane",
]


def classify_removal(q: dict) -> str | None:
    """Return removal reason or None if the question should be kept."""
    sid = q.get("source_id", "")
    url = q.get("source_url", "")
    text = q.get("self_contained_question", "")
    text_lower = text.lower()
    why = q.get("why_open", "")
    difficulty = q.get("difficulty", 0)

    # 1. Short garbage
    if len(text) < 30:
        return "short (<30 chars)"

    # 2. FDA adverse event
    if sid.startswith("fda_"):
        return "FDA adverse event"

    # 3. KEGG template patterns
    if sid.startswith("hsa"):
        if any(t in text_lower for t in KEGG_TEMPLATES):
            return "KEGG template"
        if difficulty == 1:
            return "KEGG D1 fragment"

    # 4. NCT low quality
    if sid.startswith("NCT"):
        if difficulty <= 2:
            return "NCT difficulty ≤ 2"
        if any(t in text_lower for t in NCT_PROTOCOL_PHRASES):
            return "NCT protocol text"

    # 5. StackExchange
    if "stackexchange" in url:
        return "StackExchange"

    # 6. Non-question fragment (no '?' and short)
    if "?" not in text and len(text) < 80:
        return "non-question fragment"

    # 7. Missing question mark entirely
    if "?" not in text:
        return "missing question mark"

    # 8. "How can we better understand" prefix
    if text.startswith("How can we better understand"):
        return "HCWBU prefix"

    # 9. Meta-questions & lazy extraction
    if any(why.startswith(s) for s in LAZY_WHY_STARTERS):
        return "lazy extraction"
    if any(p in text_lower for p in META_PATTERNS):
        return "meta-question"

    # 10. Cochrane review fragments
    if any(p in text_lower for p in COCHRANE_FRAGMENTS):
        return "Cochrane fragment"

    # 11. Generic template questions
    if any(p in text_lower for p in GENERIC_TEMPLATES):
        return "generic template"

    # 12. Lowercase start (fragment)
    if text[0].islower():
        return "lowercase fragment"

    # 13. Title-in-question (ends with '.' before '?')
    if text.rstrip("?").rstrip().endswith("."):
        return "title-in-question"

    # 14. Garbled patterns
    if any(text_lower.startswith(p) or p in text_lower for p in GARBLE_PATTERNS):
        return "garbled pattern"

    # 15. Very short + broken grammar
    if len(text) < 60:
        words = text.split()
        if len(words) < 6 or text.count("?") > 1:
            return "short broken"

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
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--dry-run", action="store_true",
                        help="Show removal stats without modifying files")
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
    has_q = all("?" in q["self_contained_question"] for q in kept)
    all_upper = all(q["self_contained_question"][0].isupper() for q in kept)
    print(f"\nFinal: {len(kept)} questions, {len(domains)} domains")
    print(f"Difficulty: " + ", ".join(f"L{d}:{diff[d]}" for d in sorted(diff)))
    print(f"All have '?': {has_q} | All uppercase start: {all_upper}")


if __name__ == "__main__":
    main()
