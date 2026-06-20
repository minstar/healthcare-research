#!/usr/bin/env python3
"""Merge all gold answer batch results into a single benchmark export file.

Usage: python scripts/track_b/merge_gold_answers.py [--skip-validation] [--fix-citations]

Reads:  data/gold_answers/chunk_*.input.jsonl  (questions)
        data/gold_answers/chunk_*.output.jsonl (gold answers, top-level fields)
Writes: data/export/mcp_benchmark_with_gold.jsonl
        data/export/validation_report.json (from post-merge validation)

Pipeline: merge → validate citations (PMID/NCT) → check completeness → report
"""
import glob
import json
import os
import subprocess
import sys

BASE_DIR = "/data/project/private/minstar/workspace/healthcare-research"
GOLD_DIR = os.path.join(BASE_DIR, "data/gold_answers")
EXPORT_DIR = os.path.join(BASE_DIR, "data/export")
OUTPUT_FILE = os.path.join(EXPORT_DIR, "mcp_benchmark_with_gold.jsonl")
TOTAL_EXPECTED = 1969

GOLD_FIELDS = [
    "current_knowledge",
    "unknown_aspects",
    "evidence_landscape",
    "key_citations",
    "mcp_tool_plan",
    "answer_summary",
    "self_completeness",
]


def run_validation(fix: bool = False):
    """Run post-merge validation: PMID/NCT verification + completeness checks."""
    validate_script = os.path.join(BASE_DIR, "scripts/track_b/validate_gold_answers.py")
    report_file = os.path.join(EXPORT_DIR, "validation_report.json")

    cmd = [sys.executable, validate_script, "--input", OUTPUT_FILE, "--report", report_file]
    if fix:
        cmd.append("--fix")

    print(f"\n{'='*60}")
    print("Running post-merge validation...")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def load_chunk_pair(input_file: str, output_file: str):
    """Load input questions and output gold answers, merge by line index."""
    questions = []
    with open(input_file) as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    gold_answers = []
    with open(output_file) as f:
        for line in f:
            line = line.strip()
            if line:
                gold_answers.append(json.loads(line))

    merged = []
    for i, q in enumerate(questions):
        gold = {}
        if i < len(gold_answers):
            ga = gold_answers[i]
            # Handle both formats: top-level fields or nested under "gold_answer"
            source = ga.get("gold_answer", ga) if isinstance(ga.get("gold_answer"), dict) else ga
            for field in GOLD_FIELDS:
                if field in source:
                    gold[field] = source[field]

        doc = dict(q)
        doc["gold_answer"] = gold
        merged.append(doc)

    return merged


def main():
    skip_validation = "--skip-validation" in sys.argv
    fix_citations = "--fix-citations" in sys.argv

    input_files = sorted(glob.glob(os.path.join(GOLD_DIR, "chunk_*.input.jsonl")))
    if not input_files:
        print(f"No chunk input files found in {GOLD_DIR}", file=sys.stderr)
        sys.exit(1)

    all_questions = []
    has_gold = 0
    empty_gold = 0
    missing_fields = {}
    chunks_processed = 0
    chunks_missing_output = []

    for input_file in input_files:
        output_file = input_file.replace(".input.jsonl", ".output.jsonl")
        if not os.path.exists(output_file):
            chunks_missing_output.append(os.path.basename(input_file))
            with open(input_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        doc = json.loads(line)
                        doc["gold_answer"] = {}
                        all_questions.append(doc)
                        empty_gold += 1
            continue

        merged = load_chunk_pair(input_file, output_file)
        chunks_processed += 1

        for doc in merged:
            all_questions.append(doc)
            gold = doc.get("gold_answer", {})
            if gold and gold.get("current_knowledge"):
                has_gold += 1
            else:
                empty_gold += 1

            for field in GOLD_FIELDS:
                val = gold.get(field)
                is_present = bool(val) if not isinstance(val, (int, float)) else val is not None
                if not is_present:
                    missing_fields[field] = missing_fields.get(field, 0) + 1

    # Deduplicate by source_id + question
    seen = set()
    unique_questions = []
    duplicates = 0
    for doc in all_questions:
        key = (
            doc.get("source_id", ""),
            doc.get("self_contained_question", doc.get("original_question", "")),
        )
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        unique_questions.append(doc)

    os.makedirs(EXPORT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        for doc in unique_questions:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print("=" * 60)
    print("Gold Answer Merge Report")
    print("=" * 60)
    print(f"Chunk pairs processed:   {chunks_processed}")
    print(f"Total questions loaded:  {len(all_questions)}")
    print(f"Duplicates removed:      {duplicates}")
    print(f"Unique questions:        {len(unique_questions)}")
    print(f"Expected total:          {TOTAL_EXPECTED}")
    print(f"Coverage:                {len(unique_questions)}/{TOTAL_EXPECTED} ({100*len(unique_questions)/TOTAL_EXPECTED:.1f}%)")
    print()
    print(f"With gold answer:        {has_gold}")
    print(f"Missing/empty gold:      {empty_gold}")
    print(f"Gold answer rate:        {100*has_gold/max(len(all_questions),1):.1f}%")
    print()

    if missing_fields:
        print("Field coverage gaps (missing/empty):")
        for field, count in sorted(missing_fields.items(), key=lambda x: -x[1]):
            print(f"  {field}: {count} missing")
        print()

    if chunks_missing_output:
        print(f"Chunks without output ({len(chunks_missing_output)}):")
        for name in chunks_missing_output:
            print(f"  {name}")
        print()

    # Self-completeness distribution
    completeness_vals = []
    for doc in unique_questions:
        gold = doc.get("gold_answer", {})
        c = gold.get("self_completeness")
        if isinstance(c, (int, float)):
            completeness_vals.append(float(c))
    if completeness_vals:
        avg_c = sum(completeness_vals) / len(completeness_vals)
        print(f"Avg self_completeness:   {avg_c:.2f} (n={len(completeness_vals)})")

    print()
    print(f"Output: {OUTPUT_FILE}")
    print("=" * 60)

    if len(unique_questions) < TOTAL_EXPECTED:
        missing = TOTAL_EXPECTED - len(unique_questions)
        print(
            f"\nWarning: {missing} questions missing. Re-run failed batches.",
            file=sys.stderr,
        )

    if not skip_validation:
        run_validation(fix=fix_citations)
    else:
        print("\n[Validation skipped]")


if __name__ == "__main__":
    main()
