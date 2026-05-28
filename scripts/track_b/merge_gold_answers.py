#!/usr/bin/env python3
"""Merge all gold answer batch results into a single benchmark export file.

Usage: python scripts/track_b/merge_gold_answers.py [--skip-validation] [--fix-citations]

Reads:  data/gold_answers/*.output.jsonl (chunk-based) or batch_NNN.jsonl
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


def run_validation(fix: bool = False):
    """Run post-merge validation: PMID/NCT verification + completeness checks."""
    validate_script = os.path.join(BASE_DIR, "scripts/track_b/validate_gold_answers.py")
    report_file = os.path.join(EXPORT_DIR, "validation_report.json")

    cmd = [sys.executable, validate_script, "--input", GOLD_DIR, "--report", report_file]
    if fix:
        cmd.append("--fix")

    print(f"\n{'='*60}")
    print("Running post-merge validation...")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def main():
    skip_validation = "--skip-validation" in sys.argv
    fix_citations = "--fix-citations" in sys.argv

    batch_files = sorted(glob.glob(os.path.join(GOLD_DIR, "batch_*.jsonl")))

    if not batch_files:
        print(f"No batch files found in {GOLD_DIR}", file=sys.stderr)
        sys.exit(1)

    # Exclude debug files
    batch_files = [f for f in batch_files if not f.endswith(".debug.txt")]

    all_questions = []
    has_gold = 0
    empty_gold = 0
    missing_fields = {}

    for bf in batch_files:
        with open(bf) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    print(f"  Warning: skipping malformed line in {bf}", file=sys.stderr)
                    continue

                all_questions.append(doc)

                gold = doc.get("gold_answer", {})
                if gold and gold.get("current_knowledge"):
                    has_gold += 1
                else:
                    empty_gold += 1

                # Track field coverage
                for field in [
                    "current_knowledge",
                    "unknown_aspects",
                    "evidence_landscape",
                    "key_citations",
                    "mcp_tool_plan",
                    "answer_summary",
                    "self_completeness",
                ]:
                    val = gold.get(field, gold.get("completeness") if field == "self_completeness" else None)
                    is_present = bool(val) if not isinstance(val, (int, float)) else True
                    if not is_present:
                        missing_fields[field] = missing_fields.get(field, 0) + 1

    # Deduplicate by source_id + question (in case of overlapping batches)
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

    # Print statistics
    print("=" * 60)
    print("Gold Answer Merge Report")
    print("=" * 60)
    print(f"Batch files processed:   {len(batch_files)}")
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

    # Self-completeness distribution
    completeness_vals = []
    for doc in unique_questions:
        gold = doc.get("gold_answer", {})
        c = gold.get("self_completeness", gold.get("completeness"))
        if isinstance(c, (int, float)):
            completeness_vals.append(float(c))
    if completeness_vals:
        avg_c = sum(completeness_vals) / len(completeness_vals)
        print(f"Avg self_completeness:   {avg_c:.2f} (n={len(completeness_vals)})")

    print()
    print(f"Output: {OUTPUT_FILE}")
    print("=" * 60)

    # Exit with warning if coverage is incomplete
    if len(unique_questions) < TOTAL_EXPECTED:
        missing = TOTAL_EXPECTED - len(unique_questions)
        print(
            f"\nWarning: {missing} questions missing. Re-run failed batches.",
            file=sys.stderr,
        )

    # Run validation unless skipped
    if not skip_validation:
        run_validation(fix=fix_citations)
    else:
        print("\n[Validation skipped]")


if __name__ == "__main__":
    main()
