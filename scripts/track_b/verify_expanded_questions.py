#!/usr/bin/env python3
"""Verify and deduplicate expanded questions against existing gold answer corpus.

Steps:
1. Structural validation (required fields, value ranges)
2. Content quality filters (question length, taxonomy valid, open_status)
3. Embedding-based dedup against existing 1,969 gold answer questions
4. Internal dedup within expanded set
5. Output: verified questions ready for gold answer generation

Usage:
    python scripts/track_b/verify_expanded_questions.py
"""
import json
import sys
from pathlib import Path
from collections import Counter

import numpy as np

BASE_DIR = Path("/data/project/private/minstar/workspace/healthcare-research")
EXPANDED_FILE = BASE_DIR / "data" / "expanded" / "all_new_questions_refined.jsonl"
EXISTING_GOLD = BASE_DIR / "data" / "export" / "mcp_benchmark_with_gold.jsonl"
OUTPUT_FILE = BASE_DIR / "data" / "expanded" / "verified_for_gold.jsonl"

VALID_TAXONOMY_L1 = {
    "Clinical Medicine",
    "Oncology",
    "Neuroscience & Psychiatry",
    "Infectious Disease & Immunology",
    "Cardiovascular Medicine",
    "Genomics & Precision Medicine",
    "Pharmacology & Drug Discovery",
    "Public Health & Epidemiology",
    "Rare & Orphan Diseases",
    "Surgical Sciences",
    "Medical AI & Informatics",
    "Other",
}

VALID_OPEN_STATUS = {"open", "partially_answered"}


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def structural_check(rec: dict, idx: int) -> list[str]:
    """Validate required fields and value ranges."""
    issues = []
    required = ["source_id", "self_contained_question", "taxonomy_l1", "open_status",
                "difficulty_clinical_knowledge", "difficulty_research_depth", "difficulty_multi_step_reasoning"]
    for f in required:
        if not rec.get(f):
            issues.append(f"missing {f}")

    q = rec.get("self_contained_question", "")
    if len(q) < 30:
        issues.append(f"question too short ({len(q)} chars)")
    if len(q) > 1000:
        issues.append(f"question too long ({len(q)} chars)")

    if rec.get("taxonomy_l1") and rec["taxonomy_l1"] not in VALID_TAXONOMY_L1:
        issues.append(f"invalid taxonomy_l1: {rec['taxonomy_l1']}")

    if rec.get("open_status") and rec["open_status"] not in VALID_OPEN_STATUS:
        issues.append(f"excluded open_status: {rec['open_status']}")

    for d_field in ["difficulty_clinical_knowledge", "difficulty_research_depth", "difficulty_multi_step_reasoning"]:
        val = rec.get(d_field)
        if val is not None and (not isinstance(val, int) or val < 1 or val > 5):
            issues.append(f"invalid {d_field}: {val}")

    return issues


def embedding_dedup(new_questions: list[str], existing_questions: list[str], threshold: float = 0.90):
    """Find duplicates between new and existing questions using sentence embeddings."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("WARNING: sentence_transformers not available, skipping embedding dedup", file=sys.stderr)
        return set()

    print("Loading sentence-transformers model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print(f"Encoding {len(existing_questions)} existing questions...")
    existing_emb = model.encode(existing_questions, batch_size=256, show_progress_bar=True, normalize_embeddings=True)

    print(f"Encoding {len(new_questions)} new questions...")
    new_emb = model.encode(new_questions, batch_size=256, show_progress_bar=True, normalize_embeddings=True)

    # Cross-similarity: find new questions too similar to existing
    print("Computing cross-similarity...")
    dup_indices = set()
    batch_size = 500
    for i in range(0, len(new_emb), batch_size):
        batch = new_emb[i:i+batch_size]
        sims = np.dot(batch, existing_emb.T)
        max_sims = sims.max(axis=1)
        for j, sim in enumerate(max_sims):
            if sim >= threshold:
                dup_indices.add(i + j)

    # Internal dedup within new questions
    print("Computing internal dedup...")
    internal_dups = set()
    for i in range(0, len(new_emb), batch_size):
        batch = new_emb[i:i+batch_size]
        sims = np.dot(batch, new_emb.T)
        for j in range(len(batch)):
            global_j = i + j
            if global_j in internal_dups:
                continue
            for k in range(global_j + 1, len(new_emb)):
                if sims[j, k] >= threshold:
                    internal_dups.add(k)

    return dup_indices | internal_dups


def main():
    print("=" * 60)
    print("Expanded Questions Verification Pipeline")
    print("=" * 60)

    # Load data
    print(f"\nLoading expanded questions from {EXPANDED_FILE}...")
    expanded = load_jsonl(EXPANDED_FILE)
    print(f"  Loaded: {len(expanded)} questions")

    print(f"\nLoading existing gold answer corpus from {EXISTING_GOLD}...")
    existing = load_jsonl(EXISTING_GOLD)
    print(f"  Loaded: {len(existing)} questions")

    # Step 1: Structural validation
    print(f"\n{'='*60}")
    print("Step 1: Structural Validation")
    print(f"{'='*60}")

    valid_records = []
    rejected_structural = 0
    issue_counter = Counter()

    for i, rec in enumerate(expanded):
        issues = structural_check(rec, i)
        if issues:
            rejected_structural += 1
            for iss in issues:
                issue_counter[iss.split(":")[0].strip()] += 1
        else:
            valid_records.append(rec)

    print(f"  Passed: {len(valid_records)}/{len(expanded)}")
    print(f"  Rejected: {rejected_structural}")
    if issue_counter:
        print("  Issues:")
        for iss, count in issue_counter.most_common(10):
            print(f"    {iss}: {count}")

    # Step 2: Exact text dedup
    print(f"\n{'='*60}")
    print("Step 2: Exact Text Dedup")
    print(f"{'='*60}")

    existing_questions_set = {r.get("self_contained_question", "").strip().lower() for r in existing}
    pre_count = len(valid_records)
    valid_records = [r for r in valid_records if r.get("self_contained_question", "").strip().lower() not in existing_questions_set]
    exact_dups = pre_count - len(valid_records)
    print(f"  Exact duplicates with existing corpus: {exact_dups}")
    print(f"  Remaining: {len(valid_records)}")

    # Internal exact dedup
    seen_q = set()
    deduped = []
    internal_exact = 0
    for r in valid_records:
        q = r.get("self_contained_question", "").strip().lower()
        if q in seen_q:
            internal_exact += 1
            continue
        seen_q.add(q)
        deduped.append(r)
    valid_records = deduped
    print(f"  Internal exact duplicates: {internal_exact}")
    print(f"  After exact dedup: {len(valid_records)}")

    # Step 3: Embedding-based dedup
    print(f"\n{'='*60}")
    print("Step 3: Embedding-Based Dedup (threshold=0.90)")
    print(f"{'='*60}")

    new_questions = [r["self_contained_question"] for r in valid_records]
    existing_questions = [r.get("self_contained_question", r.get("original_question", "")) for r in existing]

    dup_indices = embedding_dedup(new_questions, existing_questions, threshold=0.90)
    print(f"\n  Semantic duplicates found: {len(dup_indices)}")

    final_records = [r for i, r in enumerate(valid_records) if i not in dup_indices]
    print(f"  Final verified questions: {len(final_records)}")

    # Step 4: Statistics
    print(f"\n{'='*60}")
    print("Step 4: Final Statistics")
    print(f"{'='*60}")

    taxonomy_dist = Counter(r.get("taxonomy_l1", "Unknown") for r in final_records)
    print(f"\n  Taxonomy L1 distribution:")
    for tax, count in taxonomy_dist.most_common():
        print(f"    {tax}: {count} ({100*count/len(final_records):.1f}%)")

    avg_diff = sum(
        (r.get("difficulty_clinical_knowledge", 3) + r.get("difficulty_research_depth", 3) + r.get("difficulty_multi_step_reasoning", 3)) / 3
        for r in final_records
    ) / max(len(final_records), 1)
    print(f"\n  Average difficulty: {avg_diff:.2f}")

    status_dist = Counter(r.get("open_status", "unknown") for r in final_records)
    print(f"\n  Open status distribution:")
    for status, count in status_dist.most_common():
        print(f"    {status}: {count}")

    # Write output
    with open(OUTPUT_FILE, "w") as f:
        for r in final_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Total verified questions: {len(final_records)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
