#!/usr/bin/env python3
"""Split all_questions_refined.jsonl into batches of 3 for gold answer generation.

Usage: python scripts/track_b/prepare_gold_batches.py

Reads:  data/extracted/all_questions_refined.jsonl
Writes: data/gold_batches/batch_000.jsonl .. batch_NNN.jsonl
"""
import json
import os
import sys

BASE_DIR = "/data/project/private/minstar/workspace/healthcare-research"
INPUT_FILE = os.path.join(BASE_DIR, "data/extracted/all_questions_refined.jsonl")
OUTPUT_DIR = os.path.join(BASE_DIR, "data/gold_batches")
BATCH_SIZE = 3


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Input file not found: {INPUT_FILE}", file=sys.stderr)
        sys.exit(1)

    # Load all questions
    questions = []
    with open(INPUT_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    print(f"Loaded {len(questions)} questions from {INPUT_FILE}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Split into batches of BATCH_SIZE
    batch_num = 0
    for start in range(0, len(questions), BATCH_SIZE):
        batch = questions[start : start + BATCH_SIZE]
        batch_file = os.path.join(OUTPUT_DIR, f"batch_{batch_num:03d}.jsonl")
        with open(batch_file, "w") as f:
            for doc in batch:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        batch_num += 1

    print(f"Created {batch_num} batch files (size={BATCH_SIZE}) in {OUTPUT_DIR}")
    print(f"Run gold generation: seq 0 {batch_num - 1} | xargs -P 10 -I{{}} python scripts/track_b/generate_gold_answers.py {{}}")


if __name__ == "__main__":
    main()
