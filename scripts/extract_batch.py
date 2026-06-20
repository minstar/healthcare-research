#!/usr/bin/env python3
"""Extract open medical questions from a batch of raw documents using Claude.

Usage: python extract_batch.py <batch_number> [--batch-dir DIR] [--output-dir DIR]
"""
import json
import sys
import os
import subprocess
import re

BATCH_DIR = "/data/project/private/minstar/workspace/healthcare-research/data/raw/_batches"
OUTPUT_DIR = "/data/project/private/minstar/workspace/healthcare-research/data/extracted"

SYSTEM_PROMPT = """You are a medical research analyst. Given a scientific document (title + abstract), extract ALL open/unsolved questions, problems, or challenges mentioned or implied.

For each open question found, output a JSON object with these fields:
- source_id: the document's source_id
- source_url: the document's url
- source_title: the document's title
- original_question: the exact text from the document that signals an open question
- self_contained_question: a reformulated, self-contained version of the question (define all abbreviations, include clinical context)
- question_type: one of [mechanism, treatment, diagnosis, prevention, epidemiology, methodology, prognosis]
- clinical_domain: the medical specialty (e.g., Neurology, Oncology, Cardiology)
- why_open: brief explanation of why this question remains unsolved
- difficulty: 1-5 rating (5 = hardest)

If no open questions are found, output nothing for that document.
Output ONLY valid JSON objects, one per line (JSONL format). No markdown, no explanations."""


def extract_from_docs(docs: list[dict]) -> list[dict]:
    """Use Claude CLI to extract questions from a set of documents."""
    prompt_parts = ["Extract all open/unsolved medical questions from these documents:\n"]
    for i, doc in enumerate(docs):
        prompt_parts.append(f"--- Document {i+1} ---")
        prompt_parts.append(f"source_id: {doc.get('source_id', '')}")
        prompt_parts.append(f"url: {doc.get('url', '')}")
        prompt_parts.append(f"title: {doc.get('title', '')}")
        abstract = doc.get('abstract', '')[:3000]
        prompt_parts.append(f"abstract: {abstract}")
        prompt_parts.append("")

    prompt = "\n".join(prompt_parts)

    try:
        result = subprocess.run(
            ["claude", "--print", "--model", "haiku", "--system-prompt", SYSTEM_PROMPT],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=300,
        )
        output = result.stdout.strip()
    except subprocess.TimeoutExpired:
        print(f"  Claude CLI timed out", file=sys.stderr)
        return []
    except Exception as e:
        print(f"  Claude CLI error: {e}", file=sys.stderr)
        return []

    questions = []
    for line in output.split("\n"):
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        try:
            q = json.loads(line)
            if isinstance(q, dict) and "source_id" in q:
                questions.append(q)
        except json.JSONDecodeError:
            continue

    return questions


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_batch.py <batch_number>", file=sys.stderr)
        sys.exit(1)

    batch_num = int(sys.argv[1])
    batch_dir = sys.argv[2] if len(sys.argv) > 2 else BATCH_DIR
    output_dir = sys.argv[3] if len(sys.argv) > 3 else OUTPUT_DIR

    batch_file = os.path.join(batch_dir, f"batch_{batch_num:03d}.jsonl")
    output_file = os.path.join(output_dir, f"batch_{batch_num:03d}.jsonl")

    if not os.path.exists(batch_file):
        print(f"Batch file not found: {batch_file}", file=sys.stderr)
        sys.exit(1)

    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        print(f"Output already exists: {output_file} — skipping")
        sys.exit(0)

    docs = []
    with open(batch_file) as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))

    print(f"Processing batch {batch_num:03d}: {len(docs)} documents")

    all_questions = []
    chunk_size = 5
    for i in range(0, len(docs), chunk_size):
        chunk = docs[i:i+chunk_size]
        print(f"  Chunk {i//chunk_size + 1}/{(len(docs)-1)//chunk_size + 1} ({len(chunk)} docs)...")
        questions = extract_from_docs(chunk)
        all_questions.extend(questions)
        print(f"    -> {len(questions)} questions extracted")

    os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "w") as f:
        for q in all_questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"Batch {batch_num:03d} complete: {len(all_questions)} questions -> {output_file}")


if __name__ == "__main__":
    main()
