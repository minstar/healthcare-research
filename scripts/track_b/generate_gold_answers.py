#!/usr/bin/env python3
"""Generate gold reference answers for a batch of medical questions via Claude CLI.

Usage: python scripts/track_b/generate_gold_answers.py <batch_number>

Reads:  data/gold_batches/batch_NNN.jsonl
Writes: data/gold_answers/batch_NNN.jsonl

Parallel execution:
  seq 0 656 | xargs -P 10 -I{} python scripts/track_b/generate_gold_answers.py {}
"""
import json
import os
import subprocess
import sys
import time

BASE_DIR = "/data/project/private/minstar/workspace/healthcare-research"
BATCH_DIR = os.path.join(BASE_DIR, "data/gold_batches")
OUTPUT_DIR = os.path.join(BASE_DIR, "data/gold_answers")

SYSTEM_PROMPT = """For each medical question below, generate a comprehensive gold reference answer as a JSON object with:

- "idx": question index (integer, as given)
- "current_knowledge": 2-3 paragraphs summarizing what IS currently known about this topic. Be specific — cite study types (RCTs, meta-analyses, cohort studies), key findings, and established mechanisms.
- "unknown_aspects": 1-2 paragraphs on what specifically remains unknown or debated. What are the key gaps?
- "evidence_landscape": Brief description of evidence quality — are there RCTs? Only preclinical data? Conflicting systematic reviews?
- "key_citations": list of objects {"type": "PMID"|"NCT"|"DOI", "id": "...", "relevance": "one sentence"}. Include 3-8 real citations you are confident about.
- "mcp_tool_plan": list of objects {"tool": "pubmed"|"clinicaltrialsgov"|etc, "query": "exact search query", "purpose": "what this retrieves"}. Include 2-5 tool queries that would help investigate this question.
- "answer_summary": 2-4 paragraph synthesis of the best current understanding, written for a researcher.
- "self_completeness": float 0.0-1.0 — model's self-assessed epistemic difficulty: how completely can this question be answered with current evidence?

Output ONLY valid JSON objects, one per line (JSONL). No markdown, no explanations."""

MAX_RETRIES = 2
TIMEOUT_SECONDS = 600  # Gold answers are longer; allow more time


def generate_gold_answers(batch_num: int):
    batch_file = os.path.join(BATCH_DIR, f"batch_{batch_num:03d}.jsonl")
    output_file = os.path.join(OUTPUT_DIR, f"batch_{batch_num:03d}.jsonl")

    if not os.path.exists(batch_file):
        print(f"Batch file not found: {batch_file}", file=sys.stderr)
        sys.exit(1)

    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        print(f"Output exists: {output_file} — skipping")
        return

    # Load batch questions
    docs = []
    with open(batch_file) as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))

    if not docs:
        print(f"Batch {batch_num:03d}: empty batch file, skipping", file=sys.stderr)
        return

    # Build prompt with question details
    prompt_parts = [
        "Generate comprehensive gold reference answers for these open medical questions:\n"
    ]
    for i, doc in enumerate(docs):
        prompt_parts.append(f"--- Question {i} ---")
        prompt_parts.append(f"source_id: {doc.get('source_id', '')}")
        prompt_parts.append(
            f"question: {doc.get('self_contained_question', doc.get('original_question', ''))}"
        )
        prompt_parts.append(f"type: {doc.get('question_type', '')}")
        prompt_parts.append(f"domain: {doc.get('clinical_domain', '')}")
        prompt_parts.append(f"why_open: {doc.get('why_open', '')}")
        prompt_parts.append(f"taxonomy: {doc.get('taxonomy_l1', '')} > {doc.get('taxonomy_l2', '')} > {doc.get('taxonomy_l3', '')}")
        prompt_parts.append(f"open_status: {doc.get('open_status', '')}")
        prompt_parts.append(f"relevant_tools: {json.dumps(doc.get('relevant_mcp_tools', []))}")
        prompt_parts.append("")

    prompt = "\n".join(prompt_parts)

    # Call Claude CLI with retries
    raw_output = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            result = subprocess.run(
                [
                    "claude",
                    "--print",
                    "--model",
                    "opus",
                    "--system-prompt",
                    SYSTEM_PROMPT,
                ],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
            )
            if result.returncode != 0:
                print(
                    f"  Batch {batch_num:03d} attempt {attempt}: claude exited {result.returncode}: {result.stderr[:200]}",
                    file=sys.stderr,
                )
                if attempt < MAX_RETRIES:
                    time.sleep(5 * (attempt + 1))
                    continue
                return

            raw_output = result.stdout.strip()
            if raw_output:
                break
            else:
                print(
                    f"  Batch {batch_num:03d} attempt {attempt}: empty output",
                    file=sys.stderr,
                )
                if attempt < MAX_RETRIES:
                    time.sleep(5 * (attempt + 1))
                    continue

        except subprocess.TimeoutExpired:
            print(
                f"  Batch {batch_num:03d} attempt {attempt}: timed out after {TIMEOUT_SECONDS}s",
                file=sys.stderr,
            )
            if attempt < MAX_RETRIES:
                time.sleep(5 * (attempt + 1))
                continue
            return
        except Exception as e:
            print(
                f"  Batch {batch_num:03d} attempt {attempt}: error: {e}",
                file=sys.stderr,
            )
            if attempt < MAX_RETRIES:
                time.sleep(5 * (attempt + 1))
                continue
            return

    if not raw_output:
        print(f"  Batch {batch_num:03d}: no output after {MAX_RETRIES + 1} attempts", file=sys.stderr)
        return

    # Parse JSONL output from Claude
    gold_answers = []
    for line in raw_output.split("\n"):
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict) and "current_knowledge" in obj:
                gold_answers.append(obj)
        except json.JSONDecodeError:
            continue

    if not gold_answers:
        # Try parsing as a JSON array (Claude sometimes outputs an array instead of JSONL)
        try:
            parsed = json.loads(raw_output)
            if isinstance(parsed, list):
                gold_answers = [
                    obj
                    for obj in parsed
                    if isinstance(obj, dict) and "current_knowledge" in obj
                ]
        except json.JSONDecodeError:
            pass

    if not gold_answers:
        # Last resort: try to extract JSON objects from markdown code blocks
        import re

        json_blocks = re.findall(r"```(?:json)?\s*\n(.*?)```", raw_output, re.DOTALL)
        for block in json_blocks:
            for line in block.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict) and "current_knowledge" in obj:
                        gold_answers.append(obj)
                except json.JSONDecodeError:
                    continue
            # Also try parsing the whole block as a JSON array
            if not gold_answers:
                try:
                    parsed = json.loads(block.strip())
                    if isinstance(parsed, list):
                        gold_answers = [
                            obj
                            for obj in parsed
                            if isinstance(obj, dict) and "current_knowledge" in obj
                        ]
                except json.JSONDecodeError:
                    pass

    if not gold_answers:
        print(
            f"  Batch {batch_num:03d}: could not parse any gold answers from output ({len(raw_output)} chars)",
            file=sys.stderr,
        )
        # Write raw output for debugging
        debug_file = os.path.join(OUTPUT_DIR, f"batch_{batch_num:03d}.debug.txt")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(debug_file, "w") as f:
            f.write(raw_output)
        return

    # Merge gold answers back into original questions by idx
    answer_by_idx = {a.get("idx", -1): a for a in gold_answers}
    merged = []
    for i, doc in enumerate(docs):
        answer = answer_by_idx.get(i, {})
        doc["gold_answer"] = {
            "current_knowledge": answer.get("current_knowledge", ""),
            "unknown_aspects": answer.get("unknown_aspects", ""),
            "evidence_landscape": answer.get("evidence_landscape", ""),
            "key_citations": answer.get("key_citations", []),
            "mcp_tool_plan": answer.get("mcp_tool_plan", []),
            "answer_summary": answer.get("answer_summary", ""),
            "self_completeness": answer.get("self_completeness", 0.0),
        }
        merged.append(doc)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(output_file, "w") as f:
        for doc in merged:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    answered_count = sum(1 for a in gold_answers if a.get("current_knowledge"))

    # Inline completeness check per answer
    quality_pass = 0
    for a in gold_answers:
        ck = len(a.get("current_knowledge", ""))
        ua = len(a.get("unknown_aspects", ""))
        ans = len(a.get("answer_summary", ""))
        cites = len(a.get("key_citations", []))
        comp = a.get("self_completeness", -1)
        if ck >= 200 and ua >= 100 and ans >= 200 and cites >= 2 and 0.0 <= comp <= 1.0:
            quality_pass += 1

    print(f"Batch {batch_num:03d}: {len(docs)} questions, {answered_count}/{len(gold_answers)} generated, {quality_pass}/{len(gold_answers)} pass quality")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python scripts/track_b/generate_gold_answers.py <batch_number>",
            file=sys.stderr,
        )
        sys.exit(1)
    generate_gold_answers(int(sys.argv[1]))
