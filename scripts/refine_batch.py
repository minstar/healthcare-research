#!/usr/bin/env python3
"""Refine a batch of questions via Claude CLI (Haiku).

Usage: python refine_batch.py <batch_number>
"""
import json
import sys
import os
import subprocess

BATCH_DIR = "/data/project/private/minstar/workspace/healthcare-research/data/refine_batches"
OUTPUT_DIR = "/data/project/private/minstar/workspace/healthcare-research/data/refined"

SYSTEM_PROMPT = """You are an expert medical research curator. For each question below, output a JSON object with:

- "idx": the question index (integer, as given)
- "taxonomy_l1": primary domain from: Clinical Medicine, Oncology, Neuroscience & Psychiatry, Infectious Disease & Immunology, Cardiovascular Medicine, Genomics & Precision Medicine, Pharmacology & Drug Discovery, Public Health & Epidemiology, Rare & Orphan Diseases, Surgical Sciences, Medical AI & Informatics, Other
- "taxonomy_l2": sub-domain (e.g., "Cancer Biology", "Neurodegeneration", "Antimicrobial Resistance")
- "taxonomy_l3": specific topic tag (e.g., "tumor microenvironment", "CRISPR therapeutics")
- "open_status": one of ["open", "partially_answered", "answered", "unknown"]
- "status_reasoning": 1-2 sentences on why this status
- "verification_venues": list of conferences/journals/workshops where this is discussed as open (e.g., ["ASCO", "Nature Medicine", "WHO Priority Pathogens"])
- "relevant_mcp_tools": list from [pubmed, clinicaltrialsgov, openfda, opentargets, chembl, uniprot, pubchem, kegg, ncbi-datasets, biomcp]
- "difficulty_clinical_knowledge": 1-5
- "difficulty_research_depth": 1-5
- "difficulty_multi_step_reasoning": 1-5

Output ONLY valid JSON objects, one per line (JSONL). No markdown, no explanations."""


def refine_batch(batch_num: int):
    batch_file = os.path.join(BATCH_DIR, f"batch_{batch_num:03d}.jsonl")
    output_file = os.path.join(OUTPUT_DIR, f"batch_{batch_num:03d}.jsonl")

    if not os.path.exists(batch_file):
        print(f"Batch file not found: {batch_file}", file=sys.stderr)
        sys.exit(1)

    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        print(f"Output exists: {output_file} — skipping")
        return

    docs = []
    with open(batch_file) as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))

    prompt_parts = ["Refine these medical questions:\n"]
    for i, doc in enumerate(docs):
        prompt_parts.append(f"--- Question {i} ---")
        prompt_parts.append(f"source_id: {doc.get('source_id', '')}")
        prompt_parts.append(f"question: {doc.get('self_contained_question', '')}")
        prompt_parts.append(f"type: {doc.get('question_type', '')}")
        prompt_parts.append(f"domain: {doc.get('clinical_domain', '')}")
        prompt_parts.append(f"why_open: {doc.get('why_open', '')}")
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
        print(f"  Claude CLI timed out for batch {batch_num}", file=sys.stderr)
        return
    except Exception as e:
        print(f"  Claude CLI error: {e}", file=sys.stderr)
        return

    refinements = []
    for line in output.split("\n"):
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        try:
            r = json.loads(line)
            if isinstance(r, dict) and "taxonomy_l1" in r:
                refinements.append(r)
        except json.JSONDecodeError:
            continue

    # Merge refinements back into original questions
    ref_by_idx = {r.get("idx", -1): r for r in refinements}
    merged = []
    for i, doc in enumerate(docs):
        ref = ref_by_idx.get(i, {})
        doc["taxonomy_l1"] = ref.get("taxonomy_l1", "")
        doc["taxonomy_l2"] = ref.get("taxonomy_l2", "")
        doc["taxonomy_l3"] = ref.get("taxonomy_l3", "")
        doc["open_status"] = ref.get("open_status", "unknown")
        doc["status_reasoning"] = ref.get("status_reasoning", "")
        doc["verification_venues"] = ref.get("verification_venues", [])
        doc["relevant_mcp_tools"] = ref.get("relevant_mcp_tools", [])
        doc["difficulty_clinical_knowledge"] = ref.get("difficulty_clinical_knowledge", doc.get("difficulty", 3))
        doc["difficulty_research_depth"] = ref.get("difficulty_research_depth", doc.get("difficulty", 3))
        doc["difficulty_multi_step_reasoning"] = ref.get("difficulty_multi_step_reasoning", doc.get("difficulty", 3))
        merged.append(doc)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(output_file, "w") as f:
        for doc in merged:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"Batch {batch_num:03d}: {len(docs)} questions, {len(refinements)} refined")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python refine_batch.py <batch_number>", file=sys.stderr)
        sys.exit(1)
    refine_batch(int(sys.argv[1]))
