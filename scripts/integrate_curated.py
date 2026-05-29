"""Integrate expert-curated open-question sources (JLA PSPs, NICE) into v2 schema.

These sources are STRUCTURALLY open (declared by expert/consensus or evidence-gap
analysis), so open_status is fixed to "open" with an expert-consensus provenance —
no Stage-1/2 retrieval verification needed. Only taxonomy, 3-axis difficulty, and
MCP-tool mapping are filled, via batched Claude CLI (cf. refine_batch.py).

Usage:
    python scripts/integrate_curated.py \
        --inputs data/raw/jla/jla_questions.jsonl data/raw/nice/nice_questions.jsonl \
        --out data/curated/curated_integrated.jsonl --batch-size 25 --workers 4
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SYSTEM = """You are an expert medical research curator. For each question, output ONE JSON \
object per line (JSONL), no markdown. Keys:
- "idx": integer index as given
- "taxonomy_l1": one of [Clinical Medicine, Oncology, Neuroscience & Psychiatry, \
Infectious Disease & Immunology, Cardiovascular Medicine, Genomics & Precision Medicine, \
Pharmacology & Drug Discovery, Public Health & Epidemiology, Rare & Orphan Diseases, \
Surgical Sciences, Medical AI & Informatics, Other]
- "taxonomy_l2": sub-domain
- "taxonomy_l3": specific topic tag
- "relevant_mcp_tools": subset of [pubmed, clinicaltrialsgov, openfda, opentargets, chembl, \
uniprot, pubchem, kegg, ncbi-datasets, biomcp]
- "difficulty_clinical_knowledge": 1-5
- "difficulty_research_depth": 1-5
- "difficulty_multi_step_reasoning": 1-5
Output ONLY the JSONL, one object per question."""


def call_claude(prompt: str, model: str, timeout: int = 300) -> list[dict]:
    try:
        r = subprocess.run(["claude", "--print", "--model", model, "--system-prompt", SYSTEM],
                           input=prompt, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return []
    out = []
    for line in r.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        try:
            o = json.loads(line)
            if isinstance(o, dict) and "taxonomy_l1" in o:
                out.append(o)
        except json.JSONDecodeError:
            continue
    return out


def build_prompt(batch: list[dict]) -> str:
    parts = ["Classify these medical research questions:\n"]
    for i, d in enumerate(batch):
        parts.append(f"--- Question {i} ---")
        parts.append(f"question: {d.get('self_contained_question','')}")
        parts.append(f"domain_hint: {d.get('clinical_domain','')}")
        parts.append("")
    return "\n".join(parts)


def integrate_batch(batch: list[dict], model: str) -> list[dict]:
    refs = {r.get("idx", -1): r for r in call_claude(build_prompt(batch), model)}
    out = []
    for i, d in enumerate(batch):
        r = refs.get(i, {})
        src = d.get("source", "curated")
        merged = {
            **{k: d.get(k, "") for k in ("source", "source_id", "source_url",
                                         "source_title", "original_question")},
            "self_contained_question": d.get("self_contained_question", ""),
            "question_type": d.get("question_type", ""),
            "clinical_domain": d.get("clinical_domain", ""),
            "why_open": d.get("why_open", ""),
            "taxonomy_l1": r.get("taxonomy_l1", "Other"),
            "taxonomy_l2": r.get("taxonomy_l2", ""),
            "taxonomy_l3": r.get("taxonomy_l3", ""),
            # structurally open — fixed, not LLM-judged
            "open_status": "open",
            "status_reasoning": d.get("why_open", ""),
            "relevant_mcp_tools": r.get("relevant_mcp_tools", ["pubmed"]),
            "difficulty_clinical_knowledge": r.get("difficulty_clinical_knowledge", 3),
            "difficulty_research_depth": r.get("difficulty_research_depth", 3),
            "difficulty_multi_step_reasoning": r.get("difficulty_multi_step_reasoning", 3),
            # provenance (parallels audit/cleanup fields)
            "status_method": "expert_consensus",
            "audit_flag": f"{src}_curated",
            "n_followups": 0,
            "src_year": "",
            "metadata": d.get("metadata", {}),
            "classified": bool(r),
        }
        out.append(merged)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", default="data/curated/curated_integrated.jsonl")
    ap.add_argument("--batch-size", type=int, default=25)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = []
    for path in args.inputs:
        rows.extend(json.loads(l) for l in open(path))
    if args.limit:
        rows = rows[: args.limit]
    batches = [rows[i:i + args.batch_size] for i in range(0, len(rows), args.batch_size)]
    print(f"Integrating {len(rows)} curated questions in {len(batches)} batches (model={args.model})...")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results, done = [], [0]

    def work(b):
        r = integrate_batch(b, args.model)
        done[0] += 1
        if done[0] % 20 == 0:
            print(f"  {done[0]}/{len(batches)} batches")
        return r

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(work, batches):
            results.extend(r)

    classified = sum(1 for r in results if r["classified"])
    with out_path.open("w") as f:
        for r in results:
            r.pop("classified", None)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"DONE: {len(results)} written ({classified} classified) → {out_path}")


if __name__ == "__main__":
    main()
