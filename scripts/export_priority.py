"""Export the priority-setting track (581 authoritative priority docs -> 525 extracted+refined)
into a SEPARATE file, without mutating v3.

Follows merge_corpus conventions: normalize to the union schema, dedup on exact
self_contained_question text, tag corpus_track="priority_setting". Also drops any
question whose text already exists in v3 (so this file is purely additive/new).

Usage: python scripts/export_priority.py
"""
from __future__ import annotations
import collections, glob, json, re
from pathlib import Path

REFINED = sorted(glob.glob("data/refined/batch_3[0-1][0-9].jsonl"))
V3 = "data/export/mcp_benchmark_v3.jsonl"
OUT = "data/export/priority_questions.jsonl"

UNION_DEFAULTS = {
    "source": "", "source_id": "", "source_url": "", "source_title": "",
    "original_question": "", "self_contained_question": "", "question_type": "",
    "clinical_domain": "", "why_open": "", "taxonomy_l1": "Other", "taxonomy_l2": "",
    "taxonomy_l3": "", "open_status": "unknown", "status_reasoning": "",
    "verification_venues": [], "relevant_mcp_tools": [],
    "difficulty_clinical_knowledge": 3, "difficulty_research_depth": 3,
    "difficulty_multi_step_reasoning": 3, "status_method": "none",
    "audit_flag": "ok", "n_followups": 0, "src_year": "", "metadata": {},
}


def key(q: str) -> str:
    return re.sub(r"\s+", " ", (q or "").lower().strip())


def normalize(r: dict) -> dict:
    out = {**UNION_DEFAULTS, **{k: r[k] for k in UNION_DEFAULTS if k in r}}
    if not out["source"]:
        sid = out["source_id"]
        out["source"] = ("pubmed" if sid.startswith("PMID") else
                         "trial" if sid.startswith("NCT") else "other")
    out["corpus_track"] = "priority_setting"
    return out


def main():
    v3_keys = {key(json.loads(l).get("self_contained_question", "")) for l in open(V3)}
    merged, seen = [], set()
    n_in = dup_internal = dup_v3 = empty = 0
    for fp in REFINED:
        for l in open(fp):
            n_in += 1
            r = normalize(json.loads(l))
            k = key(r["self_contained_question"])
            if not k:
                empty += 1; continue
            if k in seen:
                dup_internal += 1; continue
            if k in v3_keys:
                dup_v3 += 1; continue
            seen.add(k)
            merged.append(r)

    with open(OUT, "w") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"refined batches: {len(REFINED)} | input rows: {n_in}")
    print(f"  dropped: empty={empty} internal_dup={dup_internal} already_in_v3={dup_v3}")
    print(f"  exported NEW priority questions: {len(merged)} -> {OUT}")
    print("  open_status:", dict(collections.Counter(r["open_status"] for r in merged)))
    print("  taxonomy_l1:", dict(collections.Counter(r["taxonomy_l1"] for r in merged)))


if __name__ == "__main__":
    main()
