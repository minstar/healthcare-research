"""Merge retrieval-verified (v2) + expert-curated (JLA/NICE) into the final corpus.

Adds a `corpus_track` tag distinguishing how each item's openness is grounded:
  - "retrieval_verified": PubMed/trial/arXiv items with retrieval-based status
  - "expert_consensus": JLA PSP / NICE — open by expert/consensus declaration

Aligns the two schemas to a union, dedups on exact self_contained_question text
(curated wins ties — it carries an explicit openness guarantee), and reports stats.

Usage:
    python scripts/merge_corpus.py \
        --v2 data/export/mcp_benchmark_v2.jsonl \
        --curated data/curated/curated_integrated.jsonl \
        --out data/export/mcp_benchmark_v3.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

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


def normalize(r: dict, track: str) -> dict:
    out = {**UNION_DEFAULTS, **{k: r[k] for k in UNION_DEFAULTS if k in r}}
    if not out["source"]:
        sid = out["source_id"]
        out["source"] = ("pubmed" if sid.startswith("PMID") else
                         "trial" if sid.startswith("NCT") else
                         "jla" if sid.startswith("JLA") else
                         "nice" if sid.startswith("NICE") else "other")
    out["corpus_track"] = track
    return out


def _key(q: str) -> str:
    return re.sub(r"\s+", " ", (q or "").lower().strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", default="data/export/mcp_benchmark_v2.jsonl")
    ap.add_argument("--curated", default="data/curated/curated_integrated.jsonl")
    ap.add_argument("--out", default="data/export/mcp_benchmark_v3.jsonl")
    args = ap.parse_args()

    v2 = [normalize(json.loads(l), "retrieval_verified") for l in open(args.v2)]
    cur = [normalize(json.loads(l), "expert_consensus") for l in open(args.curated)]

    # dedup on exact question text; curated wins (explicit openness guarantee)
    merged, seen, dropped = [], set(), 0
    for r in cur + v2:
        k = _key(r["self_contained_question"])
        if not k or k in seen:
            dropped += 1
            continue
        seen.add(k)
        merged.append(r)

    out_path = Path(args.out)
    with out_path.open("w") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    track = collections.Counter(r["corpus_track"] for r in merged)
    src = collections.Counter(r["source"] for r in merged)
    status = collections.Counter(r["open_status"] for r in merged)
    method = collections.Counter(r["status_method"] for r in merged)
    tax = collections.Counter(r["taxonomy_l1"] for r in merged)
    report = {
        "merged_total": len(merged),
        "exact_dups_dropped": dropped,
        "by_corpus_track": dict(track),
        "by_source": dict(src),
        "by_open_status": dict(status),
        "by_status_method": dict(method),
        "by_taxonomy_l1": dict(tax.most_common()),
    }
    with out_path.with_name("v3_merge_report.json").open("w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nwrote {out_path} ({len(merged)})")


if __name__ == "__main__":
    main()
