"""Export the expansion track (priority2 + cochrane2 + medrxiv2 crawl -> extract -> refine)
into a SEPARATE file, without mutating v3/v3.2/v3.3.

Same conventions as export_priority.py: normalize to the union schema, dedup on exact
self_contained_question text, tag corpus_track. Drops anything whose question text OR
source_id already exists in v3 or priority_questions (purely additive/new).

Usage: python scripts/export_expand.py
"""
from __future__ import annotations
import collections, glob, json, re
from pathlib import Path

REFINED = sorted(glob.glob("data/refined/batch_4[0-3][0-9].jsonl"))
V3 = "data/export/mcp_benchmark_v3.jsonl"
PRIOR = "data/export/priority_questions.jsonl"
OUT = "data/export/expand_questions.jsonl"

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


def track_of(sid: str) -> str:
    # crawl provenance by id prefix (DOI = medrxiv/cochrane preprint, PMID = lit)
    return "expand_preprint" if sid.startswith("DOI") else "expand_priority_lit"


def normalize(r: dict) -> dict:
    out = {**UNION_DEFAULTS, **{k: r[k] for k in UNION_DEFAULTS if k in r}}
    sid = out["source_id"]
    if not out["source"]:
        out["source"] = ("pubmed" if sid.startswith("PMID") else
                         "preprint" if sid.startswith("DOI") else
                         "trial" if sid.startswith("NCT") else "other")
    out["corpus_track"] = track_of(sid)
    return out


def main():
    # dedup on QUESTION TEXT only (a single paper can legitimately raise several distinct
    # open questions — source_id dedup would wrongly collapse them).
    seen_q = set()
    for fp in (V3, PRIOR):
        if not Path(fp).exists():
            continue
        for l in open(fp):
            seen_q.add(key(json.loads(l).get("self_contained_question", "")))

    merged = []
    n_in = dup_q = empty = 0
    for fp in REFINED:
        for l in open(fp):
            n_in += 1
            r = normalize(json.loads(l))
            k = key(r["self_contained_question"])
            if not k:
                empty += 1; continue
            if k in seen_q:
                dup_q += 1; continue
            seen_q.add(k)
            merged.append(r)

    with open(OUT, "w") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"refined batches: {len(REFINED)} | input rows: {n_in}")
    print(f"  dropped: empty={empty} dup_question={dup_q}")
    print(f"  exported NEW expansion questions: {len(merged)} -> {OUT}")
    print("  corpus_track:", dict(collections.Counter(r["corpus_track"] for r in merged)))
    print("  open_status:", dict(collections.Counter(r["open_status"] for r in merged)))
    print("  taxonomy_l1:", dict(collections.Counter(r["taxonomy_l1"] for r in merged)))


if __name__ == "__main__":
    main()
