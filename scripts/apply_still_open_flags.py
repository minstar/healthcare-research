"""Apply the LLM-free still-open audit (audit_contamination on core_nanje) back onto v3.2.

Adds a reversible `still_open_flag` field to every row (default "none"). For the 657
core_nanje, maps the audit flag -> severity:
  trial_completed + results posted  -> "high_risk"   (trial done AND results public)
  trial_completed (no results)      -> "recheck"
  no_followup_evidence              -> "recheck"
  unverifiable_source / arxiv check -> "unverifiable"
  heavy_followup_recheck / ok       -> "none"         (normal for a live front)

We do NOT auto-quarantine: a completed trial does not prove the broader research
question is resolved. high_risk/recheck are queued for LLM stage2 grounded re-judgment
(needs the GLM endpoint) before any quarantine decision.

Usage: python scripts/apply_still_open_flags.py
"""
from __future__ import annotations
import collections, json

V32 = "data/export/mcp_benchmark_v3.2.jsonl"
AUDIT = "results/audit_core_nanje/audit_per_item.jsonl"
OUT = "data/export/mcp_benchmark_v3.2.jsonl"   # in-place rewrite (reversible via recalibrate)
DETAIL = "results/still_open_audit_core.jsonl"

SEVERITY = {
    "trial_completed": None,        # decided per results-posted below
    "no_followup_evidence": "recheck",
    "unverifiable_source": "unverifiable",
    "arxiv_citation_check_needed": "unverifiable",
    "heavy_followup_recheck": "none",
    "ok": "none",
}


def main():
    audit = {}
    for l in open(AUDIT):
        a = json.loads(l)
        flag = a["flag"]
        if flag == "trial_completed":
            sev = "high_risk" if "results=True" in str(a.get("detail", "")) else "recheck"
        else:
            sev = SEVERITY.get(flag, "recheck")
        audit[a["source_id"]] = {"flag": flag, "severity": sev, "detail": a.get("detail", "")}

    rows = [json.loads(l) for l in open(V32)]
    sev_cnt = collections.Counter()
    with open(OUT, "w") as f:
        for r in rows:
            a = audit.get(r.get("source_id"))
            if r.get("nanje_core") and a:
                r["still_open_flag"] = a["severity"]
                r["still_open_audit"] = {"flag": a["flag"], "detail": a["detail"]}
                sev_cnt[a["severity"]] += 1
            else:
                r.setdefault("still_open_flag", "none")
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(DETAIL, "w") as f:
        for sid, a in audit.items():
            f.write(json.dumps({"source_id": sid, **a}, ensure_ascii=False) + "\n")

    print(f"v3.2 rows rewritten: {len(rows)} | core_nanje flagged: {sum(sev_cnt.values())}")
    print(f"  still_open_flag (core_nanje): {dict(sev_cnt)}")
    print(f"  -> {DETAIL}")
    print(f"  high_risk + recheck queued for LLM stage2: "
          f"{sev_cnt['high_risk'] + sev_cnt['recheck']}")


if __name__ == "__main__":
    main()
