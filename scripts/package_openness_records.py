#!/usr/bin/env python3
"""Package the per-item openness audit records for the OpenBioRQ release (R5f).

Merges the 657-core still-open audit (decision summary for every core question) with the
Stage-2 judge's detailed reasoning + cited evidence IDs where available. This is the
openness-audit *decision record*, NOT the raw Stage-1 evidence bundle (the fetched abstracts
were not persisted) -- exact re-derivation still requires re-querying the (moving) literature.
GPU-free, no API.
"""
import json, os
CORE = "results/still_open_audit_core.jsonl"
DETAIL = ["results/stage2_stillopen/stage2_sample.jsonl",
          "results/status_ablation/stage2_sample.jsonl"]
OUT = "results/release/openness_audit_records.jsonl"
os.makedirs("results/release", exist_ok=True)

detail = {}
for f in DETAIL:
    if os.path.exists(f):
        for l in open(f):
            r = json.loads(l); detail[r["source_id"]] = r

n, n_det, n_resolved = 0, 0, 0
with open(OUT, "w") as o:
    for l in open(CORE):
        c = json.loads(l); sid = c["source_id"]
        rec = {"source_id": sid, "flag": c.get("flag"),
               "severity": c.get("severity"), "detail": c.get("detail"),
               "still_open": c.get("severity") != "resolved"}
        d = detail.get(sid)
        if d:
            n_det += 1
            rec["stage2"] = {"new_status": d.get("new_status"),
                             "n_evidence": d.get("n_evidence"),
                             "cited_evidence_ids": d.get("cited_ids"),
                             "reason": d.get("reason")}
        if c.get("severity") == "resolved":
            n_resolved += 1
        o.write(json.dumps(rec) + "\n"); n += 1

print(f"wrote {OUT}: {n} core records ({n_det} with Stage-2 cited-evidence+reasoning), "
      f"{n_resolved} flagged resolved")
