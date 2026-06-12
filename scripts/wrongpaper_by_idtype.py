#!/usr/bin/env python3
"""Wrong-paper (L2) rate split by citation identifier type (PMID vs NCT vs other).
Addresses the reviewer point that trial-registry (NCT) and publication (PMID) citations should
not be merged. Uses the independent-family (Opus) full-population L2 re-judge. GPU-free, no API.
"""
import json
from collections import defaultdict
L2 = "results/l2_fullpop.jsonl"
rows = [json.loads(l) for l in open(L2)]
judges = {r.get("judge_model") for r in rows}
by = defaultdict(lambda: [0, 0])  # id_type -> [wrong, total]
for r in rows:
    sup = r.get("supports")
    if sup not in ("yes", "partial", "no"):
        continue
    it = r.get("id_type") or "other"
    by[it][1] += 1
    if sup == "no":
        by[it][0] += 1
out = {"judge_model(s)": sorted(j for j in judges if j),
       "by_id_type": {it: {"wrong_paper": w, "total": t, "pct": round(100*w/t, 1)}
                      for it, (w, t) in by.items() if t}}
json.dump(out, open("results/wrongpaper_by_idtype.json", "w"), indent=2)
print(json.dumps(out, indent=2))
