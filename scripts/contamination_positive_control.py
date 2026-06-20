#!/usr/bin/env python3
"""Positive control for the contamination / still-open audit (Reviewer D).

The audit reports hard_contamination = 0/657. That is only meaningful if the detector
CAN detect contamination. We inject known-positive items that SHOULD trip the two
hard-contamination flags and measure detection recall:

  * synthetic_template      : source_id is KEGG/UniProt-derived (classify_source -> synthetic)
  * trial_resolved_but_open : a REAL completed-with-results trial, mislabeled open_status="open"

We also inject genuine-open negatives (recruiting/active trials) to confirm specificity
(they must NOT be flagged hard). Output: results/contamination_poscontrol.json.
"""
import json, sys, requests
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.status_verifier import StatusVerifier, classify_source

CT = "https://clinicaltrials.gov/api/v2/studies"
HARD = {"synthetic_template", "trial_resolved_but_open"}

def fetch_ncts(term, n):
    r = requests.get(CT, params={"query.term": term, "fields": "NCTId,OverallStatus,HasResults",
                                 "pageSize": n}, timeout=30)
    r.raise_for_status()
    out = []
    for s in r.json().get("studies", []):
        nct = s.get("protocolSection", {}).get("identificationModule", {}).get("nctId")
        st = s.get("protocolSection", {}).get("statusModule", {}).get("overallStatus")
        if nct:
            out.append((nct, st, s.get("hasResults")))
    return out

def build():
    items = []
    # (1) trial_resolved_but_open positives: completed + results, but labeled open
    for nct, st, hr in fetch_ncts("AREA[OverallStatus]COMPLETED AND AREA[HasResults]true", 20):
        items.append({"source_id": f"NCT:{nct}", "open_status": "open",
                      "self_contained_question": f"(injected) Is the primary endpoint of trial {nct} still unresolved?",
                      "_expect": "trial_resolved_but_open", "_category": "trial_resolved_positive"})
    # (2) synthetic_template positives: KEGG / UniProt / map source ids
    synth_ids = ["hsa04010", "hsa04151", "hsa05200", "hsa04210", "hsa04150", "map00010", "map01100",
                 "P04637", "P38398", "Q9Y6K9", "P00533", "O15111", "P01308", "Q16665"]
    for sid in synth_ids:
        assert classify_source(sid) == "synthetic", sid
        items.append({"source_id": sid, "open_status": "open",
                      "self_contained_question": f"(injected) What is the unresolved regulatory role in {sid}?",
                      "_expect": "synthetic_template", "_category": "synthetic_positive"})
    # (3) genuine-open NEGATIVES: recruiting/active trials (must NOT be hard-flagged)
    for nct, st, hr in fetch_ncts("AREA[OverallStatus]RECRUITING", 12):
        items.append({"source_id": f"NCT:{nct}", "open_status": "open",
                      "self_contained_question": f"(injected) Will trial {nct} confirm its hypothesis?",
                      "_expect": "NOT_hard", "_category": "open_negative"})
    return items

def main():
    items = build()
    v = StatusVerifier(min_interval=0.1, citing_top_k=6)
    rows = []
    for it in items:
        res = v.screen(it)
        rows.append({**{k: it[k] for k in ("source_id", "_expect", "_category")},
                     "flag": res["flag"], "detail": res.get("detail", ""),
                     "is_hard": res["flag"] in HARD})
        print(f"  {it['_category']:24s} {it['source_id']:16s} expect={it['_expect']:22s} -> flag={res['flag']}")

    def recall(cat):
        sub = [r for r in rows if r["_category"] == cat]
        hit = sum(1 for r in sub if r["is_hard"])
        return hit, len(sub), (round(100 * hit / len(sub), 1) if sub else None)

    tr = recall("trial_resolved_positive")
    sy = recall("synthetic_positive")
    neg = [r for r in rows if r["_category"] == "open_negative"]
    fp = sum(1 for r in neg if r["is_hard"])
    pos_total = tr[1] + sy[1]
    pos_hit = tr[0] + sy[0]
    summary = {
        "trial_resolved_recall": {"detected": tr[0], "n": tr[1], "pct": tr[2]},
        "synthetic_recall": {"detected": sy[0], "n": sy[1], "pct": sy[2]},
        "overall_positive_recall": {"detected": pos_hit, "n": pos_total,
                                    "pct": round(100 * pos_hit / pos_total, 1) if pos_total else None},
        "negative_false_positive": {"flagged_hard": fp, "n": len(neg),
                                    "pct": round(100 * fp / len(neg), 1) if neg else None},
    }
    out = {"summary": summary, "rows": rows}
    Path("results").mkdir(exist_ok=True)
    json.dump(out, open("results/contamination_poscontrol.json", "w"), indent=2)
    print("\n===== POSITIVE CONTROL =====")
    print(json.dumps(summary, indent=2))
    print("wrote results/contamination_poscontrol.json")

if __name__ == "__main__":
    main()
