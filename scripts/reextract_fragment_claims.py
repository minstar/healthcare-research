#!/usr/bin/env python3
"""Re-extract proper claim snippets for the ~11% fragment-claim citations, using the FIXED
exp_cite_audit.claim_snippet on the original agent answers (traces). GPU-free, no API.
Outputs the citations needing re-judging with their corrected claims."""
import json, re, sys
sys.path.insert(0, "scripts")
from exp_cite_audit import claim_snippet, _is_fragment, PMID_RE, NCT_RE

import os
MODEL_DIR = {"GLM-5.1": "glm", "Qwen3.6": "qwen", "DeepSeek-V4": "dsv4"}
SET_DIR = {"1969": "baseline_{m}_1969", "priority": "priority_{m}", "expand": "expand_{m}"}
# (set, model, task_id) -> answer, across all 9 trace dirs
ANS = {}
for setname, pat in SET_DIR.items():
    for model, mtag in MODEL_DIR.items():
        f = f"results/{pat.format(m=mtag)}/traces.jsonl"
        if not os.path.exists(f):
            continue
        for l in open(f):
            d = json.loads(l); ANS[(setname, model, d["task_id"])] = d.get("model_answer", "")

def find_span(answer, id_type, idv):
    idv = str(idv)
    for mt in (PMID_RE if id_type == "PMID" else NCT_RE).finditer(answer):
        if mt.group(1) == idv or idv in mt.group(0):
            return mt.span()
    i = answer.find(idv)
    return (i, i + len(idv)) if i >= 0 else None

# fragment citations from the Opus L2 population (the set we will re-judge with Opus)
op = [json.loads(l) for l in open("results/l2_fullpop.jsonl")]
frag = [r for r in op if r.get("supports") in ("yes","partial","no") and _is_fragment(r.get("claim"))]
out, fixed, nofound, stillfrag = [], 0, 0, 0
with open("results/l2_fragments_reextracted.jsonl", "w") as o:
    for r in frag:
        ans = ANS.get((r["set"], r["model"], r["task_id"]), "")
        span = find_span(ans, r["id_type"], str(r["id"])) if ans else None
        newclaim = claim_snippet(ans, span) if span else ""
        rec = {k: r[k] for k in ("set","model","task_id","id_type","id")}
        rec["old_claim"] = r.get("claim"); rec["claim"] = newclaim
        rec["old_supports"] = r.get("supports")
        rec["reextracted"] = bool(span) and not _is_fragment(newclaim)
        if not ans or not span: nofound += 1
        elif _is_fragment(newclaim): stillfrag += 1
        else: fixed += 1
        o.write(json.dumps(rec) + "\n"); out.append(rec)
print(f"fragments={len(frag)}  reextracted_ok={fixed}  still_fragment={stillfrag}  answer/span_not_found={nofound}")
print("sample fixes:")
for r in out:
    if r["reextracted"]: print(f"  OLD {r['old_claim'][:45]!r}\n  NEW {r['claim'][:90]!r}\n")
    if sum(1 for x in out[:out.index(r)+1] if x['reextracted'])>=3: break
