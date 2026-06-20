#!/usr/bin/env python3
"""Compute human-vs-LLM Cohen's kappa for the L2 citation annotation pilot.
Run AFTER results/cite_kappa/human_annotation_50.csv has its human_supports column filled.
GPU-free, no API. Mirrors results/cite_kappa/kappa_summary.json (LLM-vs-LLM)."""
import csv, json, sys
ANN = "results/cite_kappa/human_annotation_50.csv"
KEY = "results/cite_kappa/human_annotation_50_KEY.csv"

def cohen_kappa(a, b):
    cats = sorted(set(a) | set(b)); idx = {c:i for i,c in enumerate(cats)}
    n = len(a)
    if n == 0: return None, 0
    po = sum(1 for x,y in zip(a,b) if x==y)/n
    from collections import Counter
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[c]/n)*(cb[c]/n) for c in cats)
    k = (po-pe)/(1-pe) if (1-pe) else 1.0
    return k, po

key = {r["row_id"]: r for r in csv.DictReader(open(KEY))}
hum, glm, cla = [], [], []
for r in csv.DictReader(open(ANN)):
    h = (r.get("human_supports(yes/partial/no)") or "").strip().lower()
    if h not in ("yes","partial","no"):
        continue  # not yet annotated / inaccessible
    k = key.get(r["row_id"])
    if not k: continue
    hum.append(h); glm.append(k["glm_verdict"]); cla.append(k["claude_verdict"])

n = len(hum)
if n == 0:
    print("No annotated rows yet — fill human_supports in", ANN); sys.exit(0)

def binary(v): return "no" if v=="no" else "support"   # wrong-paper vs (yes|partial)
out = {"n_annotated": n}
for name, llm in [("glm", glm), ("claude", cla)]:
    k3, po3 = cohen_kappa(hum, llm)
    kb, pob = cohen_kappa([binary(x) for x in hum], [binary(x) for x in llm])
    out[f"human_vs_{name}_kappa_3way"] = round(k3,3)
    out[f"human_vs_{name}_agreement_3way"] = round(po3,3)
    out[f"human_vs_{name}_kappa_wrongpaper_binary"] = round(kb,3)
out["human_wrongpaper_rate"] = round(sum(1 for x in hum if x=="no")/n, 3)
out["glm_wrongpaper_rate"] = round(sum(1 for x in glm if x=="no")/n, 3)
out["claude_wrongpaper_rate"] = round(sum(1 for x in cla if x=="no")/n, 3)
json.dump(out, open("results/cite_kappa/human_kappa_summary.json","w"), indent=2)
print(json.dumps(out, indent=2))
