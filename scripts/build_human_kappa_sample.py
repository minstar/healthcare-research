#!/usr/bin/env python3
"""Build a 50-citation human-expert kappa annotation sample for OpenBioRQ L2 (wrong-paper) audit.

Subsamples the existing 118-pair stratified frame (results/cite_kappa/pairs.jsonl) used for the
GLM-vs-Claude cross-family kappa (kappa=0.755), so human-vs-LLM kappa is directly comparable.
Outputs a BLIND annotator CSV (no LLM verdicts shown) + a hidden KEY CSV for the later join.
GPU-free, no API. Deterministic (no RNG) for reproducibility.
"""
import json, csv, os
SRC = "results/cite_kappa/pairs.jsonl"
OUT = "results/cite_kappa/human_annotation_50.csv"
KEY = "results/cite_kappa/human_annotation_50_KEY.csv"
N = 50

def id_type_and_url(idv):
    s = str(idv)
    if s.upper().startswith("NCT"):
        return "NCT", f"https://clinicaltrials.gov/study/{s}"
    if s.isdigit():
        return "PMID", f"https://pubmed.ncbi.nlm.nih.gov/{s}/"
    return "other", ""

rows = [json.loads(l) for l in open(SRC)]
for i, r in enumerate(rows):
    r["_orig"] = i
    r["_disagree"] = (r.get("glm") != r.get("claude"))

# Stratify by the PRIMARY (glm) verdict to keep yes/partial/no balanced (informative kappa),
# prioritize: (1) all GLM-5.1 model rows (only 4, else under-represented),
# (2) GLM-vs-Claude DISAGREEMENTS (most informative for human adjudication),
# (3) fill the rest evenly per glm-verdict bucket. Deterministic ordering.
chosen, seen = [], set()
def take(r):
    if r["_orig"] not in seen and len(chosen) < N:
        seen.add(r["_orig"]); chosen.append(r)

# (1) all GLM-5.1 rows
for r in sorted(rows, key=lambda x: x["_orig"]):
    if r.get("model") == "GLM-5.1":
        take(r)
# (2) disagreements, balanced across glm verdict
for v in ["no", "partial", "yes"]:
    for r in sorted([x for x in rows if x["_disagree"] and x.get("glm")==v], key=lambda x: x["_orig"]):
        take(r)
# (3) fill remaining evenly per glm verdict bucket (round-robin)
buckets = {v: [r for r in sorted(rows, key=lambda x: x["_orig"]) if r.get("glm")==v] for v in ["no","partial","yes"]}
order = ["no","partial","yes"]
ptr = {v:0 for v in order}; k=0
while len(chosen) < N:
    v = order[k % 3]; k += 1
    b = buckets[v]
    while ptr[v] < len(b) and b[ptr[v]]["_orig"] in seen: ptr[v]+=1
    if ptr[v] < len(b): take(b[ptr[v]]); ptr[v]+=1
    if k > 10000: break

chosen = chosen[:N]
# stable order for the sheet: by model then id
chosen.sort(key=lambda r:(r.get("model",""), str(r.get("id"))))

os.makedirs("results/cite_kappa", exist_ok=True)
with open(OUT,"w",newline="") as f:
    w = csv.writer(f)
    w.writerow(["row_id","model","id_type","id","paper_url","claim","paper_title",
                "human_supports(yes/partial/no)","human_notes"])
    for i,r in enumerate(chosen,1):
        it,url = id_type_and_url(r.get("id"))
        claim = " ".join(str(r.get("claim","")).split())  # collapse whitespace/newlines for the cell
        w.writerow([i, r.get("model"), it, r.get("id"), url, claim, r.get("title",""), "", ""])
with open(KEY,"w",newline="") as f:
    w = csv.writer(f)
    w.writerow(["row_id","id","model","set","glm_verdict","claude_verdict","llm_disagree"])
    for i,r in enumerate(chosen,1):
        w.writerow([i, r.get("id"), r.get("model"), r.get("set"), r.get("glm"), r.get("claude"),
                    int(r["_disagree"])])

from collections import Counter
print("wrote", OUT, "and", KEY, "n=", len(chosen))
print("model:", dict(Counter(r.get("model") for r in chosen)))
print("glm verdict:", dict(Counter(r.get("glm") for r in chosen)))
print("claude verdict:", dict(Counter(r.get("claude") for r in chosen)))
print("llm disagreements:", sum(r["_disagree"] for r in chosen))
print("id_type:", dict(Counter(id_type_and_url(r.get("id"))[0] for r in chosen)))
