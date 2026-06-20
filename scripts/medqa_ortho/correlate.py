#!/usr/bin/env python3
"""Combine closed-form MC accuracy with OpenBioRQ pass@0.5 and test orthogonality.

Reads results/medqa_ortho/<tag>_summary.json (from mc_eval) + leaderboard_core_t0.json,
prints a per-model table and the Spearman rank correlation between MedQA-style accuracy and
OpenBioRQ full-core pass@0.5. The claim: models cluster high on MedQA but spread on OpenBioRQ,
and the rank correlation is weak -> OpenBioRQ measures a dimension MedQA does not.
Writes results/medqa_ortho/orthogonality.json.
"""
import json, glob, os
from collections import OrderedDict

ROOT = "/data/project/private/minstar/workspace/healthcare-research"
MO = f"{ROOT}/results/medqa_ortho"

# tag -> (display, OpenBioRQ leaderboard name)
TAGS = OrderedDict([
    ("glm51", ("GLM-5.1", "GLM-5.1")),
    ("qwen36", ("Qwen3.6", "Qwen3.6")),
    ("dsv4", ("DeepSeek-V4", "DeepSeek-V4")),
    ("glm5", ("GLM-5", "GLM-5")),
    ("qwen35_397b", ("Qwen3.5-397B", "Qwen3.5-397B")),
    ("qwen3_235b", ("Qwen3-235B", "Qwen3-235B")),
    # frontier (filled if frontier summaries exist; OpenBioRQ name = robust-core leaderboard)
    ("gemini3pro", ("Gemini-3-Pro", None)),
    ("opus47", ("Opus-4.7", None)),
    ("gpt55", ("GPT-5.5", None)),
])
# frontier robust-core pass@0.5 (from frontier_robust_leaderboard.json) for the y-axis
FRONTIER_ROBUST = {"Gemini-3-Pro": 0.288, "Opus-4.7": 0.378, "GPT-5.5": 0.596}

lb = {e["model"]: e for e in json.load(open(f"{ROOT}/results/leaderboard_core_t0.json"))["leaderboard"]}

def spearman(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    def rank(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    vx = sum((rx[i] - mx) ** 2 for i in range(n)) ** 0.5
    vy = sum((ry[i] - my) ** 2 for i in range(n)) ** 0.5
    return cov / (vx * vy) if vx and vy else None

rows = []
for tag, (disp, obrq) in TAGS.items():
    sp = f"{MO}/{tag}_summary.json"
    if not os.path.exists(sp):
        continue
    s = json.load(open(sp))["per_dataset"]
    medqa = s.get("MedQA", {}).get("acc")
    pubmed = s.get("PubMedQA", {}).get("acc")
    medmcqa = s.get("MedMCQA", {}).get("acc")
    accs = [a for a in (medqa, pubmed, medmcqa) if a is not None]
    mcavg = round(sum(accs) / len(accs), 1) if accs else None
    # OpenBioRQ y-value
    full = robust = None
    if obrq and obrq in lb:
        full = round(100 * lb[obrq]["full_core"]["solve@0.5"], 1)
        robust = round(100 * lb[obrq]["robust_core"]["solve@0.5"], 1)
    elif disp in FRONTIER_ROBUST:
        robust = round(100 * FRONTIER_ROBUST[disp], 1)
    rows.append({"model": disp, "MedQA": medqa, "PubMedQA": pubmed, "MedMCQA": medmcqa,
                 "MC_avg": mcavg, "obrq_full": full, "obrq_robust": robust})

# orthogonality on the open-weight set (have both MedQA and full-core)
ow = [r for r in rows if r["MedQA"] is not None and r["obrq_full"] is not None]
rho_full = spearman([r["MedQA"] for r in ow], [r["obrq_full"] for r in ow]) if len(ow) >= 3 else None
rho_avg = spearman([r["MC_avg"] for r in ow], [r["obrq_full"] for r in ow]) if len(ow) >= 3 else None

out = {"rows": rows,
       "open_weight_n": len(ow),
       "spearman_MedQA_vs_OBRQfull": round(rho_full, 3) if rho_full is not None else None,
       "spearman_MCavg_vs_OBRQfull": round(rho_avg, 3) if rho_avg is not None else None}
if ow:
    mq = [r["MedQA"] for r in ow]
    out["MedQA_range"] = [min(mq), max(mq)]
    out["OBRQfull_range"] = [min(r["obrq_full"] for r in ow), max(r["obrq_full"] for r in ow)]
json.dump(out, open(f"{MO}/orthogonality.json", "w"), indent=2)

print(f"{'Model':14s} {'MedQA':>7s} {'PubMed':>7s} {'MedMCQA':>8s} {'MCavg':>7s} | {'OBRQ-full':>9s} {'OBRQ-rob':>9s}")
print("-" * 78)
for r in rows:
    f = lambda v: f"{v:.1f}" if v is not None else "  -"
    print(f"{r['model']:14s} {f(r['MedQA']):>7s} {f(r['PubMedQA']):>7s} {f(r['MedMCQA']):>8s} {f(r['MC_avg']):>7s} | {f(r['obrq_full']):>9s} {f(r['obrq_robust']):>9s}")
print("-" * 78)
if ow:
    print(f"open-weight (n={len(ow)}): MedQA range {out['MedQA_range']}  vs  OpenBioRQ-full range {out['OBRQfull_range']}")
    print(f"Spearman  MedQA vs OBRQ-full = {out['spearman_MedQA_vs_OBRQfull']}")
    print(f"Spearman  MCavg vs OBRQ-full = {out['spearman_MCavg_vs_OBRQfull']}")
print(f"\nwrote {MO}/orthogonality.json")
