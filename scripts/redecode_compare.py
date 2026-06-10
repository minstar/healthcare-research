#!/usr/bin/env python3
"""Reproducibility of the robust core (Reviewer C): seed-1 vs seed-2 at T=0.

Compares the original roster T=0 decode (seed 1) against an independent second T=0 decode
(seed 2, days later -> tests agentic / live-API non-determinism) over the 423 robust-core
questions. Reports per-model test-retest score agreement, robust-core membership retention
(a question stays iff all three roster models still score <0.5), and a bootstrap CI on the
retained count. Output: results/redecode_stability.json
"""
import json, glob
from statistics import median

ROOT = "/data/project/private/minstar/workspace/healthcare-research"
R = f"{ROOT}/results"
# seed1 (original) dir -> seed2 (re-decode) dir, per model
PAIRS = {
    "GLM-5.1":     ("baseline_glm51_tools", "redecode_glm51"),
    "Qwen3.6":     ("baseline_roster_qwen36_t0", "redecode_qwen36"),
    "DeepSeek-V4": ("baseline_roster_dsv4_t0", "redecode_dsv4"),
}
THR = 0.5
rc = set(json.load(open(f"{R}/robust_core_ids.json")))

def load(d):
    s = {}
    p = f"{R}/{d}/checklist_glm.jsonl"
    if not glob.glob(p):
        return s
    for l in open(p):
        r = json.loads(l)
        if r.get("checklist_score") is not None and r["task_id"] in rc:
            s[r["task_id"]] = float(r["checklist_score"])
    return s

per_model = {}
retest = {}
for m, (d1, d2) in PAIRS.items():
    s1, s2 = load(d1), load(d2)
    common = [t for t in rc if t in s1 and t in s2]
    if not common:
        per_model[m] = {"error": "seed2 not found or unjudged", "seed2_n": len(s2)}
        continue
    # test-retest: mean abs score delta, fail-label agreement (<0.5)
    deltas = [abs(s1[t] - s2[t]) for t in common]
    lbl_agree = sum(1 for t in common if (s1[t] < THR) == (s2[t] < THR)) / len(common)
    # how many that were failing (in robust core => <0.5 at seed1) now PASS at seed2
    flips_to_pass = sum(1 for t in common if s1[t] < THR and s2[t] >= THR)
    per_model[m] = {"n": len(common), "mean_abs_delta": round(sum(deltas)/len(deltas), 3),
                    "median_abs_delta": round(median(deltas), 3),
                    "fail_label_agreement": round(lbl_agree, 3),
                    "flips_fail_to_pass": flips_to_pass}
    retest[m] = (s1, s2, common)

# membership retention: a robust-core item is RETAINED iff all 3 models still <0.5 at seed2
ids_all = [t for t in rc if all(t in retest.get(m, ({},{},[]))[1] for m in PAIRS)] if len(retest) == 3 else []
retained = []
left = []
for t in ids_all:
    if all(retest[m][1][t] < THR for m in PAIRS):
        retained.append(t)
    else:
        left.append(t)
retention = {"checked": len(ids_all), "retained": len(retained), "left": len(left),
             "retention_pct": round(100*len(retained)/len(ids_all), 1) if ids_all else None}

# bootstrap CI on retention pct (resample the checked ids)
def bootstrap_ci(flags, B=2000):
    if not flags:
        return None
    n = len(flags)
    # deterministic-ish bootstrap without RNG: use index rotations as pseudo-resamples
    import math
    # simple percentile bootstrap via cyclic resampling seeds derived from index
    means = []
    for b in range(B):
        # LCG for reproducibility (no Math.random/Date)
        seed = (b * 2654435761) & 0xFFFFFFFF
        acc = 0
        for _ in range(n):
            seed = (1103515245 * seed + 12345) & 0x7FFFFFFF
            acc += flags[seed % n]
        means.append(100 * acc / n)
    means.sort()
    lo = means[int(0.025 * B)]
    hi = means[int(0.975 * B)]
    return [round(lo, 1), round(hi, 1)]

flags = [1 if t in set(retained) else 0 for t in ids_all]
retention["bootstrap_95ci"] = bootstrap_ci(flags)

out = {"per_model_test_retest": per_model, "membership_retention": retention}
json.dump(out, open(f"{R}/redecode_stability.json", "w"), indent=2)
print(json.dumps(out, indent=2))
print(f"\nwrote {R}/redecode_stability.json")
