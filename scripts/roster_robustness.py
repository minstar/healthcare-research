#!/usr/bin/env python3
"""Roster-composition robustness of the robust core (Reviewer R3 round-2 W4).

The robust core is defined by one triple (GLM-5.1/Qwen3.6/DeepSeek-V4). We re-derive the
all-three-fail set with alternate triples drawn from our open-weight T=0 runs and report Jaccard
overlap with the original, to show the core is a property of the capability tier, not the triple.
Output: results/roster_robustness.json
"""
import json
def load(tag):
    return {json.loads(l)["task_id"]: float(json.loads(l)["checklist_score"])
            for l in open(f"results/baseline_{tag}/checklist_glm.jsonl")
            if json.loads(l).get("checklist_score") is not None}
M = {"GLM-5.1": load("glm51_tools"), "Qwen3.6": load("roster_qwen36_t0"),
     "DeepSeek-V4": load("roster_dsv4_t0"), "GLM-5": load("heldout_glm5"),
     "Qwen3.5-397B": load("heldout_qwen35_397b"), "Qwen3-235B": load("heldout_qwen3_235b")}
common = set.intersection(*[set(v) for v in M.values()])
def robust(triple): return set(t for t in common if all(M[m][t] < 0.5 for m in triple))
R0 = robust(("GLM-5.1", "Qwen3.6", "DeepSeek-V4"))
def jac(a, b): return len(a & b) / len(a | b) if (a | b) else 0
alts = {"swap GLM-5.1->GLM-5": ("GLM-5", "Qwen3.6", "DeepSeek-V4"),
        "swap Qwen3.6->Qwen3.5-397B": ("GLM-5.1", "Qwen3.5-397B", "DeepSeek-V4"),
        "2 held-out + DSV4": ("GLM-5", "Qwen3.5-397B", "DeepSeek-V4"),
        "all-3 held-out": ("GLM-5", "Qwen3.5-397B", "Qwen3-235B")}
out = {"common_ids": len(common), "original_core": len(R0), "alternates": {}}
for name, tr in alts.items():
    R = robust(tr)
    out["alternates"][name] = {"n": len(R), "jaccard": round(jac(R0, R), 3),
                               "containment_of_R0": round(len(R0 & R) / len(R0), 3)}
json.dump(out, open("results/roster_robustness.json", "w"), indent=2)
print(json.dumps(out, indent=2))
