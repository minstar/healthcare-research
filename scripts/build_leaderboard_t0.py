"""Temperature-0 leaderboard + robust-core re-definition.

All models evaluated at temperature 0.0 on the (temp-0.3-defined) core 657. We additionally
define the ROBUST CORE = questions all three roster models (GLM-5.1/Qwen3.6/DeepSeek-V4)
still fail (checklist < 0.5) AT TEMPERATURE 0.0 — i.e. hard under deterministic decoding, not
just at the original sampling temperature. Held-out models are then read on the robust core.
"""
import json, os, statistics

CORE = "data/eval_samples/core_eval.jsonl"
CORE_IDS = {json.loads(l)["task_id"] for l in open(CORE)}
N = len(CORE_IDS)

def load(tag):
    f = f"results/baseline_{tag}/checklist_glm.jsonl"
    if not os.path.exists(f):
        return None
    s = {}
    for l in open(f):
        r = json.loads(l)
        if r.get("checklist_score") is not None and r["task_id"] in CORE_IDS:
            s[r["task_id"]] = float(r["checklist_score"])
    return s or None

# roster @ T=0  (GLM-5.1 = the with-tools control; qwen36/dsv4 = this run)
roster = {"GLM-5.1": load("glm51_tools"), "Qwen3.6": load("roster_qwen36_t0"),
          "DeepSeek-V4": load("roster_dsv4_t0")}

# robust core = all three roster fail (<0.5) at T=0 (only over ids all three scored)
robust = set()
if all(roster.values()):
    common = set(roster["GLM-5.1"]) & set(roster["Qwen3.6"]) & set(roster["DeepSeek-V4"])
    robust = {t for t in common if all(roster[m][t] < 0.5 for m in roster)}

def stats(s, ids=None):
    if not s:
        return None
    v = [s[t] for t in (ids if ids is not None else s) if t in s]
    if not v:
        return None
    return {"n": len(v), "avg": round(statistics.mean(v), 3),
            "solve@0.5": round(sum(x >= 0.5 for x in v) / len(v), 4)}

MODELS = [
    ("glm51_tools", "GLM-5.1", "roster @ T=0"),
    ("roster_qwen36_t0", "Qwen3.6", "roster @ T=0"),
    ("roster_dsv4_t0", "DeepSeek-V4", "roster @ T=0"),
    ("heldout_glm5", "GLM-5", "held-out @ T=0"),
    ("heldout_qwen3_235b", "Qwen3-235B", "held-out (older) @ T=0"),
    ("heldout_qwen35_397b", "Qwen3.5-397B", "held-out @ T=0"),
    ("notool_glm51", "GLM-5.1 (no tools)", "ablation @ T=0"),
]
rows = []
for tag, name, role in MODELS:
    s = load(tag)
    rows.append({"model": name, "role": role, "full_core": stats(s),
                 "robust_core": stats(s, robust) if robust else None})

out = {"core_n": N, "robust_core_n": len(robust), "leaderboard": rows}
json.dump(out, open("results/leaderboard_core_t0.json", "w"), indent=2)
print(f"\n=== TEMPERATURE-0 LEADERBOARD (core n={N}; robust-core n={len(robust)}) ===")
print(f'{"model":<20}{"role":<24}{"full avg":>9}{"full s@.5":>10}{"robust s@.5":>12}')
for r in rows:
    fa = f'{r["full_core"]["avg"]:.3f}' if r["full_core"] else "-"
    fs = f'{r["full_core"]["solve@0.5"]:.1%}' if r["full_core"] else "-"
    rs = f'{r["robust_core"]["solve@0.5"]:.1%}' if r["robust_core"] else "-"
    print(f'{r["model"]:<20}{r["role"]:<24}{fa:>9}{fs:>10}{rs:>12}')
print(f"\nrobust core = {len(robust)}/{N} questions all 3 roster fail at T=0 -> results/leaderboard_core_t0.json")
