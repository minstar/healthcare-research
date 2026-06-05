"""Compile the core-set leaderboard: roster (difficulty-defining) + held-out + no-tool.

The held-out models had NO say in defining the core set; if they also score ~0 pass@0.5,
the 'core is hard' claim is no longer circular. Output: results/leaderboard_core.json + table.
"""
import json, os, statistics

CORE = "data/eval_samples/core_eval.jsonl"
N = sum(1 for _ in open(CORE))

def score_file(path):
    if not os.path.exists(path):
        return None
    s = {}
    for l in open(path):
        r = json.loads(l)
        if r.get("checklist_score") is not None:
            s[r["task_id"]] = float(r["checklist_score"])
    if not s:
        return None
    v = list(s.values())
    return {"n": len(v), "avg": round(statistics.mean(v), 3),
            "pass@0.5": round(sum(x >= 0.5 for x in v) / len(v), 4)}

rows = []
# roster (from the existing core baseline, difficulty-defining models)
try:
    base = json.load(open("results/core_nanje_baseline.json"))["models"]
    for m, d in base.items():
        rows.append({"model": m, "role": "roster (difficulty-defining)",
                     "n": d["n"], "avg": d["avg_checklist"], "pass@0.5": d["pass_at_0.5"]})
except Exception as e:
    print("roster baseline missing:", e)

# held-out + no-tool (computed from this run)
for tag, label, role in [
    ("heldout_glm5", "GLM-5", "held-out"),
    ("heldout_qwen3_235b", "Qwen3-235B-A22B", "held-out (older gen)"),
    ("heldout_qwen35_397b", "Qwen3.5-397B-A17B", "held-out"),
    ("glm51_tools", "GLM-5.1 (tools, T=0)", "control: roster @ T=0"),
    ("notool_glm51", "GLM-5.1 (no tools)", "ablation: no-tool")]:
    # NOTE: closed-API frontier baselines (GPT-5.5/Opus-4.7/Gemini-3.1-Pro) were dropped —
    # leaderboard is open-source / local models only.
    sc = score_file(f"results/baseline_{tag}/checklist_glm.jsonl")
    if sc:
        rows.append({"model": label, "role": role, **sc})
    else:
        rows.append({"model": label, "role": role, "n": 0, "avg": None, "pass@0.5": None, "status": "pending"})

json.dump({"core_n": N, "leaderboard": rows}, open("results/leaderboard_core.json", "w"), indent=2)
print(f"\n=== CORE-SET LEADERBOARD (n={N}) ===")
print(f'{"model":<22}{"role":<26}{"n":>5}{"avg":>8}{"pass@0.5":>10}')
for r in rows:
    a = "" if r["avg"] is None else f'{r["avg"]:.3f}'
    p = "" if r["pass@0.5"] is None else f'{r["pass@0.5"]:.1%}'
    print(f'{r["model"]:<22}{r["role"]:<26}{r["n"]:>5}{a:>8}{p:>10}')
print("-> results/leaderboard_core.json")
