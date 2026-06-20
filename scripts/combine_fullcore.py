#!/usr/bin/env python3
"""Combine frozen-423 (paper) + new-234 into the Full-core(657) solve% + Wilson 95% CI.

To keep the published Frozen(423) column byte-identical, the 423 solves are read from the
EXISTING results/api_*_robust/checklist_glm.jsonl (the paper's frozen scores), and only the
234 complement solves are read from the freshly-judged results/api_*_fullcore/checklist_glm.jsonl.
Full-core(657) solve% = (solves_423 + solves_234) / 657.

Also prints a judge-drift cross-check: the fullcore checklist's OWN 423-subset solve% should
match the robust frozen number; a large gap would flag judge nondeterminism (judge T=0.1).

solve := checklist_score >= 0.5 (matches build_leaderboard_t0.py).
"""
import json, math, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THRESH = 0.5
CORE = f"{ROOT}/data/eval_samples/core_eval.jsonl"
FROZ = f"{ROOT}/data/eval_samples/robust_core_eval.jsonl"

CFG = [  # (display, robust_dir(423 paper), fullcore_dir(657 fresh))
    ("GPT-5.5",            "api_gpt55_robust",        "api_gpt55_fullcore"),
    ("Opus-4.7",           "api_opus47_robust",       "api_opus47_fullcore"),
    ("Gemini-3-Pro",       "api_gemini3pro_robust",   "api_gemini3pro_fullcore"),
    ("GPT-5.5 (no-tools)", "api_gpt55_notool_robust", "api_gpt55_notool_fullcore"),
]


def scores(path):
    """task_id -> checklist_score (only scored rows)."""
    out = {}
    if not os.path.exists(path):
        return out
    for l in open(path):
        try:
            d = json.loads(l)
        except Exception:
            continue
        s = d.get("checklist_score")
        if s is not None:
            out[d["task_id"]] = float(s)
    return out


def wilson(k, n):
    if n == 0:
        return 0.0, 0.0, 0.0
    z = 1.96; p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, 100 * (c - h), 100 * (c + h)


core = [json.loads(l)["task_id"] for l in open(CORE)]
froz = {json.loads(l)["task_id"] for l in open(FROZ)}
comp = [t for t in core if t not in froz]  # 234 complement
assert len(core) == 657 and len(froz) == 423 and len(comp) == 234, (len(core), len(froz), len(comp))

print(f"{'Model':<20}{'Frozen423 (paper)':>20}{'New234':>16}{'FULL-CORE 657 [Wilson95]':>34}{'drift':>8}")
print("-" * 98)
results = []
for name, robdir, fulldir in CFG:
    s423 = scores(f"{ROOT}/results/{robdir}/checklist_glm.jsonl")
    sful = scores(f"{ROOT}/results/{fulldir}/checklist_glm.jsonl")

    # frozen 423 (paper): restrict robust checklist to the 423 frozen ids
    k423 = sum(1 for t in froz if s423.get(t, 0) >= THRESH)
    n423 = sum(1 for t in froz if t in s423)
    # new 234: restrict fullcore checklist to the complement ids
    k234 = sum(1 for t in comp if sful.get(t, 0) >= THRESH)
    n234 = sum(1 for t in comp if t in sful)
    # combine over 657
    k657, n657 = k423 + k234, n423 + n234
    p, lo, hi = wilson(k657, n657)
    pf = 100 * k423 / n423 if n423 else 0.0
    pn = 100 * k234 / n234 if n234 else 0.0
    # drift cross-check: fullcore's own 423-subset solve%
    k423f = sum(1 for t in froz if sful.get(t, 0) >= THRESH)
    n423f = sum(1 for t in froz if t in sful)
    drift = (100 * k423f / n423f - pf) if n423f else float("nan")

    miss = "" if n657 == 657 else f"  !! only {n657}/657 scored"
    print(f"{name:<20}{f'{pf:.1f}% ({k423}/{n423})':>20}{f'{pn:.1f}% ({k234}/{n234})':>16}"
          f"{f'{p:.1f}% [{lo:.1f},{hi:.1f}] ({k657}/{n657})':>34}{f'{drift:+.1f}pp':>8}{miss}")
    results.append({"model": name, "frozen423_pct": round(pf, 1), "frozen423": [k423, n423],
                    "new234_pct": round(pn, 1), "new234": [k234, n234],
                    "fullcore657_pct": round(p, 1), "fullcore657_ci": [round(lo, 1), round(hi, 1)],
                    "fullcore657": [k657, n657], "judge_drift_pp_on_423": round(drift, 2)})

json.dump(results, open(f"{ROOT}/results/fullcore_combined.json", "w"), indent=2)
print("\n-> results/fullcore_combined.json")
print("NOTE: 'drift' = fullcore-judged 423-subset minus paper frozen-423; |drift|>~2pp would flag judge nondeterminism.")
