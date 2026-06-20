#!/usr/bin/env python3
"""Package per-model predictions for the OpenBioRQ HuggingFace release.

For each leaderboard model we ship, from its run directory:
  - predictions.jsonl : the model's answer + full agentic tool-use trajectory
  - checklist.jsonl   : the frozen-rubric judge verdicts (per criterion)
  - summary.json      : REGENERATED clean (solve@0.5 on full-657 and frozen-423)

Raw summary.json/per_task.csv from the run dirs are NOT shipped (they embed the
internal serving node, e.g. "...::http://c36f-...:8000/v1::glm-5.1").

Every record is passed through a defensive scrubber that redacts API keys,
internal hostnames/IPs, and local paths; the output is then re-scanned and the
build aborts if any private pattern survives.
"""
import json, os, re, sys, csv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, "data/hf_release/openbiorq/predictions")
FROZEN = set(json.load(open(os.path.join(ROOT, "results/robust_core_ids.json"))))

# slug -> (run_dir, role, tools, set)  ; all runs are the 657-question full core,
# frozen-core (423) numbers are obtained by filtering to FROZEN ids.
MODELS = [
    ("glm-5.1",          "baseline_glm51_tools",         "roster",   True),
    ("qwen3.6",          "baseline_roster_qwen36_t0",    "roster",   True),
    ("deepseek-v4",      "baseline_roster_dsv4_t0",      "roster",   True),
    ("glm-5",            "baseline_heldout_glm5",        "held-out", True),
    ("qwen3-235b",       "baseline_heldout_qwen3_235b",  "held-out", True),
    ("qwen3.5-397b",     "baseline_heldout_qwen35_397b", "held-out", True),
    ("glm-5.1-no-tools", "baseline_notool_glm51",        "ablation", False),
    ("gemini-3-pro",     "api_gemini3pro_fullcore",      "frontier", True),
    ("opus-4.7",         "api_opus47_fullcore",          "frontier", True),
    ("gpt-5.5",          "api_gpt55_fullcore",           "frontier", True),
    ("gpt-5.5-no-tools", "api_gpt55_notool_fullcore",    "frontier", False),
]

# ---- scrubbing -------------------------------------------------------------
SCRUB = [
    (re.compile(r"sk-or-v1-[A-Za-z0-9]{20,}"),            "[REDACTED_KEY]"),
    (re.compile(r"sk-[A-Za-z0-9]{20,}"),                  "[REDACTED_KEY]"),
    (re.compile(r"AIza[A-Za-z0-9_\-]{35}"),               "[REDACTED_KEY]"),
    (re.compile(r"c36f-[a-z0-9]+-gpu[0-9]+"),             "[REDACTED_HOST]"),
    (re.compile(r"https?://\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?"), "[REDACTED_URL]"),
    (re.compile(r"https?://[a-zA-Z0-9.\-]*:\d{2,5}(?:/v1)?"),  "[REDACTED_URL]"),
    (re.compile(r"/home/[a-zA-Z0-9_\-]+"),                "[PATH]"),
    (re.compile(r"/data/project/private/[a-zA-Z0-9_\-]+"),"[PATH]"),
]
# patterns that must NOT survive in shipped files
VERIFY = [
    re.compile(r"sk-[A-Za-z0-9]{20,}"), re.compile(r"sk-or-"),
    re.compile(r"AIza[A-Za-z0-9_\-]{35}"), re.compile(r"c36f-[a-z0-9]+-gpu[0-9]+"),
    re.compile(r"\d{1,3}(?:\.\d{1,3}){3}:\d+"), re.compile(r"/home/[a-z]"),
    re.compile(r"/data/project/private"), re.compile(r"minstar"),
]

def scrub(obj):
    s = json.dumps(obj, ensure_ascii=False)
    for pat, rep in SCRUB:
        s = pat.sub(rep, s)
    return json.loads(s)

def dedup(path, fields=None):
    """Load jsonl, keep last record per task_id, optional field projection."""
    d = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if fields:
                r = {k: r[k] for k in fields if k in r}
            d[r["task_id"]] = r
    return d

def write_jsonl(path, records):
    with open(path, "w") as o:
        for r in records:
            o.write(json.dumps(scrub(r), ensure_ascii=False) + "\n")

PRED_FIELDS = ["task_id", "model_answer", "tool_calls", "trace", "tokens", "wall_time"]
CK_FIELDS   = ["task_id", "checklist_score", "n_criteria", "verdicts"]

def solve_rate(scores, ids):
    sub = [scores[t] for t in ids if t in scores]
    if not sub: return None
    return round(100 * sum(1 for s in sub if s >= 0.5) / len(sub), 1)

def avg(scores, ids):
    sub = [scores[t] for t in ids if t in scores]
    return round(sum(sub) / len(sub), 3) if sub else None

os.makedirs(OUT, exist_ok=True)
manifest, lb_rows = [], []
for slug, run, role, tools in MODELS:
    rd = os.path.join(ROOT, "results", run)
    preds = dedup(os.path.join(rd, "traces.jsonl"), PRED_FIELDS)
    ck    = dedup(os.path.join(rd, "checklist_glm.jsonl"), CK_FIELDS)
    scores = {t: float(r["checklist_score"]) for t, r in ck.items()
              if r.get("checklist_score") not in (None, "")}
    all_ids = sorted(preds)
    mdir = os.path.join(OUT, slug)
    os.makedirs(mdir, exist_ok=True)
    write_jsonl(os.path.join(mdir, "predictions.jsonl"), [preds[t] for t in all_ids])
    write_jsonl(os.path.join(mdir, "checklist.jsonl"),  [ck[t] for t in sorted(ck)])
    full_solve = solve_rate(scores, all_ids)
    froz_solve = solve_rate(scores, FROZEN)
    summ = {
        "model": slug, "role": role, "tools": tools,
        "eval_set": "full_core_657", "judge": "GLM-5.1 checklist", "decoding": "T=0",
        "n_predictions": len(preds), "n_judged": len(scores),
        "full_core_657": {"solve@0.5_pct": full_solve, "avg_score": avg(scores, all_ids)},
        "frozen_core_423": {"solve@0.5_pct": froz_solve, "avg_score": avg(scores, FROZEN)},
    }
    json.dump(summ, open(os.path.join(mdir, "summary.json"), "w"), indent=2)
    manifest.append(summ)
    lb_rows.append([slug, role, "yes" if tools else "no", len(preds), len(scores),
                    full_solve, froz_solve if role not in ("roster",) else "0*"])
    print(f"{slug:18s} {role:9s} n={len(preds):3d} judged={len(scores):3d} "
          f"full657={full_solve} frozen423={froz_solve}")

json.dump(manifest, open(os.path.join(OUT, "MANIFEST.json"), "w"), indent=2)
with open(os.path.join(OUT, "leaderboard.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "role", "tools", "n_predictions", "n_judged",
                "full_core_657_solve@0.5", "frozen_core_423_solve@0.5"])
    w.writerows(lb_rows)

# ---- post-build verification: nothing private survives ---------------------
bad = []
for dp, _, fs in os.walk(OUT):
    for fn in fs:
        p = os.path.join(dp, fn)
        txt = open(p, encoding="utf-8", errors="ignore").read()
        for pat in VERIFY:
            if pat.search(txt):
                bad.append((p, pat.pattern))
if bad:
    print("\n!!! PRIVATE CONTENT SURVIVED — ABORTING:")
    for p, pat in bad[:20]:
        print(f"   {p}  ::  {pat}")
    sys.exit(1)
print(f"\nOK: {len(MODELS)} models packaged, scrub-verified clean -> {OUT}")
