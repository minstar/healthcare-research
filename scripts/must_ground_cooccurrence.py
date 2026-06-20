#!/usr/bin/env python3
"""Co-occurrence of WRONG-PAPER citations and must_ground grounding failure.

Reviewer question: when an agent cites the wrong paper for a claim, does that
answer ALSO tend to fail the rubric's `must_ground` grounding criterion? If so,
wrong-paper is the provenance signature of an ungrounded claim rather than a
cosmetic footnote slip. If not, wrong-paper is bounded to "provenance integrity".

LINKAGE (all inputs already on disk, no API/GPU):
  - Wrong-paper labels: results/cite_audit.jsonl
        per-citation L2 audit. wrong_paper := (exists is True) AND (supports == 'no').
        Keys: set, model, task_id (e.g. PMID:38345416#0 -- per question), id, supports.
  - must_ground verdict: results/<run>/checklist_uid.jsonl
        per-(task_id, model) checklist, list of {id, v(met/partial/not_met)}.
        Mapped to criterion TYPE via the rubric below. Each task has exactly one
        must_ground criterion -> one grounding outcome per answer.
  - Criterion types: data/eval_samples/rubrics_1969_uid.jsonl
        task_id -> {criterion_id: type}, type in {must_mention, must_ground,
        must_acknowledge, must_avoid}.

JOIN LEVEL: per ANSWER = per (task_id, model). NOT per individual claim -- the
citation audit gives a per-citation wrong-paper label but the grounding verdict
is one-per-answer, so we aggregate citations up to the answer (>=1 wrong-paper
present vs all-support). We restrict the universe to answers that HAVE citations
(i.e. appear in cite_audit) so the "wrong-paper absent" group is "cited, but no
wrong paper" -- an apples-to-apples comparison, not contaminated by zero-citation
answers from a different behavioural regime.

SET: the 1969 evaluation set, the only one with per-model #N-keyed checklist_uid
files that align with the rubric/audit task_id format. Per-model checklist dirs:
  DeepSeek-V4 -> baseline_dsv4_1969, Qwen3.6 -> baseline_qwen_1969,
  GLM-5.1     -> baseline_glm_1969.

GROUNDING FAIL is reported under two definitions:
  strict : not_met            (clearly ungrounded)
  loose  : not_met or partial (anything short of fully grounded)
"""
import json, math
from collections import defaultdict

ROOT = "/data/project/private/minstar/workspace/healthcare-research"
R = f"{ROOT}/results"
D = f"{ROOT}/data"

CHECKLIST_DIR = {
    "DeepSeek-V4": "baseline_dsv4_1969",
    "Qwen3.6":     "baseline_qwen_1969",
    "GLM-5.1":     "baseline_glm_1969",
}
AUDIT_SET = "1969"

# ---- rubric: task_id -> {criterion_id: type} ----
rub = {}
for l in open(f"{D}/eval_samples/rubrics_1969_uid.jsonl"):
    o = json.loads(l)
    rub[o["task_id"]] = {c["id"]: c["type"] for c in o["criteria"]}

# ---- must_ground verdict per (model, task_id) ----
# grounding[(model, task_id)] = 'met' | 'partial' | 'not_met'
grounding = {}
for model, dirn in CHECKLIST_DIR.items():
    for l in open(f"{R}/{dirn}/checklist_uid.jsonl"):
        o = json.loads(l)
        if "verdicts" not in o:
            continue
        tid = o["task_id"]
        rb = rub.get(tid)
        if not rb:
            continue
        for vd in o["verdicts"]:
            if rb.get(vd["id"]) == "must_ground":
                grounding[(model, tid)] = vd["v"]
                break  # exactly one must_ground per task

# ---- wrong-paper presence per (model, task_id) over the 1969 audit set ----
# answers that have >=1 audited (real) citation are the universe.
has_cite = set()              # (model, task_id) with >=1 citation in audit
wrong_present = set()         # (model, task_id) with >=1 wrong-paper citation
for l in open(f"{R}/cite_audit.jsonl"):
    o = json.loads(l)
    if o.get("set") != AUDIT_SET:
        continue
    key = (o["model"], o["task_id"])
    has_cite.add(key)
    if o.get("exists") is True and o.get("supports") == "no":
        wrong_present.add(key)

# ---- assemble jointly-labelled answers ----
joint = [k for k in has_cite if k in grounding]


def contingency(keys, fail_set):
    """2x2: wrong-paper (present/absent) x must_ground (fail/pass)."""
    a = b = c = d = 0  # a=wp&fail b=wp&pass c=nowp&fail d=nowp&pass
    for k in keys:
        wp = k in wrong_present
        fail = grounding[k] in fail_set
        if wp and fail: a += 1
        elif wp and not fail: b += 1
        elif not wp and fail: c += 1
        else: d += 1
    return a, b, c, d


def stats(a, b, c, d):
    n_wp, n_nowp = a + b, c + d
    p_fail_wp = a / n_wp if n_wp else None
    p_fail_nowp = c / n_nowp if n_nowp else None
    # risk ratio with Katz log CI
    rr = rr_lo = rr_hi = None
    if p_fail_wp is not None and p_fail_nowp not in (None, 0) and a > 0 and c > 0:
        rr = p_fail_wp / p_fail_nowp
        se = math.sqrt(1/a - 1/n_wp + 1/c - 1/n_nowp)
        rr_lo = math.exp(math.log(rr) - 1.96 * se)
        rr_hi = math.exp(math.log(rr) + 1.96 * se)
    # odds ratio (Haldane-Anscombe 0.5 correction if any zero cell)
    aa, bb, cc, dd = a, b, c, d
    if 0 in (a, b, c, d):
        aa, bb, cc, dd = a + .5, b + .5, c + .5, d + .5
    orr = (aa * dd) / (bb * cc)
    se_or = math.sqrt(1/aa + 1/bb + 1/cc + 1/dd)
    or_lo = math.exp(math.log(orr) - 1.96 * se_or)
    or_hi = math.exp(math.log(orr) + 1.96 * se_or)
    return {
        "contingency_2x2": {
            "wrong_paper_present__must_ground_FAIL": a,
            "wrong_paper_present__must_ground_PASS": b,
            "wrong_paper_absent__must_ground_FAIL": c,
            "wrong_paper_absent__must_ground_PASS": d,
        },
        "n_wrong_paper_present": n_wp,
        "n_wrong_paper_absent": n_nowp,
        "P(must_ground_fail | wrong_paper_present)": round(p_fail_wp, 4) if p_fail_wp is not None else None,
        "P(must_ground_fail | all_support)":          round(p_fail_nowp, 4) if p_fail_nowp is not None else None,
        "risk_ratio": round(rr, 3) if rr else None,
        "risk_ratio_95ci": [round(rr_lo, 3), round(rr_hi, 3)] if rr else None,
        "odds_ratio": round(orr, 3),
        "odds_ratio_95ci": [round(or_lo, 3), round(or_hi, 3)],
        "haldane_correction_applied": 0 in (a, b, c, d),
    }


FAIL_DEFS = {"strict_not_met": {"not_met"},
             "loose_not_met_or_partial": {"not_met", "partial"}}

out = {
    "linkage": {
        "join_level": "per answer = per (task_id, model); NOT per claim",
        "wrong_paper_def": "cite_audit: exists is True AND supports=='no'",
        "must_ground_source": "rubrics_1969_uid.jsonl (criterion type) x <run>/checklist_uid.jsonl (verdict)",
        "universe": "answers with >=1 audited citation (present in cite_audit, set=1969) AND a must_ground verdict",
        "set": AUDIT_SET,
        "checklist_dirs": CHECKLIST_DIR,
    },
    "counts": {
        "answers_with_citations_audited": len(has_cite),
        "of_those_with_wrong_paper": len(has_cite & wrong_present),
        "answers_with_must_ground_verdict": len(grounding),
        "jointly_labelled_answers (final n)": len(joint),
    },
    "per_model_joint_n": {},
    "overall": {},
    "per_model": {},
}

# per-model joint n + per-model rubric-has-must_ground sanity
pm = defaultdict(list)
for k in joint:
    pm[k[0]].append(k)
for m, ks in pm.items():
    out["per_model_joint_n"][m] = len(ks)

for fdef, fset in FAIL_DEFS.items():
    a, b, c, d = contingency(joint, fset)
    out["overall"][fdef] = stats(a, b, c, d)

for m, ks in pm.items():
    out["per_model"][m] = {}
    for fdef, fset in FAIL_DEFS.items():
        a, b, c, d = contingency(ks, fset)
        out["per_model"][m][fdef] = stats(a, b, c, d)

# also report grounding verdict marginal among joint answers
from collections import Counter
gd = Counter(grounding[k] for k in joint)
out["grounding_verdict_marginal_among_joint"] = dict(gd)

json.dump(out, open(f"{R}/must_ground_cooccurrence.json", "w"), indent=2)

# ---- readable report ----
print("=" * 72)
print("must_ground x wrong-paper co-occurrence  (set 1969, per-answer join)")
print("=" * 72)
print(f"answers with audited citations : {out['counts']['answers_with_citations_audited']}")
print(f"  ... with >=1 wrong-paper     : {out['counts']['of_those_with_wrong_paper']}")
print(f"jointly-labelled (final n)     : {out['counts']['jointly_labelled_answers (final n)']}")
print(f"per-model joint n              : {out['per_model_joint_n']}")
print(f"grounding marginal (joint)     : {dict(gd)}")
for fdef in FAIL_DEFS:
    s = out["overall"][fdef]
    cg = s["contingency_2x2"]
    print(f"\n--- OVERALL, fail def = {fdef} ---")
    print(f"                      must_ground FAIL   PASS")
    print(f"  wrong-paper present {cg['wrong_paper_present__must_ground_FAIL']:>10d} {cg['wrong_paper_present__must_ground_PASS']:>6d}")
    print(f"  wrong-paper absent  {cg['wrong_paper_absent__must_ground_FAIL']:>10d} {cg['wrong_paper_absent__must_ground_PASS']:>6d}")
    print(f"  P(fail|wrong-paper)={s['P(must_ground_fail | wrong_paper_present)']}  "
          f"P(fail|all-support)={s['P(must_ground_fail | all_support)']}")
    print(f"  risk_ratio={s['risk_ratio']} ci={s['risk_ratio_95ci']}  "
          f"odds_ratio={s['odds_ratio']} ci={s['odds_ratio_95ci']}")
print("\nwrote results/must_ground_cooccurrence.json")
