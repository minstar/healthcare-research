#!/usr/bin/env python3
"""Close the LENIENCY loophole in must_ground x wrong-paper independence (paper 5.2).

Reviewer loophole: must_ground is lenient -- it may credit grounding merely
because an answer cites a REAL paper, even if that real paper is the WRONG paper.
If so, the observed independence (RR 1.07, CI 0.88-1.31) could be partly
DEFINITIONAL (a wrong-paper citation mechanically buys a must_ground pass) rather
than a real reasoning/grounding dissociation.

We must test this WITHOUT introducing the opposite circularity (forcing every
wrong-paper answer to fail grounding mechanically manufactures correlation).

REUSES the EXACT join of scripts/must_ground_cooccurrence.py:
  - universe  : (model, task_id) answers with >=1 audited citation (cite_audit,
                set=1969) AND a must_ground verdict  -> n = 879
  - wrong_paper(answer) := >=1 citation with exists is True AND supports=='no'
  - must_ground verdict : rubrics_1969_uid.jsonl (criterion type) x
                          <run>/checklist_uid.jsonl (verdict v in met/partial/not_met)
  - checklist dirs      : DeepSeek-V4->baseline_dsv4_1969, Qwen3.6->baseline_qwen_1969,
                          GLM-5.1->baseline_glm_1969

NEW per-answer citation features (joined on the SAME (model, task_id) key):
  - n_cite_audited(answer)      : # audited citations on that answer
  - has_support(answer)         : >=1 citation with supports in {yes, partial}
                                  (a genuinely L2-supporting citation)
  - support label availability  : PMID/NCT citations with exists True may have
                                  supports==None (not L2-judged). Those are NOT
                                  counted as supporting (can't prove support) but
                                  are counted toward n_cite_audited. We report how
                                  many answers have NO support-labelled citation at
                                  all, since for those 'has_support' is uninformative.

DIAGNOSTIC 1 (leniency, cleanest test):
  Among answers that (i) carry >=1 wrong-paper citation AND (ii) PASS must_ground
  (v in {met, partial}; also reported for v==met only), what fraction ALSO carry a
  genuinely-supporting citation? If most do, the must_ground pass was NOT bought by
  the wrong-paper citation -> leniency cannot explain the null -> dissociation REAL.
  "Bought" answers = must_ground PASS + wrong-paper present + NO supporting citation.

DIAGNOSTIC 2 (strict, non-circular re-run):
  GENUINELY grounded := must_ground met AND has_support. grounding FAIL := not that.
  To avoid the circularity (a wrong-paper citation being the answer's SOLE citation,
  so its grounding pass is trivially attributable to it), RESTRICT to answers with
  >=2 audited citations. Report 2x2 + RR/OR/CI/n on that subset AND on the full set
  (full set clearly flagged as susceptible to circularity).

DIAGNOSTIC 3: restate the original lenient result for reference.
"""
import json, math
from collections import defaultdict, Counter

ROOT = "/data/project/private/minstar/workspace/healthcare-research"
R = f"{ROOT}/results"
D = f"{ROOT}/data"

CHECKLIST_DIR = {
    "DeepSeek-V4": "baseline_dsv4_1969",
    "Qwen3.6":     "baseline_qwen_1969",
    "GLM-5.1":     "baseline_glm_1969",
}
AUDIT_SET = "1969"
SUPPORTING = {"yes", "partial"}   # genuinely L2-supports (also report yes-only)

# ---- rubric: task_id -> {criterion_id: type} ----
rub = {}
for l in open(f"{D}/eval_samples/rubrics_1969_uid.jsonl"):
    o = json.loads(l)
    rub[o["task_id"]] = {c["id"]: c["type"] for c in o["criteria"]}

# ---- must_ground verdict per (model, task_id) ----
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
                break

# ---- per-answer citation aggregation over the 1969 audit set ----
has_cite       = set()                 # (model, task_id) with >=1 audited citation
wrong_present  = set()                 # >=1 wrong-paper citation
n_cite         = Counter()             # # audited citations
n_support      = Counter()             # # supporting (yes/partial) citations
n_support_yes  = Counter()             # # yes-only supporting citations
n_lbl          = Counter()             # # citations with a support label at all
for l in open(f"{R}/cite_audit.jsonl"):
    o = json.loads(l)
    if o.get("set") != AUDIT_SET:
        continue
    key = (o["model"], o["task_id"])
    has_cite.add(key)
    n_cite[key] += 1
    sup = o.get("supports")
    if sup in ("yes", "partial", "no", "no_abstract"):
        n_lbl[key] += 1
    if o.get("exists") is True and sup == "no":
        wrong_present.add(key)
    if sup in SUPPORTING:
        n_support[key] += 1
    if sup == "yes":
        n_support_yes[key] += 1

joint = [k for k in has_cite if k in grounding]   # n = 879 (matches prior)

def has_support(k):       return n_support[k] >= 1
def has_support_yes(k):   return n_support_yes[k] >= 1

# ============================================================
# DIAGNOSTIC 1 -- leniency
# ============================================================
# PASS def for must_ground: lenient pass = v in {met, partial} (mirrors original
# 'loose fail' = not_met/partial, so PASS-loose = met only; but for "did the answer
# get CREDITED grounding" the relevant lenient pass is anything that is NOT a clear
# fail). We report both: pass_any = met or partial ; pass_met = met only.
def leniency(pass_pred, sup_pred):
    # subset: wrong-paper present AND must_ground PASS
    subset = [k for k in joint if k in wrong_present and pass_pred(grounding[k])]
    with_support = [k for k in subset if sup_pred(k)]
    bought      = [k for k in subset if not sup_pred(k)]            # only wrong-paper / no support
    # of the 'bought', how many literally have NO supporting AND the wrong-paper is
    # plausibly load-bearing -- split by citation count
    bought_sole_cite = [k for k in bought if n_cite[k] == 1]
    return {
        "n_wrongpaper_AND_mustground_PASS": len(subset),
        "n_with_genuine_support":           len(with_support),
        "frac_with_genuine_support":        round(len(with_support)/len(subset), 4) if subset else None,
        "n_bought_no_support":              len(bought),
        "frac_bought_no_support":           round(len(bought)/len(subset), 4) if subset else None,
        "n_bought_AND_sole_citation":       len(bought_sole_cite),
    }

pass_any = lambda v: v in ("met", "partial")
pass_met = lambda v: v == "met"

diag1 = {
    "definition": "support = >=1 citation with supports in {yes,partial}; bought = wrong-paper present + must_ground PASS + NO supporting citation",
    "pass=met_or_partial__support=yes_or_partial": leniency(pass_any, has_support),
    "pass=met_only__support=yes_or_partial":        leniency(pass_met, has_support),
    "pass=met_or_partial__support=yes_only":        leniency(pass_any, has_support_yes),
    "pass=met_only__support=yes_only":              leniency(pass_met, has_support_yes),
}

# ============================================================
# DIAGNOSTIC 2 -- strict, non-circular re-run
# ============================================================
# GENUINELY grounded := must_ground met AND has_support. FAIL := not genuinely grounded.
def genuine_fail(k, sup_pred, require_met):
    g = grounding[k]
    met_ok = (g == "met") if require_met else (g in ("met", "partial"))
    return not (met_ok and sup_pred(k))

def contingency(keys, fail_pred):
    a = b = c = d = 0  # a=wp&fail b=wp&pass c=nowp&fail d=nowp&pass
    for k in keys:
        wp = k in wrong_present
        f = fail_pred(k)
        if wp and f: a += 1
        elif wp and not f: b += 1
        elif (not wp) and f: c += 1
        else: d += 1
    return a, b, c, d

def stats(a, b, c, d):
    n_wp, n_nowp = a + b, c + d
    p_fail_wp   = a / n_wp   if n_wp   else None
    p_fail_nowp = c / n_nowp if n_nowp else None
    rr = rr_lo = rr_hi = None
    if p_fail_wp is not None and p_fail_nowp not in (None, 0) and a > 0 and c > 0:
        rr = p_fail_wp / p_fail_nowp
        se = math.sqrt(1/a - 1/n_wp + 1/c - 1/n_nowp)
        rr_lo = math.exp(math.log(rr) - 1.96*se)
        rr_hi = math.exp(math.log(rr) + 1.96*se)
    aa, bb, cc, dd = a, b, c, d
    hc = 0 in (a, b, c, d)
    if hc:
        aa, bb, cc, dd = a+.5, b+.5, c+.5, d+.5
    orr = (aa*dd)/(bb*cc)
    se_or = math.sqrt(1/aa+1/bb+1/cc+1/dd)
    or_lo = math.exp(math.log(orr) - 1.96*se_or)
    or_hi = math.exp(math.log(orr) + 1.96*se_or)
    return {
        "contingency_2x2": {
            "wrong_paper_present__grounding_FAIL": a,
            "wrong_paper_present__grounding_PASS": b,
            "wrong_paper_absent__grounding_FAIL":  c,
            "wrong_paper_absent__grounding_PASS":  d,
        },
        "n": a+b+c+d,
        "n_wrong_paper_present": n_wp,
        "n_wrong_paper_absent":  n_nowp,
        "P(grounding_fail | wrong_paper_present)": round(p_fail_wp, 4)   if p_fail_wp   is not None else None,
        "P(grounding_fail | all_support)":          round(p_fail_nowp, 4) if p_fail_nowp is not None else None,
        "risk_ratio":        round(rr, 3) if rr else None,
        "risk_ratio_95ci":   [round(rr_lo, 3), round(rr_hi, 3)] if rr else None,
        "odds_ratio":        round(orr, 3),
        "odds_ratio_95ci":   [round(or_lo, 3), round(or_hi, 3)],
        "haldane_correction_applied": hc,
    }

full      = joint
multi_cite = [k for k in joint if n_cite[k] >= 2]   # >=2 citations: wrong-paper not sole

# strict grounding-fail predicate (genuine = met AND has support yes/partial)
strict_fail = lambda k: genuine_fail(k, has_support, require_met=True)

_cite_dist = Counter(n_cite[k] for k in joint)
diag2 = {
    "grounding_def": "GENUINELY grounded = must_ground 'met' AND >=1 citation supports in {yes,partial}; FAIL otherwise",
    "n_full": len(full),
    "n_multicite(>=2 cites)": len(multi_cite),
    "citation_count_distribution_in_joint": {str(c): _cite_dist[c] for c in sorted(_cite_dist)},
    "min_citations_per_answer_in_joint": min(n_cite[k] for k in joint),
    "NOTE_multicite_equals_full": (len(multi_cite) == len(full)),
    "NOTE": ("EVERY answer in the joint universe has >=2 audited citations (min=%d), "
             "so the multi-cite subset == full set. The circularity the >=2 restriction "
             "guards against -- a wrong-paper citation being the answer's SOLE citation, "
             "trivially buying a grounding pass -- is STRUCTURALLY IMPOSSIBLE in this universe. "
             "The full-set result is therefore NOT susceptible to that specific circularity here."
             ) % min(n_cite[k] for k in joint),
    "FULL_SET (CIRCULARITY-SUSCEPTIBLE -- shown for comparison only)": stats(*contingency(full, strict_fail)),
    "MULTICITE_SUBSET (non-circular, primary)":                       stats(*contingency(multi_cite, strict_fail)),
}

# ============================================================
# DIAGNOSTIC 3 -- restate original lenient result
# ============================================================
orig = json.load(open(f"{R}/must_ground_cooccurrence.json"))
diag3 = {
    "source": "results/must_ground_cooccurrence.json",
    "final_n": orig["counts"]["jointly_labelled_answers (final n)"],
    "loose_not_met_or_partial": orig["overall"]["loose_not_met_or_partial"],
    "strict_not_met":           orig["overall"]["strict_not_met"],
}

# ---- support-label availability caveat ----
support_label_caveat = {
    "answers_in_joint": len(joint),
    "answers_with_no_support_labelled_citation": sum(1 for k in joint if n_lbl[k] == 0),
    "note": "for answers with no L2 support-labelled citation, has_support is uninformative (cannot prove support); they are counted as has_support=False",
    "wrongpaper_answers_in_joint": sum(1 for k in joint if k in wrong_present),
}

out = {
    "linkage": {
        "join_level": "per answer = per (task_id, model); citation features aggregated to answer",
        "universe": "answers with >=1 audited citation (cite_audit set=1969) AND must_ground verdict",
        "final_n": len(joint),
        "wrong_paper_def": "exists is True AND supports=='no'",
        "supporting_citation_def": "supports in {yes,partial} (yes-only variant also reported)",
    },
    "support_label_caveat": support_label_caveat,
    "diagnostic_1_leniency": diag1,
    "diagnostic_2_strict_rerun": diag2,
    "diagnostic_3_original_lenient_result": diag3,
}
json.dump(out, open(f"{R}/must_ground_strict.json", "w"), indent=2)

# ---------------- readable report ----------------
def pr(s):
    print(s)

pr("="*74)
pr("must_ground LENIENCY loophole test  (set 1969, per-answer join, n=%d)" % len(joint))
pr("="*74)
pr("\nSupport-label availability:")
pr(f"  answers in joint                         : {support_label_caveat['answers_in_joint']}")
pr(f"  ... with NO support-labelled citation    : {support_label_caveat['answers_with_no_support_labelled_citation']}")
pr(f"  ... wrong-paper answers in joint         : {support_label_caveat['wrongpaper_answers_in_joint']}")

pr("\n--- DIAGNOSTIC 1: leniency ---")
for k, v in diag1.items():
    if k == "definition": continue
    pr(f"  [{k}]")
    pr(f"    wrong-paper & must_ground PASS         : {v['n_wrongpaper_AND_mustground_PASS']}")
    pr(f"    ...with genuine support                : {v['n_with_genuine_support']}  ({v['frac_with_genuine_support']})")
    pr(f"    ...BOUGHT (no support)                 : {v['n_bought_no_support']}  ({v['frac_bought_no_support']})")
    pr(f"    ...bought AND wrong-paper sole citation: {v['n_bought_AND_sole_citation']}")

pr("\n--- DIAGNOSTIC 2: strict (genuine grounding) re-run ---")
pr(f"  n_full={diag2['n_full']}  n_multicite(>=2)={diag2['n_multicite(>=2 cites)']}")
for tag in ["FULL_SET (CIRCULARITY-SUSCEPTIBLE -- shown for comparison only)",
            "MULTICITE_SUBSET (non-circular, primary)"]:
    s = diag2[tag]; cg = s["contingency_2x2"]
    pr(f"\n  {tag}")
    pr(f"                        grounding FAIL   PASS")
    pr(f"    wrong-paper present {cg['wrong_paper_present__grounding_FAIL']:>10d} {cg['wrong_paper_present__grounding_PASS']:>6d}")
    pr(f"    wrong-paper absent  {cg['wrong_paper_absent__grounding_FAIL']:>10d} {cg['wrong_paper_absent__grounding_PASS']:>6d}")
    pr(f"    n={s['n']}  P(fail|wp)={s['P(grounding_fail | wrong_paper_present)']}  P(fail|nowp)={s['P(grounding_fail | all_support)']}")
    pr(f"    RR={s['risk_ratio']} ci={s['risk_ratio_95ci']}  OR={s['odds_ratio']} ci={s['odds_ratio_95ci']}")

pr("\n--- DIAGNOSTIC 3: original lenient result (reference) ---")
o = diag3["loose_not_met_or_partial"]
pr(f"  loose: RR={o['risk_ratio']} ci={o['risk_ratio_95ci']}  OR={o['odds_ratio']} ci={o['odds_ratio_95ci']}  n={diag3['final_n']}")
o = diag3["strict_not_met"]
pr(f"  strict(not_met): RR={o['risk_ratio']} ci={o['risk_ratio_95ci']}  OR={o['odds_ratio']} ci={o['odds_ratio_95ci']}")
pr("\nwrote results/must_ground_strict.json")
