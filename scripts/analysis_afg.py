#!/usr/bin/env python3
"""Re-aggregation analyses A, F, G (no new model runs).

A: two-judge L2 wrong-paper breakdown (GLM-5.1 judge vs independent Opus-4.7 judge,
   title+abstract full-population and full-text), per model + overall.
F: per-domain breakdown of headline numbers (wrong-paper, robust-core composition,
   frontier pass@0.5, GLM-5.1 zero-tool collapse).
G: no-tool ablation criterion-type decomposition for GLM-5.1 (with-tools vs no-tools),
   testing whether tool-parity is driven by abstention-credit criteria.

All inputs already on disk under results/ and data/. Outputs results/analysis_afg.json.
"""
import json, glob, math
from collections import defaultdict, Counter

ROOT = "/data/project/private/minstar/workspace/healthcare-research"
R = f"{ROOT}/results"
D = f"{ROOT}/data"

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (100*p, 100*max(0, c-h), 100*min(1, c+h))

def strip(x): return x.rsplit('#', 1)[0]

# ---- domain map ----
dom = {}
for l in open(f"{D}/export/mcp_benchmark_v3.4.jsonl"):
    o = json.loads(l); dom[o['source_id']] = o.get('taxonomy_l1')
for f in glob.glob(f"{D}/gold_answers/chunk_*.input.jsonl"):
    for l in open(f):
        o = json.loads(l); dom.setdefault(o['source_id'], o.get('taxonomy_l1'))
# fold tiny straggler domains into 'Other'
MAIN = {"Clinical Medicine","Neuroscience & Psychiatry","Public Health & Epidemiology",
        "Oncology","Surgical Sciences","Infectious Disease & Immunology","Cardiovascular Medicine",
        "Pharmacology & Drug Discovery","Genomics & Precision Medicine","Medical AI & Informatics",
        "Rare & Orphan Diseases","Other"}
def dm(sid):
    t = dom.get(sid);
    return t if t in MAIN else "Other"

out = {}

# ===== A: two-judge L2 =====
def judge_table(fn):
    by = defaultdict(Counter); ov = Counter()
    for l in open(fn):
        o = json.loads(l); by[o['model']][o['supports']] += 1; ov[o['supports']] += 1
    def row(c):
        judged = c['yes']+c['partial']+c['no']
        return {"judged": judged, "wrong_paper": c['no'],
                "wrong_paper_pct": round(100*c['no']/judged,1) if judged else None,
                "ci": [round(x,1) for x in wilson(c['no'], judged)]}
    res = {m: row(by[m]) for m in by}; res["OVERALL"] = row(ov); res["errors"] = ov.get('error',0)
    return res
# GLM judge (from existing summary)
glm_sum = json.load(open(f"{R}/cite_audit_summary.json"))
A = {"glm_judge": {m: {"judged": v["L2_judged"], "wrong_paper_pct": v["wrong_paper_pct"]}
                   for m, v in glm_sum["by_model"].items()},
     "glm_overall_pct": glm_sum["overall"]["wrong_paper_pct"],
     "glm_overall_judged": glm_sum["overall"]["real_citations_judged"],
     "opus_fullpop": judge_table(f"{R}/l2_fullpop.jsonl"),
     "opus_fulltext": judge_table(f"{R}/l2_fulltext.jsonl")}
out["A_two_judge_L2"] = A

# ===== F(i): per-domain wrong-paper (Opus full-pop, has task_id) =====
dwp = defaultdict(Counter)
for l in open(f"{R}/l2_fullpop.jsonl"):
    o = json.loads(l)
    if o['supports'] == 'error': continue
    dwp[dm(strip(o['task_id']))][o['supports']] += 1
F_dom_wp = {}
for d_, c in dwp.items():
    j = c['yes']+c['partial']+c['no']
    F_dom_wp[d_] = {"judged": j, "wrong_paper": c['no'], "pct": round(100*c['no']/j,1) if j else None,
                    "ci": [round(x,1) for x in wilson(c['no'], j)]}

# ===== F(ii): robust-core composition by domain =====
rc = json.load(open(f"{R}/robust_core_ids.json"))
comp = Counter(dm(strip(x)) for x in rc)
F_comp = {d_: {"n": n, "pct": round(100*n/len(rc),1)} for d_, n in comp.most_common()}

# ===== F(iii): frontier pass@0.5 per domain (robust core, GLM checklist judge) =====
FRONT = {"Gemini-3-Pro":"api_gemini3pro_robust","Opus-4.7":"api_opus47_robust","GPT-5.5":"api_gpt55_robust"}
F_front = {}
for model, dirn in FRONT.items():
    bd = defaultdict(lambda: [0,0])  # domain -> [solved, total]
    for l in open(f"{R}/{dirn}/checklist_glm.jsonl"):
        o = json.loads(l)
        sc = o.get('checklist_score')
        if sc is None: continue
        d_ = dm(strip(o['task_id']))
        bd[d_][1] += 1
        if sc >= 0.5: bd[d_][0] += 1
    F_front[model] = {d_: {"solved": s, "n": t, "pct": round(100*s/t,1) if t else None}
                      for d_, (s, t) in bd.items()}

# ===== F(iv): GLM-5.1 zero-tool collapse per domain (1969 set) =====
zt = defaultdict(lambda: [0,0])
for l in open(f"{R}/baseline_glm_1969/traces.jsonl"):
    o = json.loads(l); d_ = dm(strip(o['task_id']))
    zt[d_][1] += 1
    if len(o.get('tool_calls') or []) == 0: zt[d_][0] += 1
F_collapse = {d_: {"zero_tool": z, "n": t, "pct": round(100*z/t,1) if t else None}
              for d_, (z, t) in zt.items()}

out["F_per_domain"] = {"wrong_paper_opus": F_dom_wp, "robust_core_composition": F_comp,
                       "frontier_pass@0.5": F_front, "glm51_zero_tool_collapse": F_collapse}

# ===== G: no-tool ablation criterion-type decomposition (GLM-5.1) =====
# rubric: task_id -> {id: (type, weight)}
rub = {}
for l in open(f"{D}/eval_samples/rubrics_1969_uid.jsonl"):
    o = json.loads(l); rub[o['task_id']] = {c['id']: (c['type'], c.get('weight',1)) for c in o['criteria']}
VMAP = {"met":1.0, "partial":0.5, "not_met":0.0}
def decomp(checklist_fn):
    # type -> [weighted_score_sum, weight_sum]
    agg = defaultdict(lambda: [0.0, 0.0]); tasks = set()
    for l in open(checklist_fn):
        o = json.loads(l)
        if 'verdicts' not in o: continue
        tid = o['task_id']; tasks.add(tid)
        rb = rub.get(tid)
        if not rb: continue
        for vd in o['verdicts']:
            tw = rb.get(vd['id'])
            if not tw: continue
            typ, w = tw
            agg[typ][0] += w * VMAP.get(vd['v'], 0.0); agg[typ][1] += w
    return {t: {"weighted_sat_pct": round(100*s/w,1) if w else None, "weight": round(w,1)}
            for t,(s,w) in agg.items()}, tasks
tool_d, tool_tasks = decomp(f"{R}/baseline_glm_1969/checklist_uid.jsonl")
notool_d, notool_tasks = decomp(f"{R}/baseline_notool_glm51/checklist_glm.jsonl")
common = tool_tasks & notool_tasks
# restrict to common tasks for a matched comparison
def decomp_on(checklist_fn, keep):
    agg = defaultdict(lambda: [0.0,0.0])
    for l in open(checklist_fn):
        o = json.loads(l)
        if 'verdicts' not in o: continue
        tid=o['task_id']
        if tid not in keep: continue
        rb = rub.get(tid)
        if not rb: continue
        for vd in o['verdicts']:
            tw = rb.get(vd['id'])
            if not tw: continue
            typ,w = tw; agg[typ][0]+=w*VMAP.get(vd['v'],0.0); agg[typ][1]+=w
    return {t:round(100*s/w,1) if w else None for t,(s,w) in agg.items()}
G = {"n_common_tasks": len(common),
     "with_tools": decomp_on(f"{R}/baseline_glm_1969/checklist_uid.jsonl", common),
     "no_tools":   decomp_on(f"{R}/baseline_notool_glm51/checklist_glm.jsonl", common)}
G["delta_notool_minus_tool"] = {t: round((G["no_tools"].get(t) or 0)-(G["with_tools"].get(t) or 0),1)
                                for t in set(G["with_tools"])|set(G["no_tools"])}
out["G_notool_criterion_decomp"] = G

json.dump(out, open(f"{R}/analysis_afg.json","w"), indent=2)
print("wrote results/analysis_afg.json\n")

# ---- readable report ----
print("="*70); print("A. TWO-JUDGE L2 WRONG-PAPER"); print("="*70)
print(f"{'model':14s} {'GLM-judge':>14s} {'Opus full-pop':>18s} {'Opus full-text':>16s}")
for m in ["GLM-5.1","Qwen3.6","DeepSeek-V4"]:
    g = A['glm_judge'].get(m,{}); fp=A['opus_fullpop'].get(m,{}); ft=A['opus_fulltext'].get(m,{})
    print(f"{m:14s} {str(g.get('wrong_paper_pct'))+'% (n='+str(g.get('judged'))+')':>14s} "
          f"{str(fp.get('wrong_paper_pct'))+'% (n='+str(fp.get('judged'))+')':>18s} "
          f"{str(ft.get('wrong_paper_pct'))+'% (n='+str(ft.get('judged'))+')':>16s}")
ov_fp=A['opus_fullpop']['OVERALL']; ov_ft=A['opus_fulltext']['OVERALL']
print(f"{'OVERALL':14s} {str(A['glm_overall_pct'])+'% (n='+str(A['glm_overall_judged'])+')':>14s} "
      f"{str(ov_fp['wrong_paper_pct'])+'% (n='+str(ov_fp['judged'])+')':>18s} "
      f"{str(ov_ft['wrong_paper_pct'])+'% (n='+str(ov_ft['judged'])+')':>16s}")

print("\n"+"="*70); print("F. PER-DOMAIN BREAKDOWN"); print("="*70)
print("\n(i) Wrong-paper % per domain [Opus full-pop judge]:")
for d_ in sorted(F_dom_wp, key=lambda x:-F_dom_wp[x]['pct'] if F_dom_wp[x]['pct'] else 0):
    v=F_dom_wp[d_]; print(f"  {d_:34s} {v['pct']:5.1f}%  [{v['ci'][1]:.1f},{v['ci'][2]:.1f}]  n={v['judged']}")
print("\n(ii) Robust-core (423) composition by domain:")
for d_,v in F_comp.items(): print(f"  {d_:34s} {v['n']:4d}  ({v['pct']:.1f}%)")
print("\n(iii) Frontier pass@0.5 per domain (robust core):")
doms = sorted(F_comp, key=lambda x:-F_comp[x]['n'])
print(f"  {'domain':34s} {'Gem':>6s} {'Opus':>6s} {'GPT5.5':>7s}  n")
for d_ in doms:
    g=F_front['Gemini-3-Pro'].get(d_,{}); o=F_front['Opus-4.7'].get(d_,{}); p=F_front['GPT-5.5'].get(d_,{})
    n=g.get('n','-')
    print(f"  {d_:34s} {str(g.get('pct')):>6s} {str(o.get('pct')):>6s} {str(p.get('pct')):>7s}  {n}")
print("\n(iv) GLM-5.1 zero-tool collapse per domain (1969):")
for d_ in sorted(F_collapse, key=lambda x:-(F_collapse[x]['pct'] or 0)):
    v=F_collapse[d_]; print(f"  {d_:34s} {v['pct']:5.1f}%  n={v['n']}")

print("\n"+"="*70); print("G. NO-TOOL ABLATION CRITERION-TYPE DECOMP (GLM-5.1)"); print("="*70)
print(f"matched common tasks: {G['n_common_tasks']}")
print(f"{'criterion type':20s} {'with-tools':>12s} {'no-tools':>10s} {'delta':>8s}")
for t in ["must_mention","must_ground","must_acknowledge","must_avoid"]:
    wt=G['with_tools'].get(t); nt=G['no_tools'].get(t); de=G['delta_notool_minus_tool'].get(t)
    print(f"{t:20s} {str(wt)+'%':>12s} {str(nt)+'%':>10s} {str(de):>8s}")
