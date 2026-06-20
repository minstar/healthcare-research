#!/usr/bin/env python3
"""Recompute L2 wrong-paper rate on EVALUABLE claims only.

A claim-extraction bug (exp_cite_audit.claim_snippet) left ~11% of stored 'claim' snippets as bare
citation fragments (e.g. ", 2024; PMID: 40138682)."). The judge prompt feeds that snippet verbatim,
so for those the judge had no claim to evaluate and defaulted to 'no' -> spurious wrong-paper.
We report the rate on evaluable claims and the inflation from including fragments. GPU-free, no API.
"""
import json, re
def evaluable(c):
    c=(c or '').strip()
    w=len(re.sub(r'[^a-zA-Z ]',' ',c).split())
    frag = (w<6) or bool(re.match(r'^[\s,.;:\)\(*#\]]*((19|20)\d\d|PMID|NCT|et al|doi|Source)',c,re.I))
    return not frag
def split(rs):
    j=[r for r in rs if r.get('supports') in ('yes','partial','no')]
    ev=[r for r in j if evaluable(r.get('claim'))]; fr=[r for r in j if not evaluable(r.get('claim'))]
    def wp(x): n=sum(1 for r in x if r['supports']=='no'); return n,len(x),round(100*n/len(x),1) if x else 0
    return {'all':wp(j),'evaluable':wp(ev),'fragment':wp(fr),'frag_pct':round(100*len(fr)/len(j),1)}

out={}
# OPUS (independent) full population
op=[json.loads(l) for l in open('results/l2_fullpop.jsonl')]
out['opus_overall']=split(op)
# GLM (primary) from cite_audit
glm=[json.loads(l) for l in open('results/cite_audit.jsonl') if '"supports"' in l]
out['glm_overall']=split(glm)
out['glm_by_model']={m:split([r for r in glm if r.get('model')==m]) for m in ['GLM-5.1','Qwen3.6','DeepSeek-V4']}
json.dump(out, open('results/wrongpaper_evaluable.json','w'), indent=2)
def show(tag,d): print(f"{tag:16s} ALL {d['all'][2]:>5}% (n={d['all'][1]})  EVALUABLE {d['evaluable'][2]:>5}% (n={d['evaluable'][1]})  frag {d['frag_pct']}% @ {d['fragment'][2]}%wp")
show('GLM overall',out['glm_overall']); show('OPUS overall',out['opus_overall'])
for m,d in out['glm_by_model'].items(): show('  '+m,d)
