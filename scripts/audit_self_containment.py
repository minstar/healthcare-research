"""Self-containment audit (paper parity: ResearchMath-14K reports 67.2%->94.2%).

An independent LLM judges whether each question is understandable on its own — without
the source document — i.e. all abbreviations defined, condition/population/intervention
specified, and the open aspect clear. Reports the % self-contained over a sample.

Usage:
    OPENAI_API_BASE=http://<node>:8000/v1 OPENAI_API_KEY=dummy \
    python scripts/audit_self_containment.py --data data/export/mcp_benchmark_v3.jsonl \
        --model glm-5.1 --base http://<node>:8000/v1 --sample 500 --out results/self_containment.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

SYS = """You assess whether a medical research question is SELF-CONTAINED: understandable \
on its own by a domain expert WITHOUT any source document. Check: (a) all abbreviations \
defined on first use, (b) the condition/population/intervention specified, (c) the open \
question itself is clear and unambiguous, (d) no dangling references ("this study", "the \
above", "as mentioned"). Output ONLY JSON: \
{"self_contained":"yes|partial|no","missing":"<what's missing, or none, <=15 words>"}
- yes: fully standalone. - partial: understandable but missing a definition/context detail.
- no: cannot be understood without the source (undefined refs/abbreviations, vague scope)."""

V = {"yes": 1.0, "partial": 0.5, "no": 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--sample", type=int, default=500)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    cli = OpenAI(base_url=args.base, api_key="dummy")
    rows = [json.loads(l) for l in open(args.data)]
    random.seed(args.seed)
    random.shuffle(rows)
    rows = rows[: args.sample]

    def judge(r):
        q = r.get("self_contained_question") or r.get("original_question", "")
        try:
            resp = cli.chat.completions.create(model=args.model, max_tokens=3000, temperature=0.0,
                messages=[{"role": "system", "content": SYS},
                          {"role": "user", "content": f"QUESTION:\n{q}\n\nAssess self-containment."}])
            txt = resp.choices[0].message.content or ""
            m = re.search(r"\{.*\}", txt, re.S)
            d = json.loads(m.group(0)) if m else {}
        except Exception as e:
            return {"task_id": str(r.get("source_id", "")), "error": str(e)[:120]}
        v = str(d.get("self_contained", "")).lower()
        if v not in V:
            return {"task_id": str(r.get("source_id", "")), "error": "unparsed verdict"}
        return {"task_id": str(r.get("source_id", "")), "track": r.get("corpus_track", ""),
                "self_contained": v, "score": V[v], "missing": d.get("missing", "")[:80]}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        out = [r for r in ex.map(judge, rows) if r]
    with open(args.out, "w") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

    ok = [o for o in out if "score" in o]
    import statistics as st
    dist = collections.Counter(o["self_contained"] for o in ok)
    by_track = collections.defaultdict(list)
    for o in ok:
        by_track[o.get("track", "?")].append(o["score"])
    print(f"audited {len(ok)} (errs {len(out)-len(ok)})")
    print(f"  self-containment score (yes=1/partial=.5/no=0): {st.mean(o['score'] for o in ok):.3f}")
    print(f"  fully self-contained (yes): {dist['yes']}/{len(ok)} = {100*dist['yes']/len(ok):.1f}%")
    print(f"  verdicts: {dict(dist)}")
    for tr, scs in by_track.items():
        print(f"  track {tr}: mean={st.mean(scs):.3f} (n={len(scs)})")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
