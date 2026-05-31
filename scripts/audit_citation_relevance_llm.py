"""LLM citation-relevance audit (paper parity: agent-judge reference verification).

The Jaccard relevance metric in validate_gold_answers is uninformative (short relevance
note vs long abstract -> ~0 by construction). This instead fetches each cited PMID's
title+abstract (Europe PMC, reliable) and asks an LLM whether the paper actually SUPPORTS
the stated relevance claim. Reports % supported over a sample.

Usage:
    OPENAI_API_BASE=http://<node>:8000/v1 OPENAI_API_KEY=dummy \
    python scripts/audit_citation_relevance_llm.py \
        --gold data/export/mcp_benchmark_with_gold.jsonl \
        --model glm-5.1 --base http://<node>:8000/v1 --sample 400 \
        --out results/citation_relevance_llm.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import requests
from openai import OpenAI

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
SYS = """You verify a CITATION. Given a paper's title+abstract and a one-sentence claim about \
why it is relevant to a medical question, decide whether the paper actually SUPPORTS that \
claim. Output ONLY JSON: {"supports":"yes|partial|no","why":"<=15 words"}
- yes: the abstract clearly supports the stated relevance.
- partial: related topic but the specific claim is not clearly supported by the abstract.
- no: unrelated, or the claim misrepresents the paper."""

V = {"yes": 1.0, "partial": 0.5, "no": 0.0}


def fetch_abstract(pmid: str) -> dict | None:
    try:
        r = requests.get(EPMC, params={"query": f"EXT_ID:{pmid} AND SRC:MED", "format": "json",
                                       "resultType": "core", "pageSize": 1}, timeout=30)
        res = r.json().get("resultList", {}).get("result", [])
        if not res:
            return None
        x = res[0]
        return {"title": x.get("title", ""), "abstract": (x.get("abstractText") or "")[:1500]}
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--sample", type=int, default=400)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=21)
    args = ap.parse_args()

    cli = OpenAI(base_url=args.base, api_key="dummy")
    # collect (pmid, relevance, source_id) citation records
    cites = []
    for l in open(args.gold):
        d = json.loads(l)
        g = d.get("gold_answer") or {}
        for c in (g.get("key_citations") or []):
            if str(c.get("type", "")).upper() == "PMID" and c.get("id") and c.get("relevance"):
                cites.append({"source_id": str(d.get("source_id", "")), "pmid": str(c["id"]),
                              "relevance": c["relevance"]})
    random.seed(args.seed)
    random.shuffle(cites)
    cites = cites[: args.sample]
    print(f"checking {len(cites)} citations with {args.model} ...")

    def judge(c):
        meta = fetch_abstract(c["pmid"])
        if not meta or not meta["abstract"]:
            return {**c, "supports": "no_abstract", "score": None}
        user = (f"PAPER TITLE: {meta['title']}\nABSTRACT: {meta['abstract']}\n\n"
                f"RELEVANCE CLAIM: {c['relevance']}\n\nDoes the paper support the claim?")
        try:
            resp = cli.chat.completions.create(model=args.model, max_tokens=2500, temperature=0.0,
                messages=[{"role": "system", "content": SYS}, {"role": "user", "content": user}])
            txt = resp.choices[0].message.content or ""
            m = re.search(r"\{.*\}", txt, re.S)
            v = str(json.loads(m.group(0)).get("supports", "")).lower() if m else ""
        except Exception as e:
            return {**c, "supports": "error", "error": str(e)[:80], "score": None}
        return {**c, "supports": v, "score": V.get(v)}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        out = list(ex.map(judge, cites))
    with open(args.out, "w") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

    scored = [o for o in out if o.get("score") is not None]
    dist = collections.Counter(o["supports"] for o in out)
    import statistics as st
    print(f"judged {len(scored)} (no_abstract={dist.get('no_abstract',0)}, error={dist.get('error',0)})")
    if scored:
        print(f"  relevance support score (yes=1/partial=.5/no=0): {st.mean(o['score'] for o in scored):.3f}")
        print(f"  supported (yes): {dist['yes']}/{len(scored)} = {100*dist['yes']/len(scored):.1f}%")
    print(f"  verdicts: {dict(dist)}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
