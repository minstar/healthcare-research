"""Generate per-question evaluation rubrics (instance-specific checklists).

For each question, an LLM drafts 5-8 concrete, checkable criteria tailored to that
question (must_mention / must_acknowledge / must_ground / must_avoid). The rubric is
generated ONCE per question and frozen, so every judge grades against the SAME
checklist — this is what drives up inter-judge agreement vs vague 0-1 dimensions.

Usage:
    OPENAI_API_BASE=http://<node>:8000/v1 OPENAI_API_KEY=dummy \
    python scripts/gen_rubrics.py --data data/eval_samples/gold_strat_60.jsonl \
        --model glm-5.1 --base http://<node>:8000/v1 --out data/eval_samples/rubrics_60.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

SYS = """You write a QUESTION-SPECIFIC evaluation rubric (checklist) for grading an AI's \
answer to an OPEN medical research question. Output ONLY JSON: \
{"criteria":[{"text":"...","type":"...","weight":1-3}]}.
Criterion types:
- must_mention: a key fact / mechanism / method the answer should include
- must_acknowledge: an uncertainty / gap / open-aspect the answer must correctly flag (critical for OPEN questions)
- must_ground: the answer must cite or use real evidence (PMID / trial / tool results) for its claims
- must_avoid: something that should NOT happen (fabricating a definitive answer to an open question; \
falsely claiming tools failed/returned nothing; unsupported or invented claims)
Write 5-8 concrete, checkable, question-specific criteria. weight 3=critical, 2=important, 1=minor. \
Include at least one must_acknowledge and one must_avoid."""


def gold_blurb(gold) -> str:
    if isinstance(gold, dict):
        return ("KNOWN: " + (gold.get("current_knowledge", "")[:600])
                + " | UNKNOWN: " + (gold.get("unknown_aspects", "")[:400]))
    return str(gold or "")[:600]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    cli = OpenAI(base_url=args.base, api_key="dummy")
    rows = [json.loads(l) for l in open(args.data)]

    def gen(r):
        q = r.get("self_contained_question") or r.get("original_question", "")
        msg = f"QUESTION: {q}\n\nGOLD (known + unknown):\n{gold_blurb(r.get('gold_answer'))}\n\nWrite the rubric."
        try:
            resp = cli.chat.completions.create(model=args.model, max_tokens=3500, temperature=0.2,
                messages=[{"role": "system", "content": SYS}, {"role": "user", "content": msg}])
            txt = resp.choices[0].message.content or ""
            m = re.search(r"\{.*\}", txt, re.S)
            crit = json.loads(m.group(0)).get("criteria", []) if m else []
        except Exception as e:
            crit = []
            print(f"  gen fail {r.get('source_id')}: {e}", file=sys.stderr)
        # normalise + id
        clean = []
        for i, c in enumerate(crit, 1):
            if c.get("text") and c.get("type") in ("must_mention", "must_acknowledge", "must_ground", "must_avoid"):
                clean.append({"id": i, "text": c["text"], "type": c["type"], "weight": int(c.get("weight", 2))})
        return {"task_id": str(r.get("source_id", "")), "question": q, "criteria": clean}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        out = list(ex.map(gen, rows))
    with open(args.out, "w") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    ncrit = [len(o["criteria"]) for o in out]
    print(f"wrote {len(out)} rubrics → {args.out} | criteria/q: "
          f"min {min(ncrit)} avg {sum(ncrit)/len(ncrit):.1f} max {max(ncrit)} | empty {sum(1 for n in ncrit if n==0)}")


if __name__ == "__main__":
    main()
