"""Classify all-pass (model-easy) questions: WHY did all 3 strong models pass?

A "core nanje" should be hard. If all 3 models passed, the question is suspect.
This classifies each into the reason it was easy, so we can quarantine/relabel:
  - vague_aspirational: buzzword/aspirational framing, no sharp answerable target
  - already_resolved: textbook/well-established answer the models can recite
  - broad_goal: legitimate but broad research goal where any plausible synthesis scores
  - genuine_hard: actually a hard open question that models happened to handle well (KEEP)

Uses the served GLM judge (openai/ via litellm), same path as checklist_judge.
"""
from __future__ import annotations
import argparse, json
from concurrent.futures import ThreadPoolExecutor
import litellm

SYS = """You audit whether a biomedical research question is a genuine OPEN research \
problem ("nanje") or only appears so. Three strong models all answered it adequately, \
which is suspicious for a question meant to be unsolved. Classify the REASON. Output ONLY JSON."""

PROMPT = """QUESTION ({type}): {q}

All three strong models scored it as PASS (adequate synthesis). Classify why into ONE label:
- "vague_aspirational": buzzword/aspirational framing (e.g. "realize P4 medicine", "personalized approaches") with no sharp, answerable target — any reasonable essay scores.
- "already_resolved": the core answer is textbook / well-established; models recite known pathophysiology or established practice.
- "broad_goal": a legitimate but very broad research goal where many plausible syntheses all look adequate, so passing is easy regardless of true difficulty.
- "genuine_hard": despite the pass, this is a sharply-scoped genuine open problem the models happened to handle well — KEEP it.

Output JSON ONLY:
  "label": one of the four
  "keep": true only if "genuine_hard", else false
  "reason": one sentence
"""

def classify(rec, model, base):
    try:
        r = litellm.completion(model=model, api_base=base, api_key="dummy",
            messages=[{"role":"system","content":SYS},
                      {"role":"user","content":PROMPT.format(type=rec["type"], q=rec["question"])}],
            max_tokens=2048, temperature=0)
        txt = r.choices[0].message.content or ""
        import re
        m = re.search(r"\{.*\}", txt, re.S)
        out = json.loads(m.group(0)) if m else {"label":"parse_fail","keep":False,"reason":""}
    except Exception as e:
        out = {"label":"error","keep":False,"reason":str(e)[:120]}
    out.update(source_id=rec["source_id"], glm=rec["glm"], qwen=rec["qwen"], v4=rec["v4"],
               type=rec["type"], question=rec["question"])
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="/tmp/allpass51.json")
    ap.add_argument("--model", default="openai/glm-5.1")
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", default="results/allpass_classification.jsonl")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    recs = json.load(open(args.inp))
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(lambda r: classify(r, args.model, args.base), recs))
    import collections
    with open(args.out, "w") as f:
        for r in res: f.write(json.dumps(r, ensure_ascii=False)+"\n")
    c = collections.Counter(r["label"] for r in res)
    print("classified", len(res), "->", dict(c))
    print("KEEP (genuine_hard):", sum(1 for r in res if r.get("keep")))
    print("QUARANTINE:", sum(1 for r in res if not r.get("keep")))

if __name__ == "__main__":
    main()
