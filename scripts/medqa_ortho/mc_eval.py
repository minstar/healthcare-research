#!/usr/bin/env python3
"""Closed-form multiple-choice eval over an OpenAI-compatible endpoint.

Scores a served model on MedQA / PubMedQA / MedMCQA (data/eval_samples/closedform_mc.jsonl)
by exact-match on the extracted answer letter. No LLM judge — deterministic grading.
Matches OpenBioRQ decoding (T=0) so the accuracy is comparable to the same model's
robust-core pass@0.5.

Usage:
  python mc_eval.py --base http://NODE:8000/v1 --model glm-5.1 --tag roster_glm51 \
      [--datasets MedQA,PubMedQA,MedMCQA] [--limit N] [--workers 16] [--api-key dummy]
Outputs results/medqa_ortho/<tag>.jsonl (per-question) and prints a summary.
"""
import argparse, asyncio, json, os, re, sys
from collections import defaultdict
from openai import AsyncOpenAI

ROOT = "/data/project/private/minstar/workspace/healthcare-research"
DATA = f"{ROOT}/data/eval_samples/closedform_mc.jsonl"
LETTERS = "ABCDE"

SYS = ("You are a medical expert answering a multiple-choice question. "
       "Reason briefly if needed, then end your reply with a line exactly of the form "
       "'Answer: X' where X is the single letter of the correct option.")

def build_user(rec):
    opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(rec["choices"]))
    return f"{rec['prompt']}\n\nOptions:\n{opts}\n\nAnswer:"

def extract(text, n_choices):
    if not text:
        return None
    valid = set(LETTERS[:n_choices])
    # 1) last explicit "Answer: X"
    m = re.findall(r"[Aa]nswer\s*[:\-]?\s*\(?([A-E])\b", text)
    for cand in reversed(m):
        if cand.upper() in valid:
            return cand.upper()
    # 2) last standalone capital letter token (e.g. "(C)" or "C.")
    m = re.findall(r"\b([A-E])\b", text)
    for cand in reversed(m):
        if cand.upper() in valid:
            return cand.upper()
    return None

async def one(client, model, rec, sem, retries=2):
    user = build_user(rec)
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": user}]
    async with sem:
        for attempt in range(retries + 1):
            try:
                kw = dict(model=model, messages=msgs, max_tokens=4096)
                # T=0 to match OpenBioRQ; some reasoning models reject temperature -> retry without
                if attempt == 0:
                    kw["temperature"] = 0.0
                r = await client.chat.completions.create(**kw)
                txt = r.choices[0].message.content or ""
                return {"id": rec["id"], "dataset": rec["dataset"], "gold": rec["answer"],
                        "pred": extract(txt, len(rec["choices"])), "raw_tail": txt[-160:]}
            except Exception as e:
                if attempt == retries:
                    return {"id": rec["id"], "dataset": rec["dataset"], "gold": rec["answer"],
                            "pred": None, "error": str(e)[:120]}
                await asyncio.sleep(1.5 * (attempt + 1))

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--datasets", default="MedQA,PubMedQA,MedMCQA")
    ap.add_argument("--limit", type=int, default=0, help="subsample per dataset (0=all)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "dummy"))
    a = ap.parse_args()

    keep = set(a.datasets.split(","))
    recs = [json.loads(l) for l in open(DATA)]
    recs = [r for r in recs if r["dataset"] in keep]
    if a.limit:
        byd = defaultdict(list)
        for r in recs:
            byd[r["dataset"]].append(r)
        recs = [r for ds in byd.values() for r in ds[:a.limit]]
    print(f"[{a.tag}] {len(recs)} questions, model={a.model} base={a.base}", file=sys.stderr)

    client = AsyncOpenAI(base_url=a.base, api_key=a.api_key, timeout=120, max_retries=0)
    sem = asyncio.Semaphore(a.workers)
    results = await asyncio.gather(*(one(client, a.model, r, sem) for r in recs))

    outdir = f"{ROOT}/results/medqa_ortho"
    os.makedirs(outdir, exist_ok=True)
    outp = f"{outdir}/{a.tag}.jsonl"
    with open(outp, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    agg = defaultdict(lambda: [0, 0, 0])  # dataset -> [correct, total, unparsed]
    for r in results:
        d = agg[r["dataset"]]
        d[1] += 1
        if r["pred"] is None:
            d[2] += 1
        elif r["pred"] == r["gold"]:
            d[0] += 1
    summary = {"tag": a.tag, "model": a.model, "per_dataset": {}}
    print(f"\n=== {a.tag} ({a.model}) ===")
    for ds in ["MedQA", "PubMedQA", "MedMCQA"]:
        if ds not in agg:
            continue
        c, t, u = agg[ds]
        acc = 100 * c / t if t else 0
        summary["per_dataset"][ds] = {"acc": round(acc, 1), "correct": c, "n": t, "unparsed": u}
        print(f"  {ds:9s} acc={acc:5.1f}%  ({c}/{t}, unparsed={u})")
    json.dump(summary, open(f"{outdir}/{a.tag}_summary.json", "w"), indent=2)
    print(f"wrote {outp}")

if __name__ == "__main__":
    asyncio.run(main())
