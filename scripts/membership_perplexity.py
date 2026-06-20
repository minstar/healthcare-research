#!/usr/bin/env python3
"""Model-based membership probe (Reviewer C/D): perplexity vs a likely-seen reference.

Memorized text has anomalously low perplexity. We compare a served open-weight model's mean
token log-prob on the robust-core questions against its log-prob on MedQA-USMLE questions --- a
long-public benchmark very likely present in pretraining corpora. If the robust-core questions
are NOT lower-perplexity (i.e. not more predictable) than the known-public MedQA set, they show
no memorization signature beyond a benchmark we already expect to be seen.

Uses vLLM /v1/completions with echo=True, max_tokens=0, logprobs=1 to read prompt logprobs.
Output: results/membership_perplexity.json

Usage: python membership_perplexity.py --base http://NODE:8000/v1 --model glm-5.1 [--limit 200]
"""
import argparse, json
import requests
import statistics as st

ROOT = "/data/project/private/minstar/workspace/healthcare-research"

def mean_logprob(base, model, text, key):
    r = requests.post(f"{base}/completions", json={
        "model": model, "prompt": text, "max_tokens": 0, "echo": True,
        "logprobs": 1, "temperature": 0.0}, timeout=60)
    r.raise_for_status()
    lp = r.json()["choices"][0]["logprobs"]["token_logprobs"]
    vals = [x for x in lp if x is not None]
    return sum(vals) / len(vals) if vals else None

def collect(base, model, texts):
    out = []
    for t in texts:
        if len((t or "").split()) < 6:
            continue
        try:
            v = mean_logprob(base, model, t, "")
        except Exception:
            v = None
        if v is not None:
            out.append(v)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=200)
    a = ap.parse_args()

    # robust-core questions
    corp = {json.loads(l)["task_id"]: json.loads(l).get("self_contained_question", "")
            for l in open(f"{ROOT}/data/eval_samples/core_eval.jsonl")}
    rc_ids = json.load(open(f"{ROOT}/results/robust_core_ids.json"))[: a.limit]
    rc_q = [corp[t] for t in rc_ids if t in corp]

    # MedQA reference (likely-seen public benchmark) — use the question stem only
    medqa = [json.loads(l)["prompt"] for l in open(f"{ROOT}/data/eval_samples/closedform_mc.jsonl")
             if json.loads(l)["dataset"] == "MedQA"][: a.limit]

    rc_lp = collect(a.base, a.model, rc_q)
    mq_lp = collect(a.base, a.model, medqa)

    summary = {
        "model": a.model,
        "robust_core": {"n": len(rc_lp), "mean_logprob": round(st.mean(rc_lp), 3),
                        "median_logprob": round(st.median(rc_lp), 3)},
        "medqa_reference": {"n": len(mq_lp), "mean_logprob": round(st.mean(mq_lp), 3),
                            "median_logprob": round(st.median(mq_lp), 3)},
        "delta_mean_logprob_rc_minus_medqa": round(st.mean(rc_lp) - st.mean(mq_lp), 3),
        "interpretation": "logprob is per-token mean (higher = more predictable = lower perplexity). "
                          "If robust-core <= MedQA (a long-public, likely-pretrained benchmark), the "
                          "robust core shows no stronger memorization signature than a set we already "
                          "expect to be seen -> no evidence of verbatim contamination.",
    }
    json.dump(summary, open(f"{ROOT}/results/membership_perplexity.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print(f"wrote {ROOT}/results/membership_perplexity.json")

if __name__ == "__main__":
    main()
