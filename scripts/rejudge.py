"""Re-judge an existing run's answers with a different (neutral) judge.

Removes self-judging: feed a run's saved model answers + tool calls back through
the judge, using a judge model that is NOT the one that produced the answers.
No regeneration — only judge calls, so it's cheap.

Joins traces (task_id, model_answer, tool_calls) with the data file to recover the
question text and gold answer (keyed by source_id == task_id).

Usage:
    OPENAI_API_BASE=http://<judge-node>:8000/v1 OPENAI_API_KEY=dummy \
    python scripts/rejudge.py \
        --traces results/baseline_glm_60/traces.jsonl \
        --data data/eval_samples/gold_strat_60.jsonl \
        --judge openai/qwen-3.6 --out results/baseline_glm_60/rejudge_qwen.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.judge import judge_single  # noqa: E402

PASS_THRESHOLD = 0.60


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--judge", required=True, help="litellm judge spec, e.g. openai/qwen-3.6")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    # gold + question keyed by source_id
    meta = {}
    for l in open(args.data):
        d = json.loads(l)
        meta[str(d.get("source_id", ""))] = {
            "question": d.get("self_contained_question") or d.get("original_question", ""),
            "gold": d.get("gold_answer"),
        }
    traces = [json.loads(l) for l in open(args.traces)]
    print(f"Re-judging {len(traces)} answers with {args.judge} ...")

    sem = asyncio.Semaphore(args.workers)
    results = []

    async def one(tr):
        tid = tr.get("task_id", "")
        m = meta.get(tid, {})
        if not m.get("question") or m.get("gold") is None:
            return None
        async with sem:
            try:
                jr = await judge_single(
                    task_id=tid, question=m["question"], gold_answer=m["gold"],
                    model_answer=tr.get("model_answer", "") or "",
                    tool_calls=tr.get("tool_calls", []), judge_model=args.judge,
                )
            except Exception as e:
                return {"task_id": tid, "error": str(e)[:200]}
        return {"task_id": tid, "scores": jr.scores, "weighted_total": jr.weighted_total,
                "passed": jr.weighted_total >= PASS_THRESHOLD, "reasoning": jr.reasoning[:300]}

    async def run():
        out = await asyncio.gather(*[one(t) for t in traces])
        return [r for r in out if r]

    results = asyncio.run(run())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ok = [r for r in results if "weighted_total" in r]
    errs = len(results) - len(ok)
    import statistics as st
    if ok:
        print(f"judged {len(ok)} (errors {errs}) | avg={st.mean(r['weighted_total'] for r in ok):.3f} "
              f"| pass={sum(r['passed'] for r in ok)/len(ok):.1%}")
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
