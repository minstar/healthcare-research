"""Checklist judge — OpenRouter / arbitrary OpenAI-compatible judge variant.

Thin wrapper around the GLM-judge logic in checklist_judge.py, with three changes
needed for an independent different-family judge (e.g. Claude Opus-4.7 via OpenRouter):

  1. API key from env (OPENAI_API_KEY / OPENROUTER_API_KEY), not hardcoded "dummy".
  2. Model id sent VERBATIM (no .split("/")[-1]) — OpenRouter needs the full
     "anthropic/claude-opus-4.7" slug, which the original script would mangle to
     "claude-opus-4.7" and 404.
  3. --ids (restrict to a subset, e.g. the 423 robust ids) and --limit (dry-run cap).

Grading prompt, scoring, resume/append, and verdict schema are IDENTICAL to
checklist_judge.py (imported), so Opus and GLM grade the same rubric the same way —
the only swapped variable is the judge model.

Usage (dry run, <=5 calls):
    set -a; source /data/project/private/minstar/.env; set +a   # OPENROUTER_API_KEY
    OPENAI_API_KEY=$OPENROUTER_API_KEY python scripts/checklist_judge_or.py \
        --traces results/api_opus47_robust/traces.jsonl \
        --rubrics data/eval_samples/rubrics_1969_uid.jsonl \
        --judge anthropic/claude-opus-4.7 \
        --base https://openrouter.ai/api/v1 \
        --out results/judge_swap/opus_opus47_robust.jsonl \
        --workers 4 --limit 5
"""
from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

# reuse the exact prompt / scoring / formatting / resume helpers
from checklist_judge import SYS, score, fmt_tools, _load_done, _ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--rubrics", required=True)
    ap.add_argument("--judge", required=True, help="model id sent VERBATIM (e.g. anthropic/claude-opus-4.7)")
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ids", default=None, help="JSON list file of task_ids to keep (e.g. robust_core_ids.json)")
    ap.add_argument("--limit", type=int, default=0, help="cap number of NEW judge calls (0 = all). For dry runs.")
    args = ap.parse_args()

    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("No OPENAI_API_KEY / OPENROUTER_API_KEY in env (do NOT hardcode).")
    cli = OpenAI(base_url=args.base, api_key=key)

    rubrics = {json.loads(l)["task_id"]: json.loads(l) for l in open(args.rubrics)}
    traces = [json.loads(l) for l in open(args.traces)]

    if args.ids:
        keep = set(json.load(open(args.ids)))
        traces = [t for t in traces if t.get("task_id") in keep]

    lat_tok = []  # (latency_s, in_tok, out_tok) for the dry-run report

    def judge(tr):
        tid = tr.get("task_id", "")
        rub = rubrics.get(tid)
        if not rub or not rub["criteria"]:
            return None
        crit_txt = "\n".join(f'{c["id"]}. [{c["type"]} w{c["weight"]}] {c["text"]}' for c in rub["criteria"])
        ans = (tr.get("model_answer") or "")[:6000]
        user = (f"QUESTION: {rub['question']}\n\nCHECKLIST:\n{crit_txt}\n\n"
                f"AI ANSWER:\n{ans}\n\nTOOL CALLS:\n{fmt_tools(tr.get('tool_calls'))}\n\nGrade each criterion.")
        try:
            t0 = time.time()
            resp = cli.chat.completions.create(
                model=args.judge,                 # VERBATIM — no split
                max_tokens=3500, temperature=0.1,
                messages=[{"role": "system", "content": SYS}, {"role": "user", "content": user}])
            dt = time.time() - t0
            txt = resp.choices[0].message.content or ""
            u = getattr(resp, "usage", None)
            it = getattr(u, "prompt_tokens", 0) or 0
            ot = getattr(u, "completion_tokens", 0) or 0
            lat_tok.append((dt, it, ot))
            m = re.search(r"\{.*\}", txt, re.S)
            verdicts = json.loads(m.group(0)).get("verdicts", []) if m else []
        except Exception as e:
            return {"task_id": tid, "error": str(e)[:200]}
        if not verdicts:
            return {"task_id": tid, "error": "no verdicts"}
        s = score(verdicts, rub["criteria"])
        return {"task_id": tid, "checklist_score": round(s, 3), "n_criteria": len(rub["criteria"]),
                "verdicts": verdicts}

    done = _load_done(args.out)
    todo = [t for t in traces if t.get("task_id") not in done]
    if args.limit and args.limit > 0:
        todo = todo[:args.limit]
    print(f"{_ts()} checklist_judge_or: {len(traces)} traces in scope, {len(done)} done, {len(todo)} to judge"
          f" (model={args.judge}, base={args.base})", flush=True)

    lock = threading.Lock()
    fh = open(args.out, "a")
    n_ok = n_err = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(judge, t): t for t in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            o = fut.result()
            if not o:
                continue
            with lock:
                fh.write(json.dumps(o, ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            if "checklist_score" in o:
                n_ok += 1
            else:
                n_err += 1
                print(f"{_ts()}   ERR {o.get('task_id')}: {o.get('error')}", flush=True)
            if i % 25 == 0 or i == len(todo):
                print(f"{_ts()}   {i}/{len(todo)} judged (ok={n_ok} err={n_err})", flush=True)
    fh.close()

    if lat_tok:
        import statistics as st
        lats = [x[0] for x in lat_tok]; ins = [x[1] for x in lat_tok]; outs = [x[2] for x in lat_tok]
        print(f"{_ts()} PER-CALL: n={len(lat_tok)} "
              f"lat_med={st.median(lats):.1f}s lat_mean={sum(lats)/len(lats):.1f}s "
              f"in_tok_mean={sum(ins)/len(ins):.0f} out_tok_mean={sum(outs)/len(outs):.0f} "
              f"in_tok_max={max(ins)} out_tok_max={max(outs)}", flush=True)
    print(f"{_ts()} done: +{n_ok} scored, {n_err} errors this run -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
