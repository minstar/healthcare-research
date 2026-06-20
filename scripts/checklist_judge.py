"""Checklist judge: grade a run's answers against per-question rubrics.

Replaces vague 0-1 dimension scoring with per-criterion met/partial/not_met verdicts
against the question's frozen rubric. Score = Σ(weight·v)/Σweight where
v ∈ {met:1.0, partial:0.5, not_met:0.0}. Both judges grade the SAME rubric, so
agreement should be far higher than free-form dimension scoring.

Usage:
    OPENAI_API_BASE=http://<judge>:8000/v1 OPENAI_API_KEY=dummy \
    python scripts/checklist_judge.py \
        --traces results/baseline_glm_60/traces.jsonl \
        --rubrics data/eval_samples/rubrics_60.jsonl \
        --judge openai/qwen-3.6 --base http://<judge>:8000/v1 \
        --out results/baseline_glm_60/checklist_qwen.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _load_done(path: str) -> set:
    """task_ids already scored OK in a prior (possibly preempted) run — skip on resume."""
    done = set()
    if path and os.path.exists(path):
        for l in open(path):
            try:
                r = json.loads(l)
                if r.get("checklist_score") is not None:
                    done.add(r["task_id"])
            except Exception:
                pass
    return done

V = {"met": 1.0, "yes": 1.0, "partial": 0.5, "not_met": 0.0, "no": 0.0}

SYS = """You grade an AI answer against a fixed checklist of criteria for an open medical \
question. For EACH criterion decide: "met", "partial", or "not_met", judging ONLY from the \
answer text and its tool calls. For must_avoid criteria, "met" means the answer correctly \
AVOIDED the bad behavior. Be strict and literal. Grading guide: "met" = the answer clearly and correctly satisfies
the criterion; "partial" = touches it but vaguely/incompletely or with a minor error;
"not_met" = absent, wrong, or (for must_ground) the claim has no real cited evidence.
For must_avoid: "met" = the bad behavior did NOT occur. An answer that says "tools
returned nothing" while its tool calls returned results FAILS the relevant must_avoid.

WORKED EXAMPLE.
Criterion: "3. [must_ground w2] Cites real primary evidence (PMID) for mechanistic claims"
- Answer cites "PMID:28555461 (Zeppenfeld 2017) for AQP4 depolarization" -> {"id":3,"v":"met","why":"cites real PMID for the claim"}
- Answer says "studies show..." with no IDs -> {"id":3,"v":"not_met","why":"no citations for claims"}
- Answer cites one PMID but most claims uncited -> {"id":3,"v":"partial","why":"only partially grounded"}

Output ONLY JSON: {"verdicts":[{"id":<int>,"v":"met|partial|not_met","why":"<=12 words"}]}"""


def score(verdicts, criteria):
    by_id = {c["id"]: c for c in criteria}
    num = den = 0.0
    for v in verdicts:
        c = by_id.get(v.get("id"))
        if not c:
            continue
        den += c["weight"]
        num += c["weight"] * V.get(str(v.get("v", "not_met")).lower(), 0.0)
    return (num / den) if den else 0.0


def fmt_tools(tcs):
    lines = []
    for tc in (tcs or [])[:12]:
        lines.append(f"{tc.get('tool')}({json.dumps(tc.get('arguments', {}), ensure_ascii=False)[:80]}) "
                     f"-> {str(tc.get('result_preview',''))[:120]}")
    return "\n".join(lines) or "(no tool calls)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--rubrics", required=True)
    ap.add_argument("--judge", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    cli = OpenAI(base_url=args.base, api_key="dummy")
    rubrics = {json.loads(l)["task_id"]: json.loads(l) for l in open(args.rubrics)}
    traces = [json.loads(l) for l in open(args.traces)]

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
            resp = cli.chat.completions.create(model=args.judge.split("/")[-1], max_tokens=3500, temperature=0.1,
                messages=[{"role": "system", "content": SYS}, {"role": "user", "content": user}])
            txt = resp.choices[0].message.content or ""
            m = re.search(r"\{.*\}", txt, re.S)
            verdicts = json.loads(m.group(0)).get("verdicts", []) if m else []
        except Exception as e:
            return {"task_id": tid, "error": str(e)[:150]}
        if not verdicts:
            return {"task_id": tid, "error": "no verdicts"}
        s = score(verdicts, rub["criteria"])
        return {"task_id": tid, "checklist_score": round(s, 3), "n_criteria": len(rub["criteria"]),
                "verdicts": verdicts}

    # resume: skip task_ids already scored OK; append so a preemption keeps progress
    done = _load_done(args.out)
    todo = [t for t in traces if t.get("task_id") not in done]
    print(f"{_ts()} checklist_judge: {len(traces)} traces, {len(done)} already done, {len(todo)} to judge",
          flush=True)

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
                os.fsync(fh.fileno())          # survive node loss on preemption
            if "checklist_score" in o:
                n_ok += 1
            else:
                n_err += 1
            if i % 25 == 0 or i == len(todo):
                print(f"{_ts()}   {i}/{len(todo)} judged (ok={n_ok} err={n_err})", flush=True)
    fh.close()
    print(f"{_ts()} done: +{n_ok} scored, {n_err} errors this run → {args.out}", flush=True)


if __name__ == "__main__":
    main()
