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
import re
import sys
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

V = {"met": 1.0, "yes": 1.0, "partial": 0.5, "not_met": 0.0, "no": 0.0}

SYS = """You grade an AI answer against a fixed checklist of criteria for an open medical \
question. For EACH criterion decide: "met", "partial", or "not_met", judging ONLY from the \
answer text and its tool calls. For must_avoid criteria, "met" means the answer correctly \
AVOIDED the bad behavior. Be strict and literal. Output ONLY JSON: \
{"verdicts":[{"id":<int>,"v":"met|partial|not_met","why":"<=12 words"}]}"""


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

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        out = [r for r in ex.map(judge, traces) if r]
    with open(args.out, "w") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    ok = [o for o in out if "checklist_score" in o]
    import statistics as st
    if ok:
        print(f"judged {len(ok)} (errs {len(out)-len(ok)}) | avg={st.mean(o['checklist_score'] for o in ok):.3f}")
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
