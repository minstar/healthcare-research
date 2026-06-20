"""Trajectory -> SFT flywheel (ResearchMath-14K H.3 adapted to medical nanje).

Eval traces ARE agentic trajectories (question -> tool calls -> grounded answer). This
filters them into SFT-ready training data, applying the paper's filters plus an
in-trace grounding check that's cleaner than web-verification:

Filters (a trajectory is KEPT only if all pass):
  1. not a NON-ATTEMPT  — the model actually used tools and didn't bail to parametric
     memory ("tools returned nothing", "training knowledge", no tool returned data).
  2. not FAKE-CITED     — every PMID cited in the final answer appears in THIS
     trajectory's own retrieved tool results (cited-but-not-retrieved = fabricated).
  3. QUALITY gate       — checklist judge score >= --min-score (good teaching example).

Output: SFT messages JSONL (system/user/assistant+tool_calls/tool/.../assistant) + meta.

Usage:
    python scripts/build_sft_trajectories.py \
        --runs results/baseline_glm_300 results/baseline_qwen_300 results/baseline_dsv4_300 \
        --data data/eval_samples/gold_strat_300.jsonl --min-score 0.5 \
        --out data/sft/nanje_sft_trajectories.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

BAIL = ["not returning", "returning empty", "tools are not", "training knowledge",
        "unable to retrieve", "relying on my training", "no published", "search tools"]
PMID_RE = re.compile(r"\bPM(?:ID|C)[:\s]?(\d+)\b", re.I)

SYS = ("You are a biomedical research agent answering an open research question using "
       "search/database tools. Ground claims in retrieved evidence (cite PMIDs) and map "
       "what is known vs unknown.")


def cited_pmids(text: str) -> set[str]:
    return {m for m in PMID_RE.findall(text or "")}


def retrieved_pmids(tool_calls: list) -> set[str]:
    out = set()
    for tc in tool_calls or []:
        out |= {m for m in PMID_RE.findall(str(tc.get("result_preview", "")))}
    return out


def to_messages(question: str, trace: list, tool_calls: list, final: str) -> list:
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": question}]
    by_round = collections.defaultdict(list)
    for tc in tool_calls or []:
        by_round[str(tc.get("round", "0"))].append(tc)
    rounds = sorted({str(t.get("round", i)) for i, t in enumerate(trace or [])}, key=lambda x: int(x) if x.isdigit() else 0)
    for r in rounds:
        tr = next((t for t in trace if str(t.get("round")) == r), None)
        calls = by_round.get(r, [])
        content = (tr or {}).get("content") or ""
        if calls:
            msgs.append({"role": "assistant", "content": content,
                         "tool_calls": [{"name": c.get("tool"), "arguments": c.get("arguments")} for c in calls]})
            for c in calls:
                msgs.append({"role": "tool", "name": c.get("tool"),
                             "content": str(c.get("result_preview", ""))[:1500]})
    msgs.append({"role": "assistant", "content": final})
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--min-score", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    qmap = {}
    for l in open(args.data):
        d = json.loads(l)
        qmap[str(d.get("source_id", ""))] = d.get("self_contained_question") or d.get("original_question", "")

    kept, reasons = [], collections.Counter()
    per_model = collections.Counter()
    for run in args.runs:
        model = Path(run).name.replace("baseline_", "").rsplit("_", 1)[0]
        scores = {}
        cl = Path(run) / "checklist_glm.jsonl"
        if cl.exists():
            scores = {json.loads(l)["task_id"]: json.loads(l)["checklist_score"]
                      for l in open(cl) if "checklist_score" in json.loads(l)}
        tpath = Path(run) / "traces.jsonl"
        if not tpath.exists():
            print(f"  skip {run}: no traces"); continue
        for l in open(tpath):
            d = json.loads(l)
            tid = d.get("task_id", "")
            ans = d.get("model_answer", "") or ""
            tcs = d.get("tool_calls", [])
            reasons["total"] += 1
            # filter 1: non-attempt
            ret = retrieved_pmids(tcs)
            if any(b in ans.lower() for b in BAIL) or not ret:
                reasons["drop_non_attempt"] += 1; continue
            # filter 2: fake citation (cited PMID not retrieved in-trace)
            cited = cited_pmids(ans)
            if cited and not cited.issubset(ret):
                reasons["drop_fake_citation"] += 1; continue
            # filter 3: quality gate
            sc = scores.get(tid)
            if sc is None or sc < args.min_score:
                reasons["drop_low_quality"] += 1; continue
            q = qmap.get(tid) or ""
            if not q:
                reasons["drop_no_question"] += 1; continue
            kept.append({"task_id": tid, "teacher_model": model, "checklist_score": sc,
                         "messages": to_messages(q, d.get("trace", []), tcs, ans)})
            per_model[model] += 1

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for k in kept:
            f.write(json.dumps(k, ensure_ascii=False) + "\n")
    tot = reasons["total"]
    print(f"=== SFT trajectory build ===")
    print(f"  input trajectories: {tot}")
    print(f"  KEPT: {len(kept)} ({100*len(kept)/max(1,tot):.0f}%)  by teacher: {dict(per_model)}")
    print(f"  dropped: non_attempt={reasons['drop_non_attempt']} fake_citation={reasons['drop_fake_citation']} "
          f"low_quality(<{args.min_score})={reasons['drop_low_quality']} no_question={reasons['drop_no_question']}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
