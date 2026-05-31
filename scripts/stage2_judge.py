"""Stage 2 — retrieval-grounded open_status re-judgment (the part that needs an LLM).

For each item: gather real follow-up evidence (StatusVerifier), then ask the LLM to
judge open_status USING ONLY that evidence. Hallucinated citations are rejected:
status_evidence_ids must be a subset of the retrieved evidence IDs, else the item
falls back to "unknown". This is what finally makes `answered`/`unknown` reachable.

LLM access uses the project's existing Claude CLI path (cf. scripts/refine_batch.py),
since no API key is configured in the environment.

Usage:
    python scripts/stage2_judge.py --data data/export/mcp_benchmark_v2.jsonl \
        --sample 30 --model sonnet --out data/stage2
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.status_verifier import StatusVerifier  # noqa: E402

STAGE2_FLAGS = {"heavy_followup_recheck", "trial_completed", "arxiv_citation_check_needed"}

SYSTEM = """You judge whether a medical research question is STILL OPEN, using ONLY the \
retrieved follow-up evidence provided. Do NOT use prior knowledge to claim a question \
is resolved — resolution must be visible in the evidence. Output ONLY a JSON object."""

PROMPT = """QUESTION: {q}
SOURCE FRAMING (raised {date}): {orig}

RETRIEVED FOLLOW-UP EVIDENCE (papers citing the source / trial results posted AFTER \
the question was raised):
{evidence}

Decide open_status:
- "answered": a follow-up DIRECTLY and substantially resolves the question.
- "partially_answered": follow-ups show real progress but leave clear gaps. (Trials \
that are COMPLETED with results posted are at least partially_answered.)
- "open": follow-ups still treat it as unresolved, or none address it.
- "unknown": evidence is absent or irrelevant.

Output JSON ONLY with keys:
  "open_status": one of the four labels
  "status_evidence_ids": list of evidence IDs (e.g. "PMID:123","NCT:456") drawn ONLY \
from the evidence above that justify your label; [] if none
  "resolution_confidence": 0.0-1.0
  "status_reasoning": one or two sentences grounded in the cited evidence
"""


def call_claude(model: str, system: str, user: str, timeout: int = 300) -> dict | None:
    try:
        r = subprocess.run(
            ["claude", "--print", "--model", model, "--system-prompt", system],
            input=user, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None
    out = r.stdout.strip()
    m = re.search(r"\{.*\}", out, re.S)  # tolerate ```json fences / prose
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/export/mcp_benchmark_v2.jsonl")
    ap.add_argument("--sample", type=int, default=30)
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--out", default="data/stage2")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only-flags", default=",".join(STAGE2_FLAGS))
    args = ap.parse_args()

    flags = set(args.only_flags.split(","))
    rows = [json.loads(l) for l in open(args.data)]
    cand = [r for r in rows if r.get("audit_flag") in flags][: args.sample]
    print(f"Stage-2 judging {len(cand)} items (model={args.model})...")

    v = StatusVerifier(cache_path="data/_status_cache.jsonl", citing_top_k=6)
    results = []

    def judge(r):
        b = v.gather_evidence(r)
        valid_ids = b.evidence_ids()
        rec = {"source_id": r["source_id"], "old_status": r.get("open_status"),
               "audit_flag": r.get("audit_flag"), "method": b.method,
               "n_evidence": len(b.items)}
        if b.method != "retrieval" or not b.items:
            rec.update(new_status="unknown", changed=(r.get("open_status") != "unknown"),
                       confidence=0.0, reason="no usable evidence", guardrail="no_evidence")
            return rec
        ans = call_claude(args.model, SYSTEM, PROMPT.format(
            q=r.get("self_contained_question", ""), date=b.source_date,
            orig=r.get("original_question", ""), evidence=b.to_prompt_block()))
        if not ans:
            rec.update(new_status=r.get("open_status"), changed=False, confidence=0.0,
                       reason="LLM parse failed", guardrail="llm_fail")
            return rec
        new = ans.get("open_status", "unknown")
        cited = ans.get("status_evidence_ids", []) or []
        # guardrail: cited IDs must come from the retrieved evidence
        hallucinated = [c for c in cited if c not in valid_ids]
        guard = "ok"
        if hallucinated:
            guard = f"hallucinated:{len(hallucinated)}"
            if new in ("answered", "partially_answered"):
                new = "unknown"  # resolution claim not backed by real evidence
        rec.update(new_status=new, changed=(new != r.get("old_status", r.get("open_status"))),
                   confidence=ans.get("resolution_confidence", 0.0),
                   reason=ans.get("status_reasoning", "")[:300],
                   cited_ids=cited, guardrail=guard)
        return rec

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, rec in enumerate(ex.map(judge, cand), 1):
            results.append(rec)
            if i % 10 == 0:
                print(f"  {i}/{len(cand)}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "stage2_sample.jsonl").open("w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    trans = collections.Counter(f"{r['old_status']} -> {r['new_status']}" for r in results)
    summary = {
        "judged": len(results),
        "changed": sum(1 for r in results if r["changed"]),
        "new_status_dist": dict(collections.Counter(r["new_status"] for r in results)),
        "transitions": dict(trans),
        "guardrail_triggers": dict(collections.Counter(
            r["guardrail"] for r in results if r["guardrail"] != "ok")),
    }
    print("\n" + json.dumps(summary, indent=2, ensure_ascii=False))
    with (out / "stage2_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
