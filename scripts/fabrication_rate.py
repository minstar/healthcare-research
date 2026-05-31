"""Citation-fabrication rate per model (ResearchMath-14K H.4 factuality, adapted).

The paper found newer/agentic-RL models cite far more and FABRICATE far more when
search results aren't grounding them (DeepSeek R1->V4-Pro: 0.5->11.6 fakes/trace).
Here we measure, per model, on the medical nanje eval traces:
  - mention-level: cited PMIDs, of which UNGROUNDED (cited in the final answer but NOT
    present in that trajectory's own retrieved tool results = recalled/invented).
  - trace-level: % of traces with >=1 ungrounded citation.
This is an in-trace grounding check (stricter and cleaner than web-verification): a
well-grounded agent only cites what it retrieved.

Usage:
    python scripts/fabrication_rate.py --runs results/baseline_glm_300 \
        results/baseline_qwen_300 results/baseline_dsv4_300 --out results/fabrication_rate.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PMID_RE = re.compile(r"\bPM(?:ID|C)[:\s]?(\d+)\b", re.I)


def pmids(text: str) -> set[str]:
    return set(PMID_RE.findall(text or ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    report = {}
    for run in args.runs:
        model = Path(run).name.replace("baseline_", "").rsplit("_", 1)[0]
        tpath = Path(run) / "traces.jsonl"
        if not tpath.exists():
            continue
        n_traces = n_cited_traces = n_fab_traces = 0
        tot_mentions = tot_fab = tot_retrieved = 0
        for l in open(tpath):
            d = json.loads(l)
            ans = d.get("model_answer", "") or ""
            ret = set()
            for tc in d.get("tool_calls", []):
                ret |= pmids(str(tc.get("result_preview", "")))
            cited = pmids(ans)
            ungrounded = cited - ret
            n_traces += 1
            tot_retrieved += len(ret)
            if cited:
                n_cited_traces += 1
                tot_mentions += len(cited)
                tot_fab += len(ungrounded)
                if ungrounded:
                    n_fab_traces += 1
        report[model] = {
            "traces": n_traces,
            "traces_with_citations": n_cited_traces,
            "traces_with_ungrounded_cite": n_fab_traces,
            "pct_traces_ungrounded": round(100 * n_fab_traces / max(1, n_cited_traces), 1),
            "total_citations": tot_mentions,
            "ungrounded_citations": tot_fab,
            "pct_mentions_ungrounded": round(100 * tot_fab / max(1, tot_mentions), 1),
            "avg_citations_per_trace": round(tot_mentions / max(1, n_traces), 2),
            "avg_retrieved_pmids_per_trace": round(tot_retrieved / max(1, n_traces), 2),
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2)
    print("=== CITATION-FABRICATION (in-trace grounding) ===")
    print(f"  {'model':14s} {'cites/trace':>11s} {'%mentions_ungrounded':>20s} {'%traces_ungrounded':>18s}")
    for m, r in report.items():
        print(f"  {m:14s} {r['avg_citations_per_trace']:>11} {r['pct_mentions_ungrounded']:>19}% {r['pct_traces_ungrounded']:>17}%")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
