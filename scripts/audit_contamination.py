"""Full-corpus contamination audit using the retrieval-grounded StatusVerifier.

LLM-free. Runs the fast `screen()` over every benchmark item to measure how many
current open_status labels are contradicted by real evidence (completed trials,
synthetic templates, no follow-up, etc.).

Usage:
    python scripts/audit_contamination.py \
        --data data/export/mcp_benchmark_with_gold.jsonl \
        --out data/audit --workers 12
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.status_verifier import StatusVerifier  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/export/mcp_benchmark_with_gold.jsonl")
    ap.add_argument("--out", default="data/audit")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--min-interval", type=float, default=0.08)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data)]
    if args.limit:
        rows = rows[: args.limit]
    print(f"Auditing {len(rows)} items with {args.workers} workers...")

    v = StatusVerifier(min_interval=args.min_interval, citing_top_k=6)

    results = [None] * len(rows)
    done = [0]

    def work(i_row):
        i, row = i_row
        res = v.screen(row)
        results[i] = res
        done[0] += 1
        if done[0] % 200 == 0:
            print(f"  {done[0]}/{len(rows)}")
        return res

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, enumerate(rows)))

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "audit_per_item.jsonl").open("w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- aggregate ----
    n = len(results)
    by_type = collections.Counter(r["source_type"] for r in results)
    by_flag = collections.Counter(r["flag"] for r in results)
    by_method = collections.Counter(r["method"] for r in results)

    hard = sum(1 for r in results if r["flag"] in
               ("trial_resolved_but_open", "synthetic_template"))
    unverifiable = sum(1 for r in results if r["method"] == "none")
    no_followup = by_flag.get("no_followup_evidence", 0)
    recheck = by_flag.get("heavy_followup_recheck", 0)

    summary = {
        "total": n,
        "by_source_type": dict(by_type),
        "by_method": dict(by_method),
        "by_flag": dict(by_flag),
        "headline": {
            "hard_contamination": hard,
            "hard_contamination_pct": round(100 * hard / n, 1),
            "verification_impossible": unverifiable,
            "verification_impossible_pct": round(100 * unverifiable / n, 1),
            "pubmed_no_followup": no_followup,
            "pubmed_recheck_candidates": recheck,
            "verifiable_pct": round(100 * (1 - unverifiable / n), 1),
        },
    }
    with (outdir / "audit_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n===== CONTAMINATION AUDIT =====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
