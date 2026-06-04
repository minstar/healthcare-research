"""Compute 3-model difficulty buckets from three checklist_judge outputs.

Mirrors the 1969 bucketing: for each task_id common to all three models, a model
"fails" if checklist_score < threshold (default 0.5). all-3-fail -> core nanje,
all-3-pass -> easy, otherwise split.

Usage:
    python scripts/compute_buckets.py \
        --glm results/priority_glm/checklist_glm.jsonl \
        --qwen results/priority_qwen/checklist_glm.jsonl \
        --dsv4 results/priority_dsv4/checklist_glm.jsonl \
        --out /tmp/buckets_priority.json --threshold 0.5
"""
from __future__ import annotations
import argparse, collections, json


def load(path):
    d = {}
    for l in open(path):
        r = json.loads(l)
        tid = r.get("task_id") or r.get("source_id")
        s = r.get("checklist_score")
        if tid is not None and s is not None:
            d[str(tid)] = float(s)          # last write wins (dup task_ids tolerated)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glm", required=True)
    ap.add_argument("--qwen", required=True)
    ap.add_argument("--dsv4", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    g, q, v = load(args.glm), load(args.qwen), load(args.dsv4)
    common = set(g) & set(q) & set(v)
    th = args.threshold
    out, cnt = {}, collections.Counter()
    for tid in common:
        sg, sq, sv = g[tid], q[tid], v[tid]
        nfail = sum(s < th for s in (sg, sq, sv))
        bucket = "all3_fail" if nfail == 3 else "all3_pass" if nfail == 0 else "split"
        out[tid] = {"b": bucket, "glm": sg, "qwen": sq, "v4": sv}
        cnt[bucket] += 1

    json.dump(out, open(args.out, "w"))
    n = len(common) or 1
    print(f"common task_ids: {len(common)} (glm {len(g)} / qwen {len(q)} / dsv4 {len(v)})")
    print(f"  all3_fail {cnt['all3_fail']} ({100*cnt['all3_fail']/n:.0f}%) | "
          f"split {cnt['split']} ({100*cnt['split']/n:.0f}%) | "
          f"all3_pass {cnt['all3_pass']} ({100*cnt['all3_pass']/n:.0f}%)")
    if common:
        print(f"  model avg: GLM {sum(g[t] for t in common)/n:.3f} "
              f"Qwen {sum(q[t] for t in common)/n:.3f} V4 {sum(v[t] for t in common)/n:.3f}")
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
