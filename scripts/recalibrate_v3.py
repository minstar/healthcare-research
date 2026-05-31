"""Recalibrate v3 difficulty with empirical 3-model labels (judge-reliable).

Self-rated difficulty is deprecated (shown uncorrelated with model performance).
Where we have 3-model checklist evals (288 items), attach an empirical label:
  all3_fail -> empirical_difficulty=core_nanje (the 143), split -> discriminating,
  all3_pass -> easy/demote. Match v3 rows by exact question text (v3 is text-deduped),
  joining verification (keyed by source_id) through the eval sample's source_id->question.

Items without an empirical eval keep self-rated difficulty as deprecated metadata,
flagged difficulty_source=self_rated_deprecated.

Usage: python scripts/recalibrate_v3.py
"""
from __future__ import annotations

import collections
import json

V3 = "data/export/mcp_benchmark_v3.jsonl"
SAMPLE = "data/eval_samples/gold_strat_300.jsonl"
VERIF = "results/nanje_3model_verification.jsonl"
OUT = "data/export/mcp_benchmark_v3.1.jsonl"

BUCKET_TO_DIFF = {"all3_fail": "core_nanje", "split": "discriminating", "all3_pass": "easy"}


def norm(q: str) -> str:
    import re
    return re.sub(r"\s+", " ", (q or "").lower().strip())


def main():
    # source_id -> question (from the eval sample)
    sid2q = {}
    for l in open(SAMPLE):
        d = json.loads(l)
        sid2q[str(d.get("source_id", ""))] = norm(d.get("self_contained_question", ""))

    # question_text -> empirical record (via verification keyed by source_id)
    q2emp = {}
    for l in open(VERIF):
        v = json.loads(l)
        q = sid2q.get(v["task_id"])
        if q:
            q2emp[q] = v

    rows = [json.loads(l) for l in open(V3)]
    labeled = 0
    cnt = collections.Counter()
    with open(OUT, "w") as f:
        for r in rows:
            q = norm(r.get("self_contained_question", ""))
            emp = q2emp.get(q)
            if emp:
                r["empirical_difficulty"] = BUCKET_TO_DIFF[emp["bucket"]]
                r["nanje_core"] = (emp["bucket"] == "all3_fail")
                r["model_scores_glmjudge"] = {"glm": emp["glm"], "qwen": emp["qwen"],
                                              "deepseek_v4": emp["deepseek_v4"]}
                r["n_models_fail"] = emp["n_models_fail"]
                r["difficulty_source"] = "3model_empirical"
                labeled += 1
                cnt[r["empirical_difficulty"]] += 1
            else:
                r["empirical_difficulty"] = None
                r["nanje_core"] = None
                r["difficulty_source"] = "self_rated_deprecated"
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"v3 rows: {len(rows)} | empirically labeled: {labeled}")
    print(f"  by empirical_difficulty: {dict(cnt)}")
    print(f"  nanje_core (all-3-fail): {cnt['core_nanje']}")
    print(f"  remaining (self_rated_deprecated): {len(rows) - labeled}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
