"""Build v3.3 = v3.2 (core_nanje finalized) + priority 525 with empirical labels.

Priority questions (data/export/priority_questions.jsonl) get the same 3-model
empirical_difficulty as v3.2, from /tmp/buckets_priority.json:
  all3_fail -> core_nanje, split -> discriminating, all3_pass -> easy.

Priority rows not covered by buckets (eval/judge dropouts) keep empirical None.
all3_pass priority rows are flagged quarantine_candidate (the deep all-pass audit
that v3.2 got is deferred — noted, not silently treated as nanje).

Usage: python scripts/build_v3_3.py
"""
from __future__ import annotations
import collections, json

V32 = "data/export/mcp_benchmark_v3.2.jsonl"
PRIO = "data/export/priority_questions.jsonl"
BUCKETS = "/tmp/buckets_priority.json"
OUT = "data/export/mcp_benchmark_v3.3.jsonl"

B2D = {"all3_fail": "core_nanje", "split": "discriminating", "all3_pass": "easy"}


def main():
    base = [json.loads(l) for l in open(V32)]
    buckets = json.load(open(BUCKETS))             # source_id -> {b, glm, qwen, v4}
    prio = [json.loads(l) for l in open(PRIO)]

    cnt = collections.Counter()
    labeled = 0
    for r in prio:
        r["difficulty_source"] = "self_rated_deprecated"
        r["empirical_difficulty"] = None
        r["nanje_core"] = None
        r["quarantine"] = False
        r["quarantine_reason"] = None
        # buckets are keyed by the UNIQUE task_id (source_id#k), set during the eval salvage
        emp = buckets.get(str(r.get("task_id"))) or buckets.get(str(r.get("source_id")))
        if emp:
            b = emp["b"]
            r["empirical_difficulty"] = B2D[b]
            r["nanje_core"] = (b == "all3_fail")
            r["model_scores_glmjudge"] = {"glm": emp["glm"], "qwen": emp["qwen"], "deepseek_v4": emp["v4"]}
            r["n_models_fail"] = sum(emp[k] < 0.5 for k in ("glm", "qwen", "v4"))
            r["difficulty_source"] = "3model_empirical_priority"
            if b == "all3_pass":
                r["quarantine"] = True
                r["quarantine_reason"] = "priority_allpass_unaudited"
            labeled += 1
            cnt[r["empirical_difficulty"]] += 1

    with open(OUT, "w") as f:
        for r in base + prio:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    base_core = sum(1 for r in base if r.get("nanje_core") and not r.get("quarantine"))
    prio_core = sum(1 for r in prio if r.get("nanje_core") and not r.get("quarantine"))
    print(f"v3.3: {len(base)} (v3.2) + {len(prio)} (priority) = {len(base)+len(prio)} rows")
    print(f"  priority empirically labeled: {labeled} -> {dict(cnt)}")
    print(f"  core_nanje: v3.2 {base_core} + priority {prio_core} = {base_core+prio_core}")
    print(f"  priority all3_pass quarantined (unaudited): {cnt['easy']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
