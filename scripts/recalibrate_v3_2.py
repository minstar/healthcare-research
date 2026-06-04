"""v3.2 recalibration — apply the 1969-scale 3-model empirical labels + all-pass audit.

Supersedes the 300-item labeling in v3.1. For the ~1396 questions with 3-model
checklist evals (buckets1969.json), attach empirical_difficulty:
  all3_fail -> core_nanje (nanje_core=True), split -> discriminating, all3_pass -> easy.

all-pass questions were individually audited (results/allpass_classification.jsonl):
only genuine_hard is kept; vague_aspirational / already_resolved / broad_goal are
QUARANTINED (quarantine=True, quarantine_reason=<label>) — they passed because they
are not sharply-scoped open problems, not because they are easy nanje.

Join: bucket task_id == eval source_id -> question text (mcp_benchmark_with_gold) ->
normalized text match against v3 rows (v3 is text-deduped).

Usage: python scripts/recalibrate_v3_2.py
"""
from __future__ import annotations
import collections, json, re

V3 = "data/export/mcp_benchmark_v3.jsonl"
BUCKETS = "/tmp/buckets1969.json"
GOLD = "data/export/mcp_benchmark_with_gold.jsonl"
ALLPASS = "results/allpass_classification.jsonl"
OUT = "data/export/mcp_benchmark_v3.2.jsonl"

BUCKET_TO_DIFF = {"all3_fail": "core_nanje", "split": "discriminating", "all3_pass": "easy"}


def norm(q: str) -> str:
    return re.sub(r"\s+", " ", (q or "").lower().strip())


def main():
    buckets = json.load(open(BUCKETS))                 # source_id -> {b, glm, qwen, v4}
    sid2q = {}
    for l in open(GOLD):
        d = json.loads(l)
        sid2q[str(d["source_id"])] = norm(d.get("self_contained_question", ""))

    # all-pass audit: source_id -> (keep, label)
    apaudit = {}
    for l in open(ALLPASS):
        a = json.loads(l)
        apaudit[a["source_id"]] = (bool(a.get("keep")), a.get("label", ""))

    # question_text -> empirical record
    q2emp = {}
    for sid, v in buckets.items():
        q = sid2q.get(sid)
        if q:
            q2emp[q] = {"sid": sid, **v}

    rows = [json.loads(l) for l in open(V3)]
    labeled = 0
    cnt = collections.Counter()
    quarantined = 0
    with open(OUT, "w") as f:
        for r in rows:
            q = norm(r.get("self_contained_question", ""))
            emp = q2emp.get(q)
            if emp:
                bucket = emp["b"]
                r["empirical_difficulty"] = BUCKET_TO_DIFF[bucket]
                r["nanje_core"] = (bucket == "all3_fail")
                r["model_scores_glmjudge"] = {"glm": emp["glm"], "qwen": emp["qwen"], "deepseek_v4": emp["v4"]}
                r["n_models_fail"] = sum(emp[k] < 0.5 for k in ("glm", "qwen", "v4"))
                r["difficulty_source"] = "3model_empirical_1969"
                r["quarantine"] = False
                r["quarantine_reason"] = None
                if bucket == "all3_pass":
                    keep, label = apaudit.get(emp["sid"], (False, "unaudited_allpass"))
                    if not keep:
                        r["quarantine"] = True
                        r["quarantine_reason"] = f"allpass_{label}"
                        quarantined += 1
                labeled += 1
                cnt[r["empirical_difficulty"]] += 1
            else:
                r["empirical_difficulty"] = None
                r["nanje_core"] = None
                r["difficulty_source"] = "self_rated_deprecated"
                r["quarantine"] = False
                r["quarantine_reason"] = None
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"v3 rows: {len(rows)} | empirically labeled: {labeled}")
    print(f"  by empirical_difficulty: {dict(cnt)}")
    print(f"  nanje_core (all-3-fail): {cnt['core_nanje']}")
    print(f"  all-pass quarantined: {quarantined}")
    print(f"  remaining (self_rated_deprecated): {len(rows) - labeled}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
