"""Build v3.4 — fully QUESTION-GRANULAR empirical labels across all tracks.

Supersedes v3.3's source-paper-granular core_nanje. Three bucket sources, all keyed by
the UNIQUE task_id (source_id#k) assigned during the 2026-06-03 salvage:
  - /tmp/buckets_1969_uid.json   -> relabel the v3 base rows (join uid->gold text->v3 text)
  - /tmp/buckets_priority.json   -> priority_questions.jsonl (join by task_id)
  - /tmp/buckets_expand.json     -> expand_questions.jsonl   (join by task_id)

Bucket -> empirical_difficulty: all3_fail=core_nanje, split=discriminating, all3_pass=easy.
all3_pass rows are quarantined as unaudited (deep all-pass audit is a separate step).
Any bucket file that doesn't exist yet is skipped (run is incremental/idempotent).

Usage: python scripts/build_v3_4.py
"""
from __future__ import annotations
import collections, json, re
from pathlib import Path

V3 = "data/export/mcp_benchmark_v3.jsonl"
GOLD = "data/export/mcp_benchmark_with_gold.jsonl"
PRIO = "data/export/priority_questions.jsonl"
EXPAND = "data/export/expand_questions.jsonl"
B1969 = "/tmp/buckets_1969_uid.json"
BPRIO = "/tmp/buckets_priority.json"
BEXP = "/tmp/buckets_expand.json"
OUT = "data/export/mcp_benchmark_v3.4.jsonl"

B2D = {"all3_fail": "core_nanje", "split": "discriminating", "all3_pass": "easy"}


def norm(q: str) -> str:
    return re.sub(r"\s+", " ", (q or "").lower().strip())


def load_json(p):
    return json.load(open(p)) if Path(p).exists() else {}


def emp_fields(emp):
    b = emp["b"]
    return {
        "empirical_difficulty": B2D[b],
        "nanje_core": (b == "all3_fail"),
        "model_scores_glmjudge": {"glm": emp["glm"], "qwen": emp["qwen"], "deepseek_v4": emp["v4"]},
        "n_models_fail": sum(emp[k] < 0.5 for k in ("glm", "qwen", "v4")),
        "difficulty_source": emp["src"],
        "quarantine": (b == "all3_pass"),
        "quarantine_reason": ("allpass_unaudited" if b == "all3_pass" else None),
    }


def main():
    b1969, bprio, bexp = load_json(B1969), load_json(BPRIO), load_json(BEXP)

    # 1969 buckets -> question text (via gold unique task_id) for joining to v3 base rows
    gold_uid2text = {r["task_id"]: norm(r.get("self_contained_question", ""))
                     for r in map(json.loads, open(GOLD))}
    text2emp = {}
    for tid, v in b1969.items():
        q = gold_uid2text.get(tid)
        if q:
            text2emp[q] = {"src": "3model_empirical_1969_uid", **v}

    cnt = collections.Counter()
    out_rows = []

    # base v3 rows, relabeled question-granular
    for r in map(json.loads, open(V3)):
        emp = text2emp.get(norm(r.get("self_contained_question", "")))
        if emp:
            r.update(emp_fields(emp))
            cnt[("v3", r["empirical_difficulty"])] += 1
        else:
            r.setdefault("empirical_difficulty", None)
            r.setdefault("nanje_core", None)
            r.setdefault("difficulty_source", "self_rated_deprecated")
            r.setdefault("quarantine", False)
        out_rows.append(r)

    # appended tracks: priority + expand, joined by unique task_id
    for path, buckets, tag in [(PRIO, bprio, "3model_empirical_priority"),
                               (EXPAND, bexp, "3model_empirical_expand")]:
        if not Path(path).exists():
            continue
        for r in map(json.loads, open(path)):
            emp = buckets.get(str(r.get("task_id")))
            if emp:
                r.update(emp_fields({**emp, "src": tag}))
                cnt[(tag, r["empirical_difficulty"])] += 1
            else:
                r["empirical_difficulty"] = None
                r["nanje_core"] = None
                r["difficulty_source"] = "self_rated_deprecated"
                r["quarantine"] = False
            out_rows.append(r)

    with open(OUT, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    core = sum(1 for r in out_rows if r.get("nanje_core") and not r.get("quarantine"))
    disc = sum(1 for r in out_rows if r.get("empirical_difficulty") == "discriminating")
    easy = sum(1 for r in out_rows if r.get("empirical_difficulty") == "easy")
    print(f"v3.4: {len(out_rows)} rows -> {OUT}")
    print(f"  by track/difficulty: { {f'{k[0]}:{k[1]}': v for k, v in sorted(cnt.items())} }")
    print(f"  TOTAL core_nanje (question-granular): {core}")
    print(f"  discriminating: {disc} | easy(quarantined): {easy}")


if __name__ == "__main__":
    main()
