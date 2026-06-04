"""Exp B1 — behavioral metrics across models (no LLM, no GPU).

Quantifies, per model and per benchmark set, the directly-measurable behaviors the paper
mirrors from ResearchMath plus the agentic-collapse signal unique to OpenBioRQ:

  - non_attempt:  empty answer OR explicit refusal / "too complex" deflection
  - zero_tool:    answered with 0 tool calls (agentic collapse)
  - avg_tools:    mean tool calls per trajectory
  - cite_rate:    fraction of answers that cite >=1 PMID/NCT (citation behavior, model-divergent)
  - mean_ans_len: mean answer length (chars)

Problem-substitution requires semantic judgment and is handled in Exp B2 (LLM).

Usage: python scripts/exp_behavioral.py [--out results/exp_behavioral.json]
"""
from __future__ import annotations
import argparse, glob, json, re, statistics

PMID = re.compile(r"PMID[:\s]*\d{5,9}|\b\d{8}\b")
NCT = re.compile(r"NCT\d{8}")
REFUSE = re.compile(
    r"\b(too complex|cannot (?:answer|provide|determine)|unable to|i (?:do not|don't) have"
    r"|beyond (?:my|the scope)|consult (?:a|your) (?:doctor|physician|healthcare)"
    r"|insufficient (?:information|data) to|not enough (?:information|data))\b", re.I)

SETS = {
    "1969": ["baseline_glm_1969", "baseline_qwen_1969", "baseline_dsv4_1969"],
    "priority": ["priority_glm", "priority_qwen", "priority_dsv4"],
    "expand": ["expand_glm", "expand_qwen", "expand_dsv4"],
}
MODEL_OF = {"glm": "GLM-5.1", "qwen": "Qwen3.6", "dsv4": "DeepSeek-V4"}


def model_key(dirname: str) -> str:
    for k in ("glm", "qwen", "dsv4"):
        if k in dirname:
            return k
    return "?"


def metrics(path):
    tr = [json.loads(l) for l in open(path)]
    n = len(tr) or 1
    ans = [(t.get("model_answer") or "") for t in tr]
    tools = [len(t.get("tool_calls") or []) for t in tr]
    empty = [a for a in ans if len(a.strip()) < 20]
    refuse = [a for a in ans if len(a.strip()) >= 20 and REFUSE.search(a)]
    non_attempt = len(empty) + len(refuse)
    cite = sum(1 for a in ans if PMID.search(a) or NCT.search(a))
    return {
        "n": len(tr),
        "non_attempt": round(non_attempt / n, 3),
        "  empty": len(empty), "  refuse": len(refuse),
        "zero_tool": round(sum(1 for x in tools if x == 0) / n, 3),
        "avg_tools": round(statistics.mean(tools), 1),
        "cite_rate": round(cite / n, 3),
        "mean_ans_len": int(statistics.mean(len(a) for a in ans)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/exp_behavioral.json")
    args = ap.parse_args()

    out = {}
    for setname, dirs in SETS.items():
        out[setname] = {}
        for d in dirs:
            p = f"results/{d}/traces.jsonl"
            if not glob.glob(p):
                continue
            out[setname][MODEL_OF[model_key(d)]] = metrics(p)

    json.dump(out, open(args.out, "w"), indent=2)
    # pretty table
    cols = ["n", "non_attempt", "zero_tool", "avg_tools", "cite_rate", "mean_ans_len"]
    for setname, bym in out.items():
        print(f"\n=== {setname} ===")
        print(f'{"model":<14}' + "".join(f"{c:>13}" for c in cols))
        for model, m in bym.items():
            print(f"{model:<14}" + "".join(f"{m[c]:>13}" for c in cols))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
