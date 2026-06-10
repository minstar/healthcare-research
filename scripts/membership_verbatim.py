#!/usr/bin/env python3
"""Verbatim-membership probe (Reviewer D, cheap minimum).

The evaluated question is the LLM-refined `self_contained_question`, not the source text.
If it is substantially reworded from its source, the exact evaluated string cannot be present
verbatim in any pre-cutoff pretraining corpus, so exact-string memorization of the benchmark
question is not a contamination route (topic familiarity is separate; openness is verified).

We quantify rewrite distance between the refined question and its original source phrasing over
the robust core (423): token Jaccard, longest common verbatim word-span (LCS of word seq), and
the fraction of refined questions whose longest verbatim shared span is short.
The model-based perplexity/min-K membership inference is deferred to Exp C (needs serving).
Output: results/membership_verbatim.json
"""
import json, glob, re
from statistics import median

def toks(s): return re.findall(r"[a-z0-9]+", (s or "").lower())

def longest_common_run(a, b):
    """Longest run of consecutive shared word tokens (verbatim span length, in words)."""
    bset = {}
    for j, w in enumerate(b):
        bset.setdefault(w, []).append(j)
    best = 0
    # DP over matching positions
    prev = {}
    for i, w in enumerate(a):
        cur = {}
        for j in bset.get(w, []):
            run = prev.get(j - 1, 0) + 1
            cur[j] = run
            best = max(best, run)
        prev = cur
    return best

corp = {}
for f in ["data/export/mcp_benchmark_v3.4.jsonl"] + glob.glob("data/gold_answers/chunk_*.input.jsonl"):
    for l in open(f):
        try: d = json.loads(l)
        except: continue
        sid = d.get("source_id")
        if sid and sid not in corp and d.get("self_contained_question"):
            corp[sid] = {"orig": d.get("original_question", ""), "sc": d.get("self_contained_question", ""),
                         "title": d.get("source_title", "")}

rc = json.load(open("results/robust_core_ids.json"))
def strip(x): return x.rsplit("#", 1)[0]

jac, runs, runs_vs_src = [], [], []
for x in rc:
    e = corp.get(strip(x))
    if not e: continue
    sc = toks(e["sc"]); og = toks(e["orig"]); ti = toks(e["title"])
    src = og + ti  # source phrasing the refined Q could have been lifted from
    sset, oset = set(sc), set(src)
    jac.append(len(sset & oset) / len(sset | oset) if (sset | oset) else 0)
    runs.append(longest_common_run(sc, og))                 # vs original question
    runs_vs_src.append(longest_common_run(sc, src))         # vs original + source title
    e["_sc_len"] = len(sc)

sc_lens = [len(toks(corp[strip(x)]["sc"])) for x in rc if strip(x) in corp]
n = len(jac)
summary = {
    "n": n,
    "token_jaccard_refined_vs_source": {"median": round(median(jac), 3),
                                        "mean": round(sum(jac) / n, 3)},
    "longest_verbatim_word_span_vs_original": {"median": median(runs), "max": max(runs),
                                               "mean": round(sum(runs) / n, 1)},
    "longest_verbatim_word_span_vs_source": {"median": median(runs_vs_src), "max": max(runs_vs_src),
                                             "mean": round(sum(runs_vs_src) / n, 1)},
    "median_refined_question_len_words": median(sc_lens),
    "frac_verbatim_span_le_5_words": round(sum(1 for r in runs_vs_src if r <= 5) / n, 3),
    "frac_verbatim_span_ge_half_question": round(
        sum(1 for r, x in zip(runs_vs_src, rc) if strip(x) in corp and r >= 0.5 * len(toks(corp[strip(x)]["sc"]))) / n, 3),
}
json.dump(summary, open("results/membership_verbatim.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
print("wrote results/membership_verbatim.json")
