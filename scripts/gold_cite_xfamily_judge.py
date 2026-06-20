#!/usr/bin/env python3
# ===========================================================================
# MONEY SCRIPT (paid API). Cross-family RE-JUDGE of the GOLD-ANSWER citation
# audit, to give the headline "~74% wrong-paper" claim the same two-judge
# floor the trajectory audit already has (R2 concern: gold rate rests on a
# SINGLE GLM judge, n=360, abstract-only, no cross-family check).
#
# Design: parity with the GLM gold judge (scripts/audit_citation_relevance_llm.py)
#   * VERBATIM gold SYS prompt + same title+abstract user format.
#   * Same Europe-PMC fetch (resultType=core, abstract[:1500]).
#   * Candidates are read DIRECTLY from the GLM output file, which is provably
#     cites[:400] under seed=21 (verified) — so every Opus row shares a
#     (source_id,pmid) with a GLM verdict -> clean paired sample for kappa.
#   * Stratified sample over the GLM verdict {yes,partial,no} so kappa is not
#     dominated by the majority 'no' class; 'no_abstract'/'error' excluded
#     (no LLM call was/should be made on those).
# Judge = Opus-4.7 via OpenRouter (cross-family from primary GLM-5.1 judge).
# Resumable, HARD --budget stop, retry+backoff, env-guarded billing, exact
# token capture. Reports Opus wrong-paper rate + Cohen's kappa vs GLM.
# ===========================================================================
import os, sys, json, argparse, time, random, re
import requests

# --- env guard: never let the .env MiniMax Anthropic proxy distort billing ---
for _v in ("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_MODEL"):
    os.environ.pop(_v, None)
assert os.environ.get("OPENROUTER_API_KEY"), "OPENROUTER_API_KEY not set — `source .env` first"

import litellm
litellm.request_timeout = 120

# VERBATIM gold judge prompt (scripts/audit_citation_relevance_llm.py) -------
SYS = """You verify a CITATION. Given a paper's title+abstract and a one-sentence claim about \
why it is relevant to a medical question, decide whether the paper actually SUPPORTS that \
claim. Output ONLY JSON: {"supports":"yes|partial|no","why":"<=15 words"}
- yes: the abstract clearly supports the stated relevance.
- partial: related topic but the specific claim is not clearly supported by the abstract.
- no: unrelated, or the claim misrepresents the paper."""

V = {"yes": 1.0, "partial": 0.5, "no": 0.0}
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

# verified pricing USD/token (provider listings 2026-06; same source as l2_claude_judge.py)
PRICE = {
    "openrouter/anthropic/claude-opus-4.7": (5e-6, 25e-6),
    "openrouter/anthropic/claude-haiku-4.5": (1e-6, 5e-6),
}


def fetch_abstract(pmid):
    """Same fetch the GLM gold judge used (Europe PMC core, abstract[:1500])."""
    try:
        r = requests.get(EPMC, params={"query": f"EXT_ID:{pmid} AND SRC:MED", "format": "json",
                                        "resultType": "core", "pageSize": 1}, timeout=30)
        res = r.json().get("resultList", {}).get("result", [])
        if not res:
            return None
        x = res[0]
        return {"title": x.get("title", ""), "abstract": (x.get("abstractText") or "")[:1500]}
    except Exception:
        return None


def judge(model, title, abstract, claim):
    """Returns (supports, in_tok, out_tok, reason). Retries transient errors."""
    user = (f"PAPER TITLE: {title}\nABSTRACT: {abstract}\n\n"
            f"RELEVANCE CLAIM: {claim}\n\nDoes the paper support the claim?")
    last = ""
    for attempt in range(5):
        try:
            r = litellm.completion(model=model, temperature=0.0, max_tokens=120,
                                   messages=[{"role": "system", "content": SYS},
                                             {"role": "user", "content": user}])
            txt = r.choices[0].message.content or ""
            u = r.usage
            pin = (u.prompt_tokens or 0) if u else 0
            pout = (u.completion_tokens or 0) if u else 0
            sup = "error"
            try:
                s = txt[txt.find("{"):txt.rfind("}") + 1]
                sup = str(json.loads(s).get("supports", "error")).lower()
            except Exception:
                pass
            if sup not in ("yes", "partial", "no"):
                return "error", pin, pout, "parse_fail"
            return sup, pin, pout, ""
        except Exception as e:
            last = str(e)[:120]
            if attempt == 4:
                return "error", 0, 0, "api_" + last
            time.sleep(2 ** attempt)


def kappa(pairs):
    """Cohen's kappa on paired categorical labels [(a,b),...] over labels yes/partial/no."""
    labs = ["yes", "partial", "no"]
    n = len(pairs)
    if n == 0:
        return None
    po = sum(1 for a, b in pairs if a == b) / n
    pe = 0.0
    for L in labs:
        pa = sum(1 for a, _ in pairs if a == L) / n
        pb = sum(1 for _, b in pairs if b == L) / n
        pe += pa * pb
    return None if pe == 1.0 else (po - pe) / (1 - pe)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glm", default="results/citation_relevance_llm.jsonl",
                    help="primary GLM gold-citation verdicts (the shared sample source)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=150, help="stratified sample size")
    ap.add_argument("--judge-model", default="openrouter/anthropic/claude-opus-4.7")
    ap.add_argument("--budget", type=float, default=10.0, help="HARD stop USD for THIS script")
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--dry-run", type=int, default=0, help="if >0, make at most this many calls then stop")
    args = ap.parse_args()
    cin, cout = PRICE[args.judge_model]

    # candidates = GLM rows that actually got an LLM verdict (exclude no_abstract/error)
    glm = [json.loads(l) for l in open(args.glm)]
    cand = [r for r in glm if r.get("supports") in ("yes", "partial", "no")]

    # stratified sample across GLM verdict so kappa isn't all-majority-class
    rng = random.Random(args.seed)
    by = {"yes": [], "partial": [], "no": []}
    for r in cand:
        by[r["supports"]].append(r)
    for v in by:
        rng.shuffle(by[v])
    # proportional allocation, but guarantee >=1 from each non-empty stratum
    total = len(cand)
    target = min(args.n, total)
    picked = []
    for v in by:
        if by[v]:
            k = max(1, round(target * len(by[v]) / total))
            picked += by[v][:k]
    rng.shuffle(picked)
    picked = picked[:target]
    if args.dry_run:
        picked = picked[:args.dry_run]
    glm_by_key = {(r["source_id"], r["pmid"]): r["supports"] for r in cand}

    # resume: skip keys already judged with a non-error verdict
    done = {}
    if os.path.exists(args.out):
        for l in open(args.out):
            d = json.loads(l)
            done[(d["source_id"], d["pmid"])] = d
    spent = sum(d.get("in_tok", 0) * cin + d.get("out_tok", 0) * cout for d in done.values())
    todo = [r for r in picked
            if done.get((r["source_id"], r["pmid"]), {}).get("supports") in (None, "error")]
    print(f"[gold-xfamily] strata={ {k: len(v) for k, v in by.items()} } candidates={len(cand)} "
          f"sample={len(picked)} done={len(done)} todo={len(todo)} spent=${spent:.4f} "
          f"budget=${args.budget} judge={args.judge_model} dry_run={args.dry_run}",
          flush=True)

    out = open(args.out, "a")
    n = 0
    for r in todo:
        if spent >= args.budget:
            print(f"BUDGET STOP at ${spent:.4f} (>= ${args.budget}); {len(todo)-n} left", flush=True)
            break
        sid, pmid, claim = r["source_id"], r["pmid"], r["relevance"]
        meta = fetch_abstract(pmid)
        base = {"source_id": sid, "pmid": pmid, "relevance": claim,
                "glm_supports": r["supports"], "judge_model": args.judge_model}
        if not meta or not meta["abstract"]:
            rec = {**base, "supports": "no_abstract", "score": None, "in_tok": 0, "out_tok": 0}
            json.dump(rec, out); out.write("\n"); out.flush(); continue
        t0 = time.time()
        sup, pin, pout, reason = judge(args.judge_model, meta["title"], meta["abstract"], claim)
        dt = time.time() - t0
        spent += pin * cin + pout * cout
        n += 1
        rec = {**base, "supports": sup, "score": V.get(sup), "reason": reason,
               "in_tok": pin, "out_tok": pout, "latency_s": round(dt, 2)}
        json.dump(rec, out); out.write("\n"); out.flush()
        print(f"  [{n}/{len(todo)}] {sid} pmid={pmid} GLM={r['supports']} -> OPUS={sup} "
              f"in={pin} out={pout} {dt:.1f}s spent=${spent:.4f}", flush=True)
    out.close()

    # ---- compute & print comparison over everything judged so far in --out ----
    rows = [json.loads(l) for l in open(args.out)]
    scored = [x for x in rows if x.get("supports") in ("yes", "partial", "no")]
    opus_no = sum(1 for x in scored if x["supports"] == "no")
    print(f"\nDONE judged_now={n} opus_judged={len(scored)} total_spent=${spent:.4f} -> {args.out}",
          flush=True)
    if scored:
        print(f"OPUS wrong-paper (supports==no): {opus_no}/{len(scored)} = "
              f"{100*opus_no/len(scored):.1f}%", flush=True)
        # GLM rate on the SAME judged keys
        pairs = [(glm_by_key[(x["source_id"], x["pmid"])], x["supports"]) for x in scored
                 if (x["source_id"], x["pmid"]) in glm_by_key]
        glm_no = sum(1 for g, _ in pairs if g == "no")
        print(f"GLM  wrong-paper on same keys:   {glm_no}/{len(pairs)} = "
              f"{100*glm_no/len(pairs):.1f}%", flush=True)
        k = kappa(pairs)
        # binary (no vs not-no) agreement too — that's the headline metric
        bpairs = [("no" if g == "no" else "ok", "no" if o == "no" else "ok") for g, o in pairs]
        kb = kappa(bpairs)
        agree = sum(1 for a, b in pairs if a == b) / len(pairs)
        kfmt = "n/a (degenerate: one label only)" if k is None else f"{k:.3f}"
        kbfmt = "n/a (degenerate: one label only)" if kb is None else f"{kb:.3f}"
        print(f"Cohen kappa (3-way yes/partial/no): {kfmt}  |  raw agreement: {agree:.3f}",
              flush=True)
        print(f"Cohen kappa (binary wrong vs ok):   {kbfmt}", flush=True)


if __name__ == "__main__":
    main()
