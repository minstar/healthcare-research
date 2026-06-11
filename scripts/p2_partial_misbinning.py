#!/usr/bin/env python3
"""P2 (Reviewer R2): is the 'partial' L2 bin hiding clinically-unsupporting citations?

wrong-paper is defined as supports=='no' and EXCLUDES 'partial' (43% of citations). If the judge
routes genuinely-unsupporting citations into the lenient 'partial' bin, wrong-paper is UNDER-reported.
We re-judge a stratified sample of 'partial' citations with a STRICT binary prompt (tangential/partial
support counts as NO) and report how many 'partial' flip to 'no'. That fraction bounds the
under-reporting: a wrong-paper UPPER estimate = no + (partial->no).

Money-code: hard --budget cap, resumable, --dry-run, env-guarded. Reuses the Europe-PMC fetch and
litellm call of scripts/gold_cite_xfamily_judge.py. Opus-4.7 via OpenRouter (funded /data/.../.env).
"""
import argparse, json, os, re, time
import requests, litellm

EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
PRICE = {"openrouter/anthropic/claude-opus-4.7": (5e-6, 25e-6)}
STRICT_SYS = """You verify a CITATION strictly. Given a paper's title+abstract and a claim the \
citation is attached to, decide if the paper DIRECTLY supports that specific claim. Be strict: \
merely related-topic, tangential, wrong-population, wrong-direction, or wrong-endpoint support \
counts as NOT supporting. Output ONLY JSON: {"supports":"yes|no","why":"<=15 words"}."""

def fetch_abstract(pid):
    try:
        r = requests.get(EPMC, params={"query": f"EXT_ID:{pid} AND SRC:MED", "format": "json",
                                        "resultType": "core", "pageSize": 1}, timeout=30)
        res = r.json().get("resultList", {}).get("result", [])
        if not res:
            return None
        x = res[0]
        return {"title": x.get("title", ""), "abstract": (x.get("abstractText") or "")[:1500]}
    except Exception:
        return None

def judge(model, title, abstract, claim):
    user = f"PAPER TITLE: {title}\nABSTRACT: {abstract}\n\nCLAIM: {claim}\n\nDoes the paper directly support the claim?"
    for attempt in range(4):
        try:
            r = litellm.completion(model=model, temperature=0.0, max_tokens=120,
                                   messages=[{"role": "system", "content": STRICT_SYS},
                                             {"role": "user", "content": user}])
            txt = r.choices[0].message.content or ""
            u = r.usage
            pin, pout = (u.prompt_tokens or 0, u.completion_tokens or 0) if u else (0, 0)
            sup = "error"
            try:
                sup = str(json.loads(txt[txt.find("{"):txt.rfind("}") + 1]).get("supports", "error")).lower()
            except Exception:
                pass
            return (sup if sup in ("yes", "no") else "error"), pin, pout
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return "error", 0, 0

def substantive(claim):
    # real assertion, not a bare citation fragment: >=12 words and not dominated by the citation
    w = re.findall(r"[A-Za-z]{3,}", claim or "")
    return len(claim.split()) >= 12 and len(w) >= 8

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/p2_partial_misbinning.jsonl")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--budget", type=float, default=2.0)
    ap.add_argument("--dry-run", type=int, default=0)
    ap.add_argument("--model", default="openrouter/anthropic/claude-opus-4.7")
    a = ap.parse_args()
    cin, cout = PRICE[a.model]

    rows = [json.loads(l) for l in open("results/l2_fullpop.jsonl")]
    partials = [r for r in rows if r["supports"] == "partial" and substantive(r["claim"])]
    # stratify across models, deterministic
    from collections import defaultdict
    by = defaultdict(list)
    for r in partials:
        by[r["model"]].append(r)
    for m in by:
        by[m].sort(key=lambda r: str(r["task_id"]))
    per = max(1, a.n // max(1, len(by)))
    picked = [r for m in sorted(by) for r in by[m][:per]][: a.n]

    done = {}
    if os.path.exists(a.out):
        for l in open(a.out):
            d = json.loads(l)
            if d.get("strict") not in (None, "error"):
                done[d["task_id"] + "|" + str(d["id"])] = d
    spent = sum(d.get("in_tok", 0) * cin + d.get("out_tok", 0) * cout for d in done.values())
    todo = [r for r in picked if (r["task_id"] + "|" + str(r["id"])) not in done]
    print(f"partials(substantive)={len(partials)} sample={len(picked)} todo={len(todo)} "
          f"spent=${spent:.4f} budget=${a.budget} dry_run={a.dry_run}", flush=True)

    n = 0
    with open(a.out, "a") as f:
        for r in todo:
            if spent >= a.budget:
                print(f"BUDGET STOP at ${spent:.4f}"); break
            if a.dry_run and n >= a.dry_run:
                print(f"DRY-RUN stop after {n}"); break
            ab = fetch_abstract(r["id"])
            if not ab:
                rec = {**{k: r[k] for k in ("task_id", "model", "id")}, "strict": "no_abstract",
                       "in_tok": 0, "out_tok": 0}
            else:
                t0 = time.time()
                strict, pin, pout = judge(a.model, ab["title"], ab["abstract"], r["claim"])
                spent += pin * cin + pout * cout
                rec = {**{k: r[k] for k in ("task_id", "model", "id")}, "orig": "partial",
                       "strict": strict, "in_tok": pin, "out_tok": pout}
                print(f"  [{n+1}] {r['model']} id={r['id']} partial->{strict} spent=${spent:.4f}", flush=True)
            f.write(json.dumps(rec) + "\n"); f.flush()
            n += 1

    # summary
    res = [json.loads(l) for l in open(a.out)]
    judged = [d for d in res if d.get("strict") in ("yes", "no")]
    to_no = sum(1 for d in judged if d["strict"] == "no")
    print(f"\n=== P2: partial -> strict ===")
    print(f"judged {len(judged)} substantive partials; {to_no} ({100*to_no/len(judged):.0f}%) become 'no' under strict binary"
          if judged else "no judged rows")
    if judged:
        json.dump({"n_judged": len(judged), "partial_to_no": to_no,
                   "pct": round(100*to_no/len(judged), 1)},
                  open("results/p2_partial_misbinning_summary.json", "w"), indent=2)

if __name__ == "__main__":
    main()
