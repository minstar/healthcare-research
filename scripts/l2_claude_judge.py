#!/usr/bin/env python3
# ===========================================================================
# MONEY SCRIPT (paid API). Reviewed 3x before launch (user mandate 2026-06-05).
# Independent cross-family L2 wrong-paper judge with EXACT token/cost capture.
#   --mode fullpop : judge ALL found real citations (title+abstract) using the
#                    GLM judge's VERBATIM prompt -> only the judge FAMILY differs,
#                    validating the headline 19.9% wrong-paper beyond n=118 (R2 P0-3).
#   --mode fulltext: sample N (required --limit), judge on FULL TEXT (OA) vs
#                    abstract, to bound the abstract-only error (R2 P1-5).
# Judge = Opus-4.7 via OpenRouter (cross-family from the primary GLM-5.1 judge).
# Resumable (skips done non-error keys). HARD budget stop at --budget USD.
# Retry+backoff so one transient 429 cannot kill the run. Self-guards env so it
# cannot silently mis-bill via the MiniMax Anthropic proxy.
# ===========================================================================
import os, sys, json, argparse, time, requests
# --- env guard: never let the .env MiniMax Anthropic proxy distort billing -----
for _v in ("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_MODEL"):
    os.environ.pop(_v, None)
assert os.environ.get("OPENROUTER_API_KEY"), "OPENROUTER_API_KEY not set — `source .env` first"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_cite_audit import fetch_pmid, fetch_nct, SYS as GLM_SYS  # tested fetch + VERBATIM prompt

import litellm
litellm.request_timeout = 120  # no indefinite hang

# verified pricing USD/token (provider listings 2026-06); same source as compute_api_cost.py
PRICE = {
    "openrouter/anthropic/claude-opus-4.7": (5e-6, 25e-6),
    "openrouter/anthropic/claude-haiku-4.5": (1e-6, 5e-6),
}
EPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EPMC_FT = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"

# fulltext variant: same taxonomy/criteria as GLM_SYS, but reads full text not abstract.
FT_SYS = GLM_SYS.replace("title+abstract", "title and FULL TEXT").replace("the abstract", "the paper")

import re as _re
def _xml_to_text(xml: str) -> str:
    """Strip tags; prefer Results/Discussion/Conclusion where claim support lives."""
    body = xml
    m = _re.search(r"<body\b.*?>(.*?)</body>", xml, _re.S | _re.I)
    if m:
        body = m.group(1)
    txt = _re.sub(r"<[^>]+>", " ", body)
    txt = _re.sub(r"\s+", " ", txt).strip()
    return txt


def fetch_fulltext(id_type, pid):
    """Best-effort OA full text via Europe PMC; (text, True) or (None, False).
    PMID -> resolve PMCID (must be inEPMC) -> GET /rest/{PMCID}/fullTextXML."""
    if id_type != "PMID":
        return None, False  # NCT has no OA full text route
    try:
        j = requests.get(EPMC_SEARCH, params={"query": f"EXT_ID:{pid} AND SRC:MED",
                         "format": "json", "resultType": "core", "pageSize": 1}, timeout=30).json()
        res = j.get("resultList", {}).get("result", [])
        if not res:
            return None, False
        pmcid = res[0].get("pmcid")
        if not pmcid or res[0].get("inEPMC") != "Y":
            return None, False
        r = requests.get(EPMC_FT.format(pmcid=pmcid), timeout=30)
        if r.status_code == 200 and len(r.text) > 1200:
            txt = _xml_to_text(r.text)
            if len(txt) > 800:
                return txt[:24000], True
    except Exception:
        pass
    return None, False


def judge(model, sysprompt, title, body, claim):
    """Returns (supports, in_tok, out_tok, reason). Retries transient errors."""
    user = (f"PAPER TITLE: {title}\nPAPER TEXT: {body}\n\n"
            f"ASSISTANT CLAIM (cites this paper): {claim}\n\nDoes the paper support the claim?")
    last = ""
    for attempt in range(5):
        try:
            r = litellm.completion(model=model, temperature=0.0, max_tokens=120,
                                   messages=[{"role": "system", "content": sysprompt},
                                             {"role": "user", "content": user}])
            txt = r.choices[0].message.content or ""
            u = r.usage
            pin = (u.prompt_tokens or 0) if u else 0
            pout = (u.completion_tokens or 0) if u else 0
            sup = "error"
            try:
                s = txt[txt.find("{"):txt.rfind("}") + 1]
                sup = json.loads(s).get("supports", "error")
            except Exception:
                pass
            if sup not in ("yes", "partial", "no"):
                return "error", pin, pout, "parse_fail"
            return sup, pin, pout, ""
        except Exception as e:
            last = str(e)[:120]
            if attempt == 4:
                return "error", 0, 0, "api_" + last
            time.sleep(2 ** attempt)  # 1,2,4,8s backoff


def key(r):
    return (r["set"], r["model"], r["task_id"], r["id"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/cite_audit.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["fullpop", "fulltext"], required=True)
    ap.add_argument("--judge-model", default="openrouter/anthropic/claude-opus-4.7")
    ap.add_argument("--limit", type=int, default=0)        # REQUIRED for fulltext
    ap.add_argument("--budget", type=float, default=120.0) # HARD stop USD for THIS script
    args = ap.parse_args()
    if args.mode == "fulltext" and args.limit <= 0:
        ap.error("--mode fulltext requires an explicit --limit (sample size)")
    cin, cout = PRICE[args.judge_model]
    sysprompt = GLM_SYS if args.mode == "fullpop" else FT_SYS

    # input: only FOUND real citations (exists True). dedup by key, deterministic order.
    recs, seen = [], set()
    for l in open(args.inp):
        r = json.loads(l)
        if r.get("exists") is True and key(r) not in seen:
            seen.add(key(r)); recs.append(r)
    recs.sort(key=key)
    if args.mode == "fulltext":
        step = max(1, len(recs) // args.limit)
        recs = recs[::step][:args.limit]   # even stride across sorted (set,model,task,id)

    # resume: skip keys already judged with a NON-error verdict
    done = {}
    if os.path.exists(args.out):
        for l in open(args.out):
            d = json.loads(l)
            done[(d["set"], d["model"], d["task_id"], d["id"])] = d
    spent = sum(d.get("in_tok", 0) * cin + d.get("out_tok", 0) * cout for d in done.values())
    todo = [r for r in recs if done.get(key(r), {}).get("supports") in (None, "error")
            or key(r) not in done]
    print(f"[{args.mode}] candidates={len(recs)} done={len(done)} todo={len(todo)} "
          f"spent=${spent:.2f} budget=${args.budget} judge={args.judge_model}", flush=True)

    out = open(args.out, "a")
    n = 0
    for r in todo:
        if spent >= args.budget:
            print(f"BUDGET STOP at ${spent:.2f} (>= ${args.budget}); {len(todo)-n} left", flush=True)
            break
        ftype = r["id_type"]
        st, meta = (fetch_pmid(r["id"]) if ftype == "PMID" else fetch_nct(r["id"]))
        base = {k: r[k] for k in ("set", "model", "task_id", "id", "id_type")}
        base["claim"] = r.get("claim", "")[:300]
        base["judge_model"] = args.judge_model
        if st != "found" or not meta:
            rec = {**base, "supports": "error", "reason": "refetch_" + st,
                   "in_tok": 0, "out_tok": 0, "src": "none"}
            json.dump(rec, out); out.write("\n"); out.flush(); continue
        body, is_ft = (None, False)
        if args.mode == "fulltext":
            body, is_ft = fetch_fulltext(ftype, r["id"])
        if not body:
            body = meta.get("abstract", "")
        sup, pin, pout, reason = judge(args.judge_model, sysprompt, meta.get("title", ""), body,
                                       r.get("claim", ""))
        spent += pin * cin + pout * cout
        n += 1
        rec = {**base, "supports": sup, "reason": reason, "in_tok": pin, "out_tok": pout,
               "src": "fulltext" if is_ft else "abstract"}
        json.dump(rec, out); out.write("\n"); out.flush()
        if n % 50 == 0:
            print(f"  judged {n}/{len(todo)} spent=${spent:.2f}", flush=True)
    out.close()
    print(f"DONE mode={args.mode} judged_now={n} total=${spent:.2f} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
