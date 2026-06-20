"""Exp ② — cross-family reliability of the L2 wrong-paper judge.

The primary L2 judge is GLM-5.1. To bound how much the headline 19.9% wrong-paper rate
depends on that single judge, we re-judge a stratified sample with an INDEPENDENT,
different-family judge (Claude via CLI) and report Cohen's kappa on the 3-way verdict
(yes/partial/no) and on the binary wrong-paper (no vs not-no). NOT human validation — a
cross-family LLM proxy; the same sample is also exported as a CSV for later human annotation.

Usage: python scripts/exp_cite_kappa.py --n 120 --model sonnet --out results/cite_kappa
"""
from __future__ import annotations
import argparse, json, os, random, re, subprocess, csv
from concurrent.futures import ThreadPoolExecutor
import requests

EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
CTG = "https://clinicaltrials.gov/api/v2/studies/"
SYS = ("You verify a CITATION emitted by a medical research assistant. Given a paper's "
       "title+abstract and the assistant's local claim citing it, decide whether the paper "
       'SUPPORTS the claim. Output ONLY JSON: {"supports":"yes|partial|no"}\n'
       "- yes: abstract clearly supports the claim.\n- partial: related but specific claim not clearly supported.\n"
       "- no: unrelated or misrepresents the paper.")


def fetch(idt, idv):
    try:
        if idt == "PMID":
            r = requests.get(EPMC, params={"query": f"EXT_ID:{idv} AND SRC:MED", "format": "json",
                                           "resultType": "core", "pageSize": 1}, timeout=25)
            res = r.json().get("resultList", {}).get("result", [])
            if not res: return None
            return {"title": res[0].get("title", ""), "abstract": (res[0].get("abstractText") or "")[:1500]}
        else:
            r = requests.get(CTG + idv, params={"format": "json"}, timeout=25)
            if r.status_code != 200: return None
            p = r.json().get("protocolSection", {})
            return {"title": p.get("identificationModule", {}).get("briefTitle", ""),
                    "abstract": (p.get("descriptionModule", {}).get("briefSummary") or "")[:1500]}
    except Exception:
        return None


def claude_judge(model, title, abstract, claim):
    user = f"PAPER TITLE: {title}\nABSTRACT: {abstract}\n\nASSISTANT CLAIM (cites this paper): {claim}\n\nDoes the paper support the claim?"
    try:
        r = subprocess.run(["claude", "--print", "--model", model, "--system-prompt", SYS],
                           input=user, capture_output=True, text=True, timeout=180)
        m = re.search(r"\{.*\}", r.stdout, re.S)
        return json.loads(m.group(0)).get("supports", "parse_fail") if m else "parse_fail"
    except Exception as e:
        return "error"


def kappa(a, b, cats):
    # Cohen's kappa
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default="results/cite_audit.jsonl")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--out", default="results/cite_kappa")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    bk = {}
    for l in open(args.audit):
        r = json.loads(l); bk[(r["set"], r["model"], r["task_id"], r["id"])] = r
    judged = [r for r in bk.values() if r.get("supports") in ("yes", "partial", "no")]
    # stratify by GLM verdict so 'no' (wrong-paper) is well represented
    random.seed(13)
    by = {"yes": [], "partial": [], "no": []}
    for r in judged: by[r["supports"]].append(r)
    per = max(1, args.n // 3)
    samp = []
    for k in by:
        random.shuffle(by[k]); samp += by[k][:per]
    print(f"sampling {len(samp)} (stratified) of {len(judged)} L2-judged", flush=True)

    def work(r):
        meta = fetch(r["id_type"], r["id"])
        if not meta or not meta.get("abstract"):
            return None
        cv = claude_judge(args.model, meta["title"], meta["abstract"], r.get("claim", ""))
        return {"set": r["set"], "model": r["model"], "id": r["id"], "claim": r.get("claim", "")[:300],
                "glm": r["supports"], "claude": cv, "title": meta["title"][:200]}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        res = [x for x in ex.map(work, samp) if x]
    res = [r for r in res if r["claude"] in ("yes", "partial", "no")]

    with open(f"{args.out}/pairs.jsonl", "w") as f:
        for r in res: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # human-annotation CSV (claude col blanked for a human to fill independently)
    with open(f"{args.out}/human_annotation.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "claim", "paper_title", "human_supports(yes/partial/no)"])
        for r in res: w.writerow([r["id"], r["claim"], r["title"], ""])

    g = [r["glm"] for r in res]; c = [r["claude"] for r in res]
    k3 = kappa(g, c, ["yes", "partial", "no"])
    gb = ["wrong" if x == "no" else "ok" for x in g]; cb = ["wrong" if x == "no" else "ok" for x in c]
    kb = kappa(gb, cb, ["wrong", "ok"])
    agree = sum(1 for x, y in zip(g, c) if x == y) / len(res)
    wrong_glm = gb.count("wrong") / len(res); wrong_claude = cb.count("wrong") / len(res)
    summary = {"n": len(res), "raw_agreement_3way": round(agree, 3),
               "cohen_kappa_3way": round(k3, 3), "cohen_kappa_wrongpaper_binary": round(kb, 3),
               "wrongpaper_rate_glm": round(wrong_glm, 3), "wrongpaper_rate_claude": round(wrong_claude, 3)}
    json.dump(summary, open(f"{args.out}/kappa_summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print(f"-> {args.out}/ (pairs.jsonl, human_annotation.csv, kappa_summary.json)")


if __name__ == "__main__":
    main()
