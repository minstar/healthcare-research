"""Exp B2 — 2-level citation audit on MODEL TRAJECTORIES (the paper's heart).

Distinct from the gold-citation audit (audit_citation_relevance_llm.py): this audits the
citations models EMIT in their agentic answers, mirroring ResearchMath's trace factuality
but one level deeper:

  L1 existence  : does the cited PMID/NCT actually exist?            (Europe PMC / CT.gov; no GPU)
  L2 support    : does that real paper support the local claim?      (GLM judge; needs serving)
  wrong-paper   : real ID whose paper does NOT support the claim (L1 ok & L2 == no)

Pipeline (resumable, append+fsync):
  1) --no-judge : extract cited IDs + local claim snippet + L1 existence -> <out>
  2) (GLM up)   : re-run with --base ... : fills L2 support for records lacking it

Usage:
  python scripts/exp_cite_audit.py --no-judge --out results/cite_audit.jsonl
  OPENAI_API_KEY=dummy python scripts/exp_cite_audit.py --base http://<glm>:8000/v1 \
      --out results/cite_audit.jsonl --sample-per-model 400
"""
from __future__ import annotations
import argparse, json, os, re, threading
from concurrent.futures import ThreadPoolExecutor

import requests

EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
CTG = "https://clinicaltrials.gov/api/v2/studies/"
PMID_RE = re.compile(r"PMID[:\s#]*(\d{5,9})", re.I)
NCT_RE = re.compile(r"(NCT\d{8})")
SENT = re.compile(r"[^.!?]*[.!?]")

SETS = {"1969": ["baseline_glm_1969", "baseline_qwen_1969", "baseline_dsv4_1969"],
        "priority": ["priority_glm", "priority_qwen", "priority_dsv4"],
        "expand": ["expand_glm", "expand_qwen", "expand_dsv4"]}
MODEL_OF = {"glm": "GLM-5.1", "qwen": "Qwen3.6", "dsv4": "DeepSeek-V4"}

SYS = """You verify a CITATION emitted by a medical research assistant. Given a paper's \
title+abstract and the assistant's local claim that cites it, decide whether the paper \
actually SUPPORTS that claim. Output ONLY JSON: {"supports":"yes|partial|no","why":"<=15 words"}
- yes: the abstract clearly supports the claim.
- partial: related topic but the specific claim is not clearly supported.
- no: unrelated, or the claim misrepresents the paper."""
V = {"yes": 1.0, "partial": 0.5, "no": 0.0}


def claim_snippet(answer: str, span: tuple[int, int]) -> str:
    """The sentence containing the citation (the model's local claim)."""
    for m in SENT.finditer(answer):
        if m.start() <= span[0] < m.end():
            return m.group(0).strip()[:400]
    return answer[max(0, span[0] - 200):span[1] + 100].strip()[:400]


def extract_cites(answer: str):
    out = []
    for m in PMID_RE.finditer(answer):
        out.append({"id_type": "PMID", "id": m.group(1), "claim": claim_snippet(answer, m.span())})
    for m in NCT_RE.finditer(answer):
        out.append({"id_type": "NCT", "id": m.group(1), "claim": claim_snippet(answer, m.span())})
    # dedup by (type,id) within an answer
    seen, uniq = set(), []
    for c in out:
        k = (c["id_type"], c["id"])
        if k not in seen:
            seen.add(k); uniq.append(c)
    return uniq


# Existence fetch returns a TRISTATE so a transient API failure is NOT mislabeled as
# fabrication: ("found", meta) | ("notfound", None) | ("error", None). Only a clean HTTP-200
# empty result counts as not-found; exceptions/non-200 are retried, then surfaced as "error"
# (exists=None -> re-checked on the next resumable pass, never counted as fabricated).
import time as _time


def _get(url, params, tries=4):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return {}
            # 429 / 5xx -> backoff and retry
        except Exception:
            pass
        _time.sleep(0.6 * (i + 1))
    return None  # exhausted -> error


def fetch_pmid(pmid):
    j = _get(EPMC, {"query": f"EXT_ID:{pmid} AND SRC:MED", "format": "json",
                    "resultType": "core", "pageSize": 1})
    if j is None:
        return ("error", None)
    res = j.get("resultList", {}).get("result", [])
    if not res:
        return ("notfound", None)
    x = res[0]
    return ("found", {"title": x.get("title", ""), "abstract": (x.get("abstractText") or "")[:1500]})


def fetch_nct(nct):
    j = _get(CTG + nct, {"format": "json"})
    if j is None:
        return ("error", None)
    p = j.get("protocolSection")
    if not p:
        return ("notfound", None)
    idm = p.get("identificationModule", {}); dm = p.get("descriptionModule", {})
    return ("found", {"title": idm.get("briefTitle", ""), "abstract": (dm.get("briefSummary") or "")[:1500]})


def collect_records():
    recs = []
    for setname, dirs in SETS.items():
        for d in dirs:
            p = f"results/{d}/traces.jsonl"
            if not os.path.exists(p):
                continue
            model = MODEL_OF[[k for k in ("glm", "qwen", "dsv4") if k in d][0]]
            for l in open(p):
                t = json.loads(l)
                for c in extract_cites(t.get("model_answer") or ""):
                    recs.append({"set": setname, "model": model, "task_id": t.get("task_id"), **c})
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/cite_audit.jsonl")
    ap.add_argument("--base", default=None)            # GLM endpoint -> enables L2
    ap.add_argument("--model", default="glm-5.1")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--sample-per-model", type=int, default=0)   # 0 = all
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    # resume: key = (set,model,task_id,id)
    done = {}
    if os.path.exists(args.out):
        for l in open(args.out):
            try:
                r = json.loads(l); done[(r["set"], r["model"], r["task_id"], r["id"])] = r
            except Exception:
                pass

    recs = collect_records()
    # keep latest known state per key (so L1 records get upgraded with L2)
    bykey = {}
    for r in recs:
        bykey[(r["set"], r["model"], r["task_id"], r["id"])] = r
    todo = []
    for k, r in bykey.items():
        prev = done.get(k)
        if prev and prev.get("exists") is not None and (args.no_judge or prev.get("supports") not in (None, "error")):
            continue
        if prev:
            r = {**r, **{kk: prev[kk] for kk in ("exists", "title") if kk in prev}}
        todo.append(r)
    print(f"citations extracted: {len(bykey)} | already complete: {len(bykey)-len(todo)} | to process: {len(todo)}",
          flush=True)

    cli = None
    if args.base and not args.no_judge:
        from openai import OpenAI
        cli = OpenAI(base_url=args.base, api_key="dummy")

    lock = threading.Lock()
    fh = open(args.out, "a")
    cnt = [0]

    def fetch(r):
        return fetch_pmid(r["id"]) if r["id_type"] == "PMID" else fetch_nct(r["id"])

    def work(r):
        meta = None
        if r.get("exists") is None:
            status, meta = fetch(r)
            # error -> exists stays None (re-checked next pass; NEVER counted as fabricated)
            r["exists"] = True if status == "found" else (False if status == "notfound" else None)
            r["fetch_status"] = status
            r["title"] = (meta or {}).get("title", "")
        # L2 support
        if cli is not None and r["exists"] and r.get("supports") in (None, "error"):
            if meta is None:
                _, meta = fetch(r)
            if meta and meta.get("abstract"):
                user = (f"PAPER TITLE: {meta['title']}\nABSTRACT: {meta['abstract']}\n\n"
                        f"ASSISTANT CLAIM (cites this paper): {r['claim']}\n\nDoes the paper support the claim?")
                try:
                    resp = cli.chat.completions.create(model=args.model, max_tokens=2000, temperature=0.0,
                        messages=[{"role": "system", "content": SYS}, {"role": "user", "content": user}])
                    m = re.search(r"\{.*\}", resp.choices[0].message.content or "", re.S)
                    r["supports"] = str(json.loads(m.group(0)).get("supports", "")).lower() if m else "parse_fail"
                except Exception as e:
                    r["supports"] = "error"; r["err"] = str(e)[:80]
            else:
                r["supports"] = "no_abstract"
        with lock:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n"); fh.flush(); os.fsync(fh.fileno())
            cnt[0] += 1
            if cnt[0] % 100 == 0:
                print(f"  {cnt[0]}/{len(todo)}", flush=True)

    # optional per-model sampling (bounds L2 cost)
    if args.sample_per_model:
        import collections
        bym = collections.defaultdict(list)
        for r in todo:
            bym[r["model"]].append(r)
        todo = []
        for m, rs in bym.items():
            todo += rs[: args.sample_per_model]
        print(f"sampled to {len(todo)} ({args.sample_per_model}/model)", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, todo))
    fh.close()
    print(f"done -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
