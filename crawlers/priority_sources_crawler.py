"""Authoritative priority-setting source crawler (societies / journals / consensus).

Harvests papers from authoritative bodies that ENUMERATE open research questions:
society research agendas, consensus/scientific/position statements, Delphi priority
exercises, CHNRI priority-setting, and "top N research priorities" papers. These flow
through the existing extract->refine->audit pipeline (they contain listed open questions).

Uses Europe PMC (reliable, concurrency-safe, no NCBI 429). Output: raw documents in the
crawler schema (source_id=PMID, source_title, text=abstract) for the extractor.

Usage:
    python crawlers/priority_sources_crawler.py --out data/raw/priority/documents.jsonl \
        --max-per-query 60
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

# Authoritative priority-setting queries. Each targets a venue/instrument that publishes
# enumerated open research questions from an authoritative body.
QUERIES = [
    '"research priorities" AND ("consensus statement" OR "scientific statement" OR "position statement")',
    '"research agenda" AND (society OR association OR college) AND medicine',
    '"top 10 research priorities" OR "top ten research priorities"',
    '"priority setting partnership" NOT "James Lind"',   # JLA already crawled separately
    'CHNRI AND "research priorities"',
    'Delphi AND "research priorities" AND (clinical OR medical OR health)',
    '"unanswered questions" AND ("consensus" OR "expert panel") AND treatment',
    '"future research" AND "Grand Rounds"',
    '"research recommendations" AND guideline AND (NICE OR WHO OR NIH OR ESMO OR ASCO OR AHA)',
    '"knowledge gaps" AND "systematic review" AND "implications for research"',
    # --- added: broaden authoritative venues/instruments for OPEN questions ---
    '"R&D Blueprint" AND WHO AND ("research priorities" OR "priority pathogens" OR roadmap)',
    'CHNRI AND ("global health" OR "child health" OR "newborn") AND "research priorities"',
    '("National Academies" OR NASEM OR "Institute of Medicine") AND "research agenda" AND ("consensus" OR committee)',
    'PCORI AND ("research priorities" OR "research agenda" OR "priority topics")',
    '("ESMO" OR "ASCO" OR "American Society of Clinical Oncology") AND ("research priorities" OR "research agenda" OR "unmet needs")',
    '("AHA" OR "American Heart Association" OR "American College of Cardiology") AND "scientific statement" AND ("research priorities" OR "knowledge gaps")',
    '("IDSA" OR "Infectious Diseases Society") AND ("research agenda" OR "research priorities" OR "unmet needs")',
    '("American Thoracic Society" OR "European Respiratory Society" OR ERS OR ATS) AND ("research statement" OR "research priorities")',
    '("EASL" OR "AASLD" OR "American College of Rheumatology" OR ACR) AND ("research agenda" OR "research priorities")',
    '("research priorities" OR "research agenda") AND (Lancet OR "New England Journal" OR NEJM OR JAMA)',
    'Delphi AND consensus AND ("unanswered questions" OR "research uncertainties" OR "priority questions")',
    '"uncertainties" AND review AND ("future research" OR "research needed") AND clinical',
    '"research gaps" AND ("expert consensus" OR "working group" OR "task force") AND (treatment OR diagnosis OR management)',
    '"priority research questions" AND (consensus OR Delphi OR "expert panel")',
]


def search(query: str, n: int) -> list[dict]:
    out, cursor = [], "*"
    while len(out) < n:
        try:
            r = requests.get(EPMC, params={
                "query": f"({query}) AND SRC:MED AND (HAS_ABSTRACT:Y)",
                "format": "json", "resultType": "core", "pageSize": min(100, n - len(out)),
                "cursorMark": cursor, "sort": "CITED desc",
            }, timeout=40)
            j = r.json()
        except Exception as e:
            print(f"  query failed: {e}")
            break
        res = j.get("resultList", {}).get("result", [])
        if not res:
            break
        for x in res:
            if x.get("pmid") and x.get("abstractText"):
                out.append({"pmid": x["pmid"], "title": x.get("title", ""),
                            "abstract": x["abstractText"], "year": x.get("pubYear", ""),
                            "journal": (x.get("journalInfo", {}) or {}).get("journal", {}).get("title", "")})
        nc = j.get("nextCursorMark")
        if not nc or nc == cursor:
            break
        cursor = nc
        time.sleep(0.15)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/priority/documents.jsonl")
    ap.add_argument("--max-per-query", type=int, default=60)
    args = ap.parse_args()

    seen, docs = set(), []
    for q in QUERIES:
        hits = search(q, args.max_per_query)
        new = [h for h in hits if h["pmid"] not in seen]
        for h in new:
            seen.add(h["pmid"])
        docs.extend(new)
        print(f"  [{len(new):3d} new / {len(hits):3d}] {q[:60]}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for d in docs:
            f.write(json.dumps({
                "source": "priority",
                "source_id": f"PMID:{d['pmid']}",
                "source_url": f"https://pubmed.ncbi.nlm.nih.gov/{d['pmid']}/",
                "source_title": d["title"],
                "text": d["abstract"],
                "metadata": {"publication_date": d["year"], "journal": d["journal"],
                             "venue_type": "authoritative_priority_setting"},
            }, ensure_ascii=False) + "\n")
    print(f"DONE: {len(docs)} unique authoritative priority docs -> {out}")


if __name__ == "__main__":
    main()
