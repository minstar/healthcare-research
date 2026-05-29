"""NICE Research Recommendations crawler (UK).

NICE flags explicit "research recommendations" — declared open questions — while
developing each guideline. They are published in a single static HTML table
(~2,200 rows), each row: Recommendation ID (e.g. CG103/01) + the question text +
a detail-page link. High precision, single-fetch crawlable (no JS, no auth).

Output: refiner-input schema (already self-contained → bypasses extraction).

Usage:
    python crawlers/nice_crawler.py --out data/raw/nice/nice_questions.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import requests

URL = "https://www.nice.org.uk/about/what-we-do/science-policy-research/research-recommendations"
BASE = "https://www.nice.org.uk"
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S)
CELL_RE = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.S)
HREF_RE = re.compile(r'href="([^"]+research-recommendations/[^"]+)"')
ID_RE = re.compile(r"^[A-Z]{2,4}\d{1,4}(?:/\d{1,3})?$")


def _clean(html: str) -> str:
    txt = re.sub(r"<[^>]+>", " ", html)
    for a, b in [("&amp;", "&"), ("&#039;", "'"), ("&quot;", '"'), ("&nbsp;", " "),
                 ("&gt;", ">"), ("&lt;", "<")]:
        txt = txt.replace(a, b)
    return re.sub(r"\s+", " ", txt).strip()


def crawl(url: str = URL) -> list[dict]:
    r = requests.get(url, headers={"User-Agent": UA}, timeout=60)
    r.raise_for_status()
    out, seen = [], set()
    for row in ROW_RE.findall(r.text):
        cells = [_clean(c) for c in CELL_RE.findall(row)]
        if len(cells) < 3:
            continue
        rec_id, name = cells[1], cells[2]
        if not ID_RE.match(rec_id) or rec_id in seen or len(name) < 30:
            continue
        seen.add(rec_id)
        href_m = HREF_RE.search(row)
        href = (BASE + href_m.group(1)) if href_m and href_m.group(1).startswith("/") else (
            href_m.group(1) if href_m else url)
        guidance_ref = rec_id.split("/")[0]
        out.append({
            "source": "nice",
            "source_id": f"NICE:{rec_id}",
            "source_url": href,
            "source_title": f"NICE research recommendation {rec_id} (guidance {guidance_ref})",
            "original_question": name,
            "self_contained_question": name,
            "question_type": "",
            "clinical_domain": "",
            "why_open": ("Formal NICE research recommendation: an evidence gap identified "
                         "during guideline development that current evidence cannot resolve."),
            "metadata": {"recommendation_id": rec_id, "guidance_ref": guidance_ref,
                         "expert_curated": True, "openness_type": "evidence_gap_analysis"},
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/nice/nice_questions.jsonl")
    args = ap.parse_args()
    recs = crawl()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for rec in recs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"DONE: {len(recs)} NICE research recommendations → {out_path}")


if __name__ == "__main__":
    main()
