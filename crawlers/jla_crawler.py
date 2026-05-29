"""James Lind Alliance (JLA) Priority Setting Partnership crawler.

JLA PSPs are the medical analog of ResearchMath's AIM-workshop open-problem lists:
patient+clinician consensus panels publish a ranked "Top 10 unanswered questions"
per condition. Unlike keyword-mined "future directions", these are *structurally*
open (declared so by an expert panel) and carry a built-in importance ranking — so
they skip Stage-1 status verification and give an expert difficulty/importance prior.

Site structure (scouted 2026-05):
  - Listing: /priority-setting-partnerships?page=N  (paginated; ~10 PSPs/page)
  - Each PSP: /priority-setting-partnerships/{slug}
      → <div id="tab-top-10-priorities"> ... <ol><li>question</li>...</ol>
  - 403s WebFetch; needs a browser User-Agent.

Output: questions in the refiner's INPUT schema (already self-contained, so they
bypass extraction and flow straight into refine + status_verifier).

Usage:
    python crawlers/jla_crawler.py --out data/raw/jla/jla_questions.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import requests

BASE = "https://www.jla.nihr.ac.uk"
LIST = BASE + "/priority-setting-partnerships"
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
SLUG_RE = re.compile(r'href="(/priority-setting-partnerships/[a-z0-9-]+)"')
LI_RE = re.compile(r"<li[^>]*>(.*?)</li>", re.S)
H1_RE = re.compile(r"<h1[^>]*>(.*?)</h1>", re.S)
PANEL_RE = re.compile(r'id="tab-top-10-priorities".*?>(.*)', re.S)


def _clean(html: str) -> str:
    txt = re.sub(r"<[^>]+>", " ", html)
    txt = (txt.replace("&amp;", "&").replace("&#039;", "'").replace("&quot;", '"')
              .replace("&nbsp;", " ").replace("&gt;", ">").replace("&lt;", "<"))
    return re.sub(r"\s+", " ", txt).strip()


class JLACrawler:
    def __init__(self, delay: float = 0.5, max_pages: int = 40):
        self.delay = delay
        self.max_pages = max_pages
        self.s = requests.Session()
        self.s.headers["User-Agent"] = UA

    def _get(self, url: str) -> str:
        for _ in range(3):
            try:
                r = self.s.get(url, timeout=30)
                if r.status_code == 200:
                    return r.text
            except requests.RequestException:
                pass
            time.sleep(1.0)
        return ""

    def list_psps(self) -> list[str]:
        slugs: list[str] = []
        seen = set()
        for page in range(self.max_pages):
            html = self._get(f"{LIST}?page={page}")
            found = [m for m in SLUG_RE.findall(html)]
            new = [s for s in found if s not in seen and s != "/priority-setting-partnerships"]
            if not new:
                break
            for s in new:
                seen.add(s)
                slugs.append(s)
            time.sleep(self.delay)
        return slugs

    def parse_psp(self, slug: str) -> list[dict]:
        url = BASE + slug
        html = self._get(url)
        if not html:
            return []
        title_m = H1_RE.search(html)
        title = _clean(title_m.group(1)) if title_m else slug.rsplit("/", 1)[-1]
        panel_m = PANEL_RE.search(html)
        if not panel_m:
            return []
        # restrict to the panel before the next major tab/footer
        panel = panel_m.group(1)
        panel = re.split(r'id="tab-|<footer', panel)[0]
        items = [_clean(li) for li in LI_RE.findall(panel)]
        items = [q for q in items if q.endswith("?") and len(q) > 25]

        slug_name = slug.rsplit("/", 1)[-1]
        out = []
        for rank, q in enumerate(items, 1):
            out.append({
                "source": "jla",
                "source_id": f"JLA:{slug_name}#{rank}",
                "source_url": url + "#tab-top-10-priorities",
                "source_title": f"JLA PSP: {title}",
                "original_question": q,
                "self_contained_question": q,
                "question_type": "",
                "clinical_domain": title,
                "why_open": ("Ranked among the top unanswered questions by a James Lind "
                             "Alliance Priority Setting Partnership (patient + carer + "
                             "clinician consensus); open by expert declaration."),
                "metadata": {"psp_slug": slug_name, "jla_rank": rank,
                             "expert_curated": True, "in_top_10": rank <= 10},
            })
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/jla/jla_questions.jsonl")
    ap.add_argument("--delay", type=float, default=0.5)
    ap.add_argument("--max-pages", type=int, default=40)
    ap.add_argument("--limit-psps", type=int, default=0)
    args = ap.parse_args()

    c = JLACrawler(delay=args.delay, max_pages=args.max_pages)
    slugs = c.list_psps()
    if args.limit_psps:
        slugs = slugs[: args.limit_psps]
    print(f"Found {len(slugs)} PSPs")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total, empty = 0, 0
    with out_path.open("w") as f:
        for i, slug in enumerate(slugs, 1):
            qs = c.parse_psp(slug)
            if not qs:
                empty += 1
            for q in qs:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")
            total += len(qs)
            if i % 20 == 0:
                print(f"  {i}/{len(slugs)} PSPs, {total} questions")
            time.sleep(args.delay)

    print(f"DONE: {total} questions from {len(slugs)-empty}/{len(slugs)} PSPs "
          f"({empty} empty) → {out_path}")


if __name__ == "__main__":
    main()
