"""medRxiv/bioRxiv crawler for medical preprints with open questions.

Uses the medRxiv API (medrxiv.org/api) for content access.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Iterator

import requests

from .base import BaseCrawler, RawDocument

logger = logging.getLogger(__name__)

MEDRXIV_API = "https://api.medrxiv.org/details/medrxiv"
BIORXIV_API = "https://api.biorxiv.org/details/biorxiv"


class MedRxivCrawler(BaseCrawler):
    def __init__(self, output_dir, rate_limit_sec: float = 1.0):
        super().__init__("medrxiv", output_dir, rate_limit_sec)

    def _fetch_page(self, base_url: str, from_date: str, to_date: str, cursor: int = 0) -> dict:
        url = f"{base_url}/{from_date}/{to_date}/{cursor}/json"
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        return resp.json()

    def _is_open_question_paper(self, title: str, abstract: str) -> bool:
        text = (title + " " + abstract).lower()
        signals = [
            "open question", "unsolved", "unresolved", "knowledge gap",
            "unanswered", "future direction", "remains unclear",
            "poorly understood", "unknown mechanism", "no consensus",
            "limited understanding", "insufficient evidence",
            "understudied", "underexplored", "grand challenge",
            "critical gap", "research priority", "unmet need",
        ]
        return any(s in text for s in signals)

    def _crawl_server(self, base_url: str, server_name: str, max_results: int) -> Iterator[RawDocument]:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=365 * 3)).strftime("%Y-%m-%d")

        cursor = 0
        total = 0
        while total < max_results:
            try:
                data = self._fetch_page(base_url, start_date, end_date, cursor)
            except Exception as e:
                logger.error(f"[{server_name}] Fetch failed at cursor {cursor}: {e}")
                break

            messages = data.get("messages", [{}])
            total_count = messages[0].get("total", 0) if messages else 0
            articles = data.get("collection", [])

            if not articles:
                break

            for art in articles:
                title = art.get("title", "")
                abstract = art.get("abstract", "")

                if not self._is_open_question_paper(title, abstract):
                    continue

                doi = art.get("doi", "")
                yield RawDocument(
                    source=server_name,
                    source_id=doi,
                    url=f"https://doi.org/{doi}" if doi else "",
                    title=title,
                    abstract=abstract,
                    authors=art.get("authors", "").split("; ") if art.get("authors") else [],
                    publication_date=art.get("date", ""),
                    document_type=art.get("type", "preprint"),
                    keywords=[art.get("category", "")],
                    metadata={"doi": doi, "version": art.get("version", "")},
                )
                total += 1
                if total >= max_results:
                    break

            cursor += len(articles)
            if cursor >= total_count:
                break
            self.rate_limit()

        logger.info(f"[{server_name}] Filtered {total} open-question papers")

    def crawl(self, queries: list[str], max_results: int) -> Iterator[RawDocument]:
        half = max_results // 2
        yield from self._crawl_server(MEDRXIV_API, "medrxiv", half)
        yield from self._crawl_server(BIORXIV_API, "biorxiv", half)
