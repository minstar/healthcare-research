"""Cochrane Library crawler for systematic reviews with evidence gaps.

Cochrane reviews often conclude with explicit statements about
insufficient evidence, making them a rich source of open clinical questions.
"""
from __future__ import annotations

import logging
import re
from typing import Iterator

import requests
from bs4 import BeautifulSoup

from .base import BaseCrawler, RawDocument

logger = logging.getLogger(__name__)

COCHRANE_SEARCH_URL = "https://www.cochranelibrary.com/cdsr/reviews"


class CochraneCrawler(BaseCrawler):
    """Crawls Cochrane systematic reviews via their public search interface.

    Falls back to PubMed search for Cochrane reviews if direct access fails.
    """

    def __init__(self, output_dir, rate_limit_sec: float = 2.0):
        super().__init__("cochrane", output_dir, rate_limit_sec)

    def _search_cochrane_via_pubmed(self, query: str, max_results: int) -> list[dict]:
        from .pubmed_crawler import ESEARCH_URL, EFETCH_URL

        full_query = f'({query}) AND "Cochrane Database Syst Rev"[journal]'
        params = {
            "db": "pubmed",
            "term": full_query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance",
        }
        resp = requests.get(ESEARCH_URL, params=params, timeout=30)
        resp.raise_for_status()
        pmids = resp.json().get("esearchresult", {}).get("idlist", [])

        if not pmids:
            return []

        import xml.etree.ElementTree as ET
        params = {
            "db": "pubmed",
            "id": ",".join(pmids[:200]),
            "retmode": "xml",
            "rettype": "abstract",
        }
        resp = requests.get(EFETCH_URL, params=params, timeout=60)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)

        results = []
        for article in root.findall(".//PubmedArticle"):
            medline = article.find(".//MedlineCitation")
            pmid = medline.findtext(".//PMID", "")
            art = medline.find(".//Article")
            title = art.findtext(".//ArticleTitle", "")

            abstract_parts = []
            for ab in art.findall(".//Abstract/AbstractText"):
                label = ab.get("Label", "")
                text = "".join(ab.itertext())
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)

            results.append({
                "pmid": pmid,
                "title": title,
                "abstract": "\n".join(abstract_parts),
            })

        return results

    def _has_evidence_gap(self, abstract: str) -> bool:
        text = abstract.lower()
        gap_signals = [
            "insufficient evidence", "no clear evidence",
            "uncertain", "low-quality evidence", "very low-quality",
            "further research", "more research needed",
            "evidence is limited", "evidence gap",
            "inconclusive", "no firm conclusions",
            "unable to draw", "cannot determine",
            "lack of evidence", "paucity of evidence",
        ]
        return any(s in text for s in gap_signals)

    def crawl(self, queries: list[str], max_results: int) -> Iterator[RawDocument]:
        per_query = max(max_results // len(queries), 100)
        seen_ids: set[str] = set(self.checkpoint.state.get("collected_ids", []))
        total = self.checkpoint.total_collected

        for query in queries:
            if self.checkpoint.is_query_done(query):
                logger.info(f"[cochrane] Skipping completed query: {query[:60]}...")
                continue

            self.checkpoint.mark_query_started(query)
            logger.info(f"[cochrane] Searching via PubMed: {query}")
            try:
                articles = self._search_cochrane_via_pubmed(query, per_query)
            except Exception as e:
                self.checkpoint.record_error(f"Search failed: {e}")
                logger.error(f"[cochrane] Search failed, continuing: {e}")
                continue

            for art in articles:
                pmid = art["pmid"]
                if pmid in seen_ids:
                    continue
                seen_ids.add(pmid)

                if not self._has_evidence_gap(art["abstract"]):
                    continue

                self.checkpoint.add_collected_id(pmid)
                yield RawDocument(
                    source="cochrane",
                    source_id=f"PMID:{pmid}",
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    title=art["title"],
                    abstract=art["abstract"],
                    document_type="Cochrane Systematic Review",
                    metadata={"pmid": pmid, "evidence_gap": True},
                )
                total += 1

            self.checkpoint.mark_query_done(query)
            self.rate_limit()

        logger.info(f"[cochrane] Total evidence-gap reviews: {total}")
