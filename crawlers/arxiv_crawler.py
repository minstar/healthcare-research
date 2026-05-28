"""arXiv crawler for biomedical/medical open-problem papers.

Targets q-bio.*, cs.AI (medical), stat.AP (biostatistics) categories.
"""
from __future__ import annotations

import logging
from typing import Iterator

import requests
import xml.etree.ElementTree as ET

from .base import BaseCrawler, RawDocument

logger = logging.getLogger(__name__)

ARXIV_API = "http://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


class ArxivCrawler(BaseCrawler):
    def __init__(self, output_dir, rate_limit_sec: float = 3.0):
        super().__init__("arxiv", output_dir, rate_limit_sec)

    def _search(self, query: str, max_results: int, start: int = 0) -> list[RawDocument]:
        params = {
            "search_query": query,
            "start": start,
            "max_results": min(max_results, 500),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        resp = requests.get(ARXIV_API, params=params, timeout=60)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)

        docs = []
        for entry in root.findall("atom:entry", NS):
            arxiv_id = entry.findtext("atom:id", "", NS).split("/abs/")[-1]
            title = entry.findtext("atom:title", "", NS).strip().replace("\n", " ")
            abstract = entry.findtext("atom:summary", "", NS).strip()
            published = entry.findtext("atom:published", "", NS)[:10]

            authors = []
            for author in entry.findall("atom:author", NS):
                name = author.findtext("atom:name", "", NS)
                if name:
                    authors.append(name)

            categories = []
            for cat in entry.findall("arxiv:primary_category", NS):
                categories.append(cat.get("term", ""))
            for cat in entry.findall("atom:category", NS):
                t = cat.get("term", "")
                if t and t not in categories:
                    categories.append(t)

            pdf_url = ""
            for link in entry.findall("atom:link", NS):
                if link.get("title") == "pdf":
                    pdf_url = link.get("href", "")

            docs.append(RawDocument(
                source="arxiv",
                source_id=arxiv_id,
                url=pdf_url or f"https://arxiv.org/abs/{arxiv_id}",
                title=title,
                abstract=abstract,
                authors=authors,
                publication_date=published,
                document_type="preprint",
                keywords=categories,
                metadata={"arxiv_id": arxiv_id, "categories": categories},
            ))
        return docs

    def crawl(self, queries: list[str], max_results: int) -> Iterator[RawDocument]:
        per_query = max(max_results // len(queries), 50)
        seen_ids: set[str] = set(self.checkpoint.state.get("collected_ids", []))
        total = self.checkpoint.total_collected

        for query in queries:
            if self.checkpoint.is_query_done(query):
                logger.info(f"[arxiv] Skipping completed query: {query[:60]}...")
                continue

            self.checkpoint.mark_query_started(query)
            logger.info(f"[arxiv] Searching: {query} (max {per_query})")
            try:
                docs = self._search(query, per_query)
            except Exception as e:
                self.checkpoint.record_error(f"Query failed: {e}")
                logger.error(f"[arxiv] Query failed, continuing: {e}")
                continue

            for doc in docs:
                if doc.source_id not in seen_ids:
                    seen_ids.add(doc.source_id)
                    self.checkpoint.add_collected_id(doc.source_id)
                    yield doc
                    total += 1

            self.checkpoint.mark_query_done(query)
            self.rate_limit()

        logger.info(f"[arxiv] Total documents fetched: {total}")
