"""PubMed/PMC crawler using NCBI E-utilities API.

Targets review papers mentioning open questions, unsolved problems,
knowledge gaps, etc. in medicine and clinical sciences.
"""
from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from typing import Iterator

import requests

from .base import BaseCrawler, RawDocument

logger = logging.getLogger(__name__)

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


class PubMedCrawler(BaseCrawler):
    def __init__(self, output_dir, email: str, rate_limit_sec: float = 0.4):
        super().__init__("pubmed", output_dir, rate_limit_sec)
        self.email = email

    def _search_ids(self, query: str, max_results: int) -> list[str]:
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "email": self.email,
            "sort": "relevance",
        }
        resp = requests.get(ESEARCH_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("esearchresult", {}).get("idlist", [])

    def _fetch_articles(self, pmids: list[str]) -> Iterator[RawDocument]:
        batch_size = 200
        for i in range(0, len(pmids), batch_size):
            batch = pmids[i : i + batch_size]
            params = {
                "db": "pubmed",
                "id": ",".join(batch),
                "retmode": "xml",
                "rettype": "abstract",
                "email": self.email,
            }
            resp = requests.get(EFETCH_URL, params=params, timeout=60)
            resp.raise_for_status()
            root = ET.fromstring(resp.text)

            for article in root.findall(".//PubmedArticle"):
                try:
                    yield self._parse_article(article)
                except Exception as e:
                    logger.warning(f"Failed to parse article: {e}")
            self.rate_limit()

    def _parse_article(self, article_elem) -> RawDocument:
        medline = article_elem.find(".//MedlineCitation")
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
        abstract = "\n".join(abstract_parts)

        authors = []
        for au in art.findall(".//AuthorList/Author"):
            last = au.findtext("LastName", "")
            fore = au.findtext("ForeName", "")
            if last:
                authors.append(f"{fore} {last}".strip())

        pub_date_elem = art.find(".//Journal/JournalIssue/PubDate")
        pub_date = ""
        if pub_date_elem is not None:
            y = pub_date_elem.findtext("Year", "")
            m = pub_date_elem.findtext("Month", "")
            pub_date = f"{y}-{m}" if m else y

        keywords = []
        for kw in medline.findall(".//MeshHeadingList/MeshHeading/DescriptorName"):
            keywords.append(kw.text or "")
        for kw in medline.findall(".//KeywordList/Keyword"):
            keywords.append("".join(kw.itertext()))

        pub_types = []
        for pt in art.findall(".//PublicationTypeList/PublicationType"):
            pub_types.append(pt.text or "")

        return RawDocument(
            source="pubmed",
            source_id=f"PMID:{pmid}",
            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            title=title,
            abstract=abstract,
            authors=authors,
            publication_date=pub_date,
            document_type="; ".join(pub_types),
            keywords=keywords,
            metadata={"pmid": pmid},
        )

    def crawl(self, queries: list[str], max_results: int) -> Iterator[RawDocument]:
        per_query = max(max_results // len(queries), 100)
        seen_ids: set[str] = set(self.checkpoint.state.get("collected_ids", []))
        total = self.checkpoint.total_collected

        for query in queries:
            if self.checkpoint.is_query_done(query):
                logger.info(f"[pubmed] Skipping completed query: {query[:60]}...")
                continue

            self.checkpoint.mark_query_started(query)
            logger.info(f"[pubmed] Searching: {query} (max {per_query})")

            try:
                pmids = self._search_ids(query, per_query)
            except Exception as e:
                self.checkpoint.record_error(f"Search failed: {e}")
                logger.error(f"[pubmed] Search error, continuing to next query: {e}")
                continue

            new_pmids = [p for p in pmids if p not in seen_ids]
            seen_ids.update(new_pmids)

            if not new_pmids:
                self.checkpoint.mark_query_done(query)
                continue

            try:
                for doc in self._fetch_articles(new_pmids):
                    self.checkpoint.add_collected_id(doc.metadata.get("pmid", doc.source_id))
                    yield doc
                    total += 1
            except Exception as e:
                self.checkpoint.record_error(f"Fetch failed: {e}")
                logger.error(f"[pubmed] Fetch error, continuing to next query: {e}")
                continue

            self.checkpoint.mark_query_done(query)
            self.rate_limit()

        logger.info(f"[pubmed] Total documents fetched: {total}")
        logger.info(f"[pubmed] Checkpoint: {self.checkpoint.summary()}")
