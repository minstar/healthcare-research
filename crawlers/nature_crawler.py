"""Nature/Science/Cell journal crawler for frontier medical challenges.

Targets high-impact journal publications about:
- Incurable diseases (불치병): cancer, ALS, Alzheimer's, Parkinson's, etc.
- Grand challenges in medicine
- Hair loss / alopecia (탈모) research frontiers
- Nature Reviews, Nature Medicine, Nature Biotechnology
- Science Translational Medicine
"""
from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from typing import Iterator

import requests

from .base import BaseCrawler, RawDocument
from .pubmed_crawler import ESEARCH_URL, EFETCH_URL

logger = logging.getLogger(__name__)

INCURABLE_DISEASE_QUERIES = [
    # 불치병 / incurable conditions
    '"incurable" AND (disease OR condition) AND (research OR treatment)',
    '"no cure" AND (disease OR disorder) AND (mechanism OR therapy)',
    '"treatment-resistant" AND (cancer OR depression OR epilepsy)',
    '"refractory" AND (disease OR cancer) AND (novel therapy OR approach)',

    # Specific incurable/hard-to-treat diseases
    '"Alzheimer disease" AND (unsolved OR mystery OR mechanism unclear)',
    '"amyotrophic lateral sclerosis" AND (open question OR pathogenesis)',
    '"Parkinson disease" AND (cure OR disease-modifying therapy) AND review',
    '"Huntington disease" AND (therapeutic target OR open problem)',
    '"multiple sclerosis" AND (progressive OR unresolved question)',
    '"pancreatic cancer" AND (resistance OR poor prognosis OR challenge)',
    '"glioblastoma" AND (treatment failure OR recurrence mechanism)',
    '"prion disease" AND (therapy OR treatment OR mechanism)',
    '"systemic lupus erythematosus" AND (pathogenesis OR unsolved)',

    # 탈모 (hair loss)
    '"alopecia" AND (mechanism OR cure OR regeneration) AND review',
    '"hair loss" AND (androgenetic OR areata) AND (unsolved OR treatment gap)',
    '"hair follicle" AND (regeneration OR stem cell) AND challenge',

    # Grand challenges / frontier
    '"grand challenge" AND (medicine OR health) AND 2024:2026[dp]',
    '"unsolved mystery" AND (biology OR medicine)',
    '"holy grail" AND (medicine OR drug discovery OR therapy)',
    '"moonshot" AND (cancer OR medicine OR cure)',
]

NATURE_JOURNAL_QUERIES = [
    # Nature family journals
    '("Nature Medicine"[journal] OR "Nature Reviews"[journal]) AND "open question" AND 2023:2026[dp]',
    '("Nature Medicine"[journal] OR "Nature Biotechnology"[journal]) AND "remains unclear" AND 2023:2026[dp]',
    '("Nature"[journal]) AND "grand challenge" AND (medicine OR health) AND 2023:2026[dp]',
    '("Science"[journal] OR "Science Translational Medicine"[journal]) AND "unsolved" AND (medical OR clinical) AND 2023:2026[dp]',
    '("Cell"[journal]) AND "open question" AND (disease OR therapy) AND 2023:2026[dp]',
    '("Lancet"[journal]) AND "knowledge gap" AND 2023:2026[dp]',
    '("New England Journal of Medicine"[journal]) AND "unanswered" AND 2023:2026[dp]',
    '("BMJ"[journal]) AND "evidence gap" AND 2023:2026[dp]',
]


class NatureCrawler(BaseCrawler):
    """Crawls high-impact journals via PubMed for incurable disease research and grand challenges."""

    def __init__(self, output_dir, email: str = "", rate_limit_sec: float = 0.4):
        super().__init__("nature", output_dir, rate_limit_sec)
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
        return resp.json().get("esearchresult", {}).get("idlist", [])

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
                    logger.warning(f"[nature] Parse failed: {e}")
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

        journal = art.findtext(".//Journal/Title", "")

        pub_date_elem = art.find(".//Journal/JournalIssue/PubDate")
        pub_date = ""
        if pub_date_elem is not None:
            y = pub_date_elem.findtext("Year", "")
            m = pub_date_elem.findtext("Month", "")
            pub_date = f"{y}-{m}" if m else y

        keywords = []
        for kw in medline.findall(".//MeshHeadingList/MeshHeading/DescriptorName"):
            keywords.append(kw.text or "")

        return RawDocument(
            source="nature",
            source_id=f"PMID:{pmid}",
            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            title=title,
            abstract=abstract,
            authors=authors,
            publication_date=pub_date,
            document_type=f"Journal Article ({journal})",
            keywords=keywords,
            metadata={"pmid": pmid, "journal": journal},
        )

    def crawl(self, queries: list[str], max_results: int) -> Iterator[RawDocument]:
        all_queries = INCURABLE_DISEASE_QUERIES + NATURE_JOURNAL_QUERIES + queries
        per_query = max(max_results // len(all_queries), 20)
        seen_ids: set[str] = set(self.checkpoint.state.get("collected_ids", []))
        total = self.checkpoint.total_collected

        for query in all_queries:
            if self.checkpoint.is_query_done(query):
                logger.info(f"[nature] Skipping completed query: {query[:60]}...")
                continue

            self.checkpoint.mark_query_started(query)
            logger.info(f"[nature] Searching: {query[:80]}... (max {per_query})")

            try:
                pmids = self._search_ids(query, per_query)
            except Exception as e:
                self.checkpoint.record_error(f"Search failed: {e}")
                logger.error(f"[nature] Search error: {e}")
                continue

            new_pmids = [p for p in pmids if p not in seen_ids]
            seen_ids.update(new_pmids)

            if not new_pmids:
                self.checkpoint.mark_query_done(query)
                continue

            try:
                for doc in self._fetch_articles(new_pmids):
                    self.checkpoint.add_collected_id(doc.metadata.get("pmid", ""))
                    yield doc
                    total += 1
            except Exception as e:
                self.checkpoint.record_error(f"Fetch failed: {e}")
                logger.error(f"[nature] Fetch error: {e}")
                continue

            self.checkpoint.mark_query_done(query)
            self.rate_limit()

        logger.info(f"[nature] Total documents fetched: {total}")
