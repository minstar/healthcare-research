"""Top-journal crawler for ALL unsolved medical/biomedical questions.

Systematically sweeps every major medical domain for open problems
published in high-impact journals (Nature, Science, Cell, Lancet, NEJM,
BMJ, JAMA, etc.) and via PubMed across all specialties.

NOT limited to specific diseases — covers the full breadth of
problems humanity has yet to solve in medicine and life sciences.
"""
from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from typing import Iterator

import requests

from .base import BaseCrawler, RawDocument
from .pubmed_crawler import ESEARCH_URL, EFETCH_URL

logger = logging.getLogger(__name__)

# ── Broad "open question" signal queries across ALL medical domains ──
OPEN_QUESTION_QUERIES = [
    # General open-problem signals
    '"open question" AND (medicine OR clinical OR biomedical OR health) AND review[pt]',
    '"unsolved problem" AND (medicine OR biology OR clinical) AND review[pt]',
    '"unresolved question" AND (disease OR treatment OR diagnosis)',
    '"remains unknown" AND (mechanism OR pathogenesis OR etiology) AND review[pt]',
    '"remains unclear" AND (clinical OR disease OR treatment) AND review[pt]',
    '"poorly understood" AND (disease OR mechanism OR pathophysiology) AND review[pt]',
    '"knowledge gap" AND (medicine OR clinical OR health) AND review[pt]',
    '"unanswered question" AND (medicine OR clinical OR therapy)',
    '"critical gap" AND (research OR knowledge) AND (medical OR clinical)',
    '"unmet need" AND (treatment OR therapy OR diagnostic)',
    '"grand challenge" AND (health OR medicine OR biomedical)',
    '"moonshot" AND (medicine OR cancer OR health OR cure)',
    '"holy grail" AND (medicine OR drug discovery OR therapy OR biology)',
    '"future direction" AND "open question" AND (medicine OR biology)',
    '"no effective treatment" AND (disease OR disorder OR condition) AND review[pt]',
    '"no cure" AND (disease OR disorder) AND (mechanism OR research)',
    '"incurable" AND (disease OR condition) AND (research OR treatment)',
    '"treatment-resistant" AND (disease OR disorder) AND (mechanism OR novel)',
    '"refractory" AND (disease OR condition) AND (therapeutic OR approach)',
    '"limited understanding" AND (disease OR mechanism) AND review[pt]',
    '"fundamental question" AND (biology OR medicine OR neuroscience)',
    '"understudied" AND (disease OR population OR mechanism)',
    '"underexplored" AND (target OR pathway OR mechanism) AND (disease OR therapy)',
    '"research priority" AND (WHO OR NIH OR global health)',
    '"evidence gap" AND (clinical OR medical OR health) AND systematic review[pt]',
    '"insufficient evidence" AND (treatment OR intervention) AND systematic review[pt]',

    # By medical discipline — catch domain-specific open problems
    '"open question" AND oncology AND review[pt]',
    '"open question" AND neuroscience AND review[pt]',
    '"open question" AND cardiology AND review[pt]',
    '"open question" AND immunology AND review[pt]',
    '"open question" AND infectious disease AND review[pt]',
    '"open question" AND psychiatry AND review[pt]',
    '"open question" AND endocrinology AND review[pt]',
    '"open question" AND gastroenterology AND review[pt]',
    '"open question" AND nephrology AND review[pt]',
    '"open question" AND pulmonology AND review[pt]',
    '"open question" AND rheumatology AND review[pt]',
    '"open question" AND dermatology AND review[pt]',
    '"open question" AND ophthalmology AND review[pt]',
    '"open question" AND hematology AND review[pt]',
    '"open question" AND pediatrics AND review[pt]',
    '"open question" AND geriatrics AND review[pt]',
    '"open question" AND surgery AND review[pt]',
    '"open question" AND radiology AND review[pt]',
    '"open question" AND anesthesiology AND review[pt]',
    '"open question" AND emergency medicine AND review[pt]',
    '"open question" AND pharmacology AND review[pt]',
    '"open question" AND genetics AND review[pt]',
    '"open question" AND epidemiology AND review[pt]',
    '"open question" AND public health AND review[pt]',
    '"open question" AND nutrition AND review[pt]',
    '"open question" AND rehabilitation AND review[pt]',
    '"open question" AND palliative care AND review[pt]',
    '"open question" AND reproductive medicine AND review[pt]',
    '"open question" AND transplantation AND review[pt]',

    # Cross-cutting themes
    '"unsolved" AND (aging OR ageing OR longevity OR senescence) AND review[pt]',
    '"unsolved" AND (microbiome OR gut-brain OR dysbiosis)',
    '"unsolved" AND (stem cell OR regeneration OR tissue engineering)',
    '"unsolved" AND (drug resistance OR antimicrobial resistance)',
    '"unsolved" AND (rare disease OR orphan disease)',
    '"unsolved" AND (pain OR chronic pain OR nociception)',
    '"unsolved" AND (sleep OR circadian OR insomnia)',
    '"unsolved" AND (obesity OR metabolic syndrome OR diabetes)',
    '"unsolved" AND (autoimmune OR allergy OR hypersensitivity)',
    '"unsolved" AND (vaccine OR vaccination OR immunization)',
    '"unsolved" AND (biomarker OR early detection OR screening)',
    '"unsolved" AND (gene therapy OR CRISPR OR genome editing)',
    '"unsolved" AND (artificial intelligence OR machine learning) AND (medical OR clinical)',
    '"unsolved" AND (telemedicine OR digital health OR wearable)',
    '"unsolved" AND (health equity OR health disparity OR social determinant)',
    '"unsolved" AND (environmental health OR pollution OR toxicology)',
    '"unsolved" AND (pandemic OR epidemic OR outbreak OR preparedness)',
    '"unsolved" AND (consciousness OR cognition OR memory) AND neuroscience',
    '"unsolved" AND (fertility OR infertility OR contraception)',
    '"unsolved" AND (wound healing OR fibrosis OR scarring)',
    '"unsolved" AND (organ preservation OR cryopreservation OR xenotransplant)',
]

# ── Top journal specific queries ──
TOP_JOURNAL_QUERIES = [
    # Nature family
    '("Nature"[journal]) AND ("open question" OR "unsolved" OR "grand challenge") AND 2020:2026[dp]',
    '("Nature Medicine"[journal]) AND ("open question" OR "remains unclear" OR "unmet need") AND 2020:2026[dp]',
    '("Nature Reviews"[journal]) AND ("open question" OR "key challenge" OR "unresolved") AND 2020:2026[dp]',
    '("Nature Biotechnology"[journal]) AND ("open question" OR "challenge" OR "unsolved") AND 2020:2026[dp]',
    '("Nature Genetics"[journal]) AND ("open question" OR "remains unclear") AND 2020:2026[dp]',
    '("Nature Neuroscience"[journal]) AND ("open question" OR "poorly understood") AND 2020:2026[dp]',
    '("Nature Immunology"[journal]) AND ("open question" OR "remains unclear") AND 2020:2026[dp]',
    '("Nature Cancer"[journal]) AND ("open question" OR "unanswered") AND 2020:2026[dp]',
    '("Nature Microbiology"[journal]) AND ("open question" OR "remains unclear") AND 2020:2026[dp]',
    '("Nature Aging"[journal]) AND ("open question" OR "unsolved") AND 2020:2026[dp]',
    '("Nature Cardiovascular Research"[journal]) AND ("open question" OR "unclear") AND 2020:2026[dp]',
    '("Nature Metabolism"[journal]) AND ("open question" OR "remains unclear") AND 2020:2026[dp]',
    '("Nature Mental Health"[journal]) AND ("open question" OR "unresolved") AND 2020:2026[dp]',

    # Science family
    '("Science"[journal]) AND ("open question" OR "unsolved" OR "grand challenge") AND (medicine OR biology OR health) AND 2020:2026[dp]',
    '("Science Translational Medicine"[journal]) AND ("open question" OR "unmet need") AND 2020:2026[dp]',
    '("Science Immunology"[journal]) AND ("open question" OR "remains unclear") AND 2020:2026[dp]',
    '("Science Advances"[journal]) AND ("open question" OR "unsolved") AND (medical OR clinical) AND 2020:2026[dp]',

    # Cell family
    '("Cell"[journal]) AND ("open question" OR "remains unclear") AND (disease OR therapy) AND 2020:2026[dp]',
    '("Cell Host Microbe"[journal]) AND ("open question" OR "unsolved") AND 2020:2026[dp]',
    '("Cell Stem Cell"[journal]) AND ("open question" OR "remains unclear") AND 2020:2026[dp]',
    '("Cancer Cell"[journal]) AND ("open question" OR "poorly understood") AND 2020:2026[dp]',
    '("Immunity"[journal]) AND ("open question" OR "remains unclear") AND 2020:2026[dp]',
    '("Neuron"[journal]) AND ("open question" OR "remains unclear") AND 2020:2026[dp]',
    '("Molecular Cell"[journal]) AND ("open question" OR "unsolved") AND (disease) AND 2020:2026[dp]',

    # Top clinical journals
    '("New England Journal of Medicine"[journal]) AND ("unanswered" OR "open question" OR "unresolved") AND 2020:2026[dp]',
    '("Lancet"[journal]) AND ("knowledge gap" OR "open question" OR "unanswered") AND 2020:2026[dp]',
    '("BMJ"[journal]) AND ("evidence gap" OR "open question" OR "unanswered") AND 2020:2026[dp]',
    '("JAMA"[journal]) AND ("unanswered" OR "open question" OR "knowledge gap") AND 2020:2026[dp]',
    '("Annals of Internal Medicine"[journal]) AND ("open question" OR "unanswered") AND 2020:2026[dp]',

    # Specialty top journals
    '("Journal of Clinical Oncology"[journal]) AND ("open question" OR "unanswered") AND 2020:2026[dp]',
    '("Circulation"[journal]) AND ("open question" OR "unanswered" OR "unresolved") AND 2020:2026[dp]',
    '("Journal of Neuroscience"[journal]) AND ("open question" OR "unsolved") AND 2020:2026[dp]',
    '("Gut"[journal]) AND ("open question" OR "unanswered") AND 2020:2026[dp]',
    '("Blood"[journal]) AND ("open question" OR "unsolved") AND 2020:2026[dp]',
    '("Journal of Clinical Investigation"[journal]) AND ("open question" OR "remains unclear") AND 2020:2026[dp]',
    '("Diabetes Care"[journal]) AND ("open question" OR "unanswered") AND 2020:2026[dp]',
    '("American Journal of Respiratory and Critical Care Medicine"[journal]) AND ("open question") AND 2020:2026[dp]',
    '("Kidney International"[journal]) AND ("open question" OR "unsolved") AND 2020:2026[dp]',
    '("Hepatology"[journal]) AND ("open question" OR "unanswered") AND 2020:2026[dp]',
]


class NatureCrawler(BaseCrawler):
    """Crawls top journals via PubMed for ALL unsolved medical problems across every specialty."""

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
        all_queries = OPEN_QUESTION_QUERIES + TOP_JOURNAL_QUERIES + queries
        per_query = max(max_results // len(all_queries), 10)
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
