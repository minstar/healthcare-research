#!/usr/bin/env python3
"""Expand PubMed coverage by querying MeSH disease descriptors for open-question reviews.

Fetches review papers mentioning open questions, unsolved problems, or knowledge gaps
across ~100 major MeSH disease categories (C01-C26 top-level and key children).
Deduplicates against existing PubMed corpus and saves results incrementally.

Usage:
    python scripts/track_a/pubmed_mesh_expansion.py [--max-per-query 50] [--resume]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Iterator

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path("/data/project/private/minstar/workspace/healthcare-research")
OUTPUT_DIR = BASE_DIR / "data" / "raw" / "pubmed_mesh"
EXISTING_PUBMED = BASE_DIR / "data" / "raw" / "pubmed" / "documents.jsonl"
CHECKPOINT_PATH = OUTPUT_DIR / ".checkpoint.json"
OUTPUT_PATH = OUTPUT_DIR / "documents.jsonl"

EMAIL = "research-bot@example.org"
RATE_LIMIT_SEC = 0.4

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# Query templates -- each combined with a MeSH term
QUERY_TEMPLATES = [
    '"open question" AND {mesh}[MeSH] AND review[pt]',
    '"unsolved" AND {mesh}[MeSH] AND review[pt]',
    '"knowledge gap" AND {mesh}[MeSH] AND review[pt]',
]

# ---------------------------------------------------------------------------
# ~100 curated MeSH disease terms (top-level C01-C26 + major children)
# ---------------------------------------------------------------------------
MESH_DISEASE_TERMS = [
    # C01 Bacterial Infections and Mycoses
    "Bacterial Infections", "Tuberculosis", "Sepsis", "Mycoses",
    "Staphylococcal Infections", "Clostridium Infections",
    # C02 Virus Diseases
    "Virus Diseases", "HIV Infections", "Hepatitis", "Influenza, Human",
    "COVID-19", "Dengue", "Ebola Virus Disease",
    # C03 Parasitic Diseases
    "Parasitic Diseases", "Malaria", "Leishmaniasis", "Trypanosomiasis",
    # C04 Neoplasms
    "Neoplasms", "Breast Neoplasms", "Lung Neoplasms", "Colorectal Neoplasms",
    "Pancreatic Neoplasms", "Brain Neoplasms", "Leukemia", "Lymphoma",
    "Melanoma", "Liver Neoplasms", "Prostatic Neoplasms",
    # C05 Musculoskeletal Diseases
    "Musculoskeletal Diseases", "Osteoarthritis", "Rheumatoid Arthritis",
    "Osteoporosis", "Sarcopenia",
    # C06 Digestive System Diseases
    "Digestive System Diseases", "Inflammatory Bowel Diseases",
    "Liver Diseases", "Pancreatitis", "Celiac Disease",
    # C07 Stomatognathic Diseases (skip -- small category)
    # C08 Respiratory Tract Diseases
    "Respiratory Tract Diseases", "Asthma", "Pulmonary Disease, Chronic Obstructive",
    "Cystic Fibrosis", "Pulmonary Fibrosis", "Pneumonia",
    # C09 Otorhinolaryngologic Diseases
    "Hearing Loss", "Tinnitus",
    # C10 Nervous System Diseases
    "Nervous System Diseases", "Alzheimer Disease", "Parkinson Disease",
    "Multiple Sclerosis", "Epilepsy", "Amyotrophic Lateral Sclerosis",
    "Stroke", "Migraine Disorders", "Huntington Disease",
    # C11 Eye Diseases
    "Eye Diseases", "Glaucoma", "Macular Degeneration", "Diabetic Retinopathy",
    # C12 Male Urogenital Diseases
    "Kidney Diseases", "Renal Insufficiency, Chronic",
    # C13 Female Urogenital Diseases and Pregnancy Complications
    "Endometriosis", "Pre-Eclampsia", "Polycystic Ovary Syndrome",
    # C14 Cardiovascular Diseases
    "Cardiovascular Diseases", "Heart Failure", "Atherosclerosis",
    "Hypertension", "Atrial Fibrillation", "Myocardial Infarction",
    "Aortic Aneurysm", "Cardiomyopathies",
    # C15 Hemic and Lymphatic Diseases
    "Anemia, Sickle Cell", "Hemophilia A", "Thrombocytopenia",
    # C16 Congenital, Hereditary, and Neonatal Diseases
    "Genetic Diseases, Inborn", "Down Syndrome", "Cystic Fibrosis",
    # C17 Skin and Connective Tissue Diseases
    "Skin Diseases", "Psoriasis", "Lupus Erythematosus, Systemic",
    "Scleroderma, Systemic", "Dermatitis, Atopic",
    # C18 Nutritional and Metabolic Diseases
    "Metabolic Diseases", "Diabetes Mellitus", "Diabetes Mellitus, Type 2",
    "Obesity", "Phenylketonurias",
    # C19 Endocrine System Diseases
    "Endocrine System Diseases", "Thyroid Diseases", "Adrenal Insufficiency",
    # C20 Immune System Diseases
    "Immune System Diseases", "Autoimmune Diseases", "Immunologic Deficiency Syndromes",
    "Graft vs Host Disease",
    # C21 Disorders of Environmental Origin (limited)
    # C22 Animal Diseases (skip)
    # C23 Pathological Conditions, Signs and Symptoms
    "Pain", "Chronic Pain", "Inflammation", "Fibrosis", "Edema",
    # C24 Occupational Diseases (skip -- small)
    # C25 Substance-Related Disorders
    "Substance-Related Disorders", "Opioid-Related Disorders",
    "Alcoholism",
    # C26 Wounds and Injuries
    "Wounds and Injuries", "Brain Injuries, Traumatic", "Spinal Cord Injuries",
    # F03 Mental Disorders (important overlap)
    "Mental Disorders", "Depressive Disorder", "Schizophrenia",
    "Anxiety Disorders", "Bipolar Disorder", "Autistic Disorder",
    "Attention Deficit Disorder with Hyperactivity",
]

# Deduplicate the list (e.g. Cystic Fibrosis appears in C08 and C16)
MESH_DISEASE_TERMS = list(dict.fromkeys(MESH_DISEASE_TERMS))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------
class MeshCheckpoint:
    """Simple checkpoint tracking completed (mesh_term, query_template) pairs."""

    def __init__(self, path: Path):
        self.path = path
        self.state: dict = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            with open(self.path) as f:
                return json.load(f)
        return {
            "started_at": datetime.now().isoformat(),
            "completed_queries": [],  # list of "mesh_term|||template_idx"
            "collected_pmids": [],
            "total_documents": 0,
            "errors": [],
        }

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.state, f, indent=2, ensure_ascii=False)

    def is_done(self, mesh_term: str, template_idx: int) -> bool:
        key = f"{mesh_term}|||{template_idx}"
        return key in self.state["completed_queries"]

    def mark_done(self, mesh_term: str, template_idx: int):
        key = f"{mesh_term}|||{template_idx}"
        if key not in self.state["completed_queries"]:
            self.state["completed_queries"].append(key)
        self.save()

    def add_pmid(self, pmid: str):
        if pmid not in self.state["collected_pmids"]:
            self.state["collected_pmids"].append(pmid)
            self.state["total_documents"] = len(self.state["collected_pmids"])

    def is_collected(self, pmid: str) -> bool:
        return pmid in self.state["collected_pmids"]

    def record_error(self, msg: str):
        self.state.setdefault("errors", []).append({
            "message": msg[:500],
            "timestamp": datetime.now().isoformat(),
        })
        # Keep only last 50 errors
        self.state["errors"] = self.state["errors"][-50:]
        self.save()


# ---------------------------------------------------------------------------
# PubMed helpers (same pattern as crawlers/pubmed_crawler.py)
# ---------------------------------------------------------------------------
def search_pmids(query: str, max_results: int) -> list[str]:
    """Use NCBI esearch to find PMIDs matching a query."""
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "email": EMAIL,
        "sort": "relevance",
    }
    resp = requests.get(ESEARCH_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("esearchresult", {}).get("idlist", [])


def fetch_articles(pmids: list[str]) -> Iterator[dict]:
    """Use NCBI efetch to retrieve article metadata for given PMIDs."""
    batch_size = 200
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i : i + batch_size]
        params = {
            "db": "pubmed",
            "id": ",".join(batch),
            "retmode": "xml",
            "rettype": "abstract",
            "email": EMAIL,
        }
        resp = requests.get(EFETCH_URL, params=params, timeout=60)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)

        for article in root.findall(".//PubmedArticle"):
            try:
                yield parse_article(article)
            except Exception as e:
                logger.warning(f"Failed to parse article: {e}")


def parse_article(article_elem) -> dict:
    """Parse a PubmedArticle XML element into a document dict."""
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

    return {
        "source": "pubmed",
        "source_id": f"PMID:{pmid}",
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        "title": title,
        "abstract": abstract,
        "full_text": "",
        "authors": authors,
        "publication_date": pub_date,
        "document_type": "; ".join(pub_types),
        "keywords": keywords,
        "metadata": {"pmid": pmid, "crawl_source": "mesh_expansion"},
    }


# ---------------------------------------------------------------------------
# Load existing PMIDs (for dedup)
# ---------------------------------------------------------------------------
def load_existing_pmids() -> set[str]:
    """Load PMIDs already in the original PubMed crawl."""
    existing: set[str] = set()
    if EXISTING_PUBMED.exists():
        with open(EXISTING_PUBMED) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                    pmid = doc.get("metadata", {}).get("pmid", "")
                    if pmid:
                        existing.add(pmid)
                    # Also extract from source_id
                    sid = doc.get("source_id", "")
                    if sid.startswith("PMID:"):
                        existing.add(sid.replace("PMID:", ""))
                except json.JSONDecodeError:
                    continue
    logger.info(f"Loaded {len(existing)} existing PMIDs for deduplication")
    return existing


# ---------------------------------------------------------------------------
# Main crawl
# ---------------------------------------------------------------------------
def run_crawl(max_per_query: int, resume: bool):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ckpt = MeshCheckpoint(CHECKPOINT_PATH)
    if not resume:
        # Clear checkpoint and output for a fresh start
        ckpt.state = {
            "started_at": datetime.now().isoformat(),
            "completed_queries": [],
            "collected_pmids": [],
            "total_documents": 0,
            "errors": [],
        }
        ckpt.save()
        # Truncate output file
        OUTPUT_PATH.write_text("")
    else:
        logger.info(
            f"Resuming: {len(ckpt.state['completed_queries'])} queries done, "
            f"{ckpt.state['total_documents']} docs collected"
        )

    # Load existing PMIDs from the original crawl for dedup
    existing_pmids = load_existing_pmids()
    # Also load PMIDs already collected in this expansion
    seen_pmids: set[str] = set(ckpt.state.get("collected_pmids", []))
    existing_pmids.update(seen_pmids)

    total_queries = len(MESH_DISEASE_TERMS) * len(QUERY_TEMPLATES)
    completed = len(ckpt.state["completed_queries"])
    new_docs = 0
    skipped_dup = 0

    logger.info(
        f"Starting MeSH expansion: {len(MESH_DISEASE_TERMS)} terms x "
        f"{len(QUERY_TEMPLATES)} templates = {total_queries} queries"
    )

    for term_idx, mesh_term in enumerate(MESH_DISEASE_TERMS):
        for tmpl_idx, template in enumerate(QUERY_TEMPLATES):
            if ckpt.is_done(mesh_term, tmpl_idx):
                continue

            query = template.format(mesh=mesh_term)
            query_label = f"[{term_idx+1}/{len(MESH_DISEASE_TERMS)}] {mesh_term} (template {tmpl_idx+1})"

            try:
                pmids = search_pmids(query, max_per_query)
            except Exception as e:
                logger.error(f"{query_label}: search failed: {e}")
                ckpt.record_error(f"Search '{query}': {e}")
                time.sleep(RATE_LIMIT_SEC)
                continue

            # Filter out already-seen PMIDs
            new_pmids = [p for p in pmids if p not in existing_pmids]
            if not new_pmids:
                logger.debug(f"{query_label}: {len(pmids)} results, all duplicates")
                ckpt.mark_done(mesh_term, tmpl_idx)
                time.sleep(RATE_LIMIT_SEC)
                continue

            skipped_dup += len(pmids) - len(new_pmids)

            try:
                articles = list(fetch_articles(new_pmids))
            except Exception as e:
                logger.error(f"{query_label}: fetch failed: {e}")
                ckpt.record_error(f"Fetch '{query}': {e}")
                time.sleep(RATE_LIMIT_SEC)
                continue

            # Append to output file
            with open(OUTPUT_PATH, "a") as f:
                for doc in articles:
                    pmid = doc["metadata"]["pmid"]
                    if pmid in existing_pmids:
                        continue
                    existing_pmids.add(pmid)
                    ckpt.add_pmid(pmid)
                    f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                    new_docs += 1

            completed += 1
            ckpt.mark_done(mesh_term, tmpl_idx)

            logger.info(
                f"{query_label}: {len(pmids)} hits, {len(new_pmids)} new, "
                f"{len(articles)} fetched | total: {new_docs} docs "
                f"({completed}/{total_queries} queries)"
            )

            time.sleep(RATE_LIMIT_SEC)

    # Final summary
    logger.info("=" * 60)
    logger.info("MeSH Expansion Complete")
    logger.info(f"  Total new documents: {new_docs}")
    logger.info(f"  Skipped (duplicate PMIDs): {skipped_dup}")
    logger.info(f"  Queries completed: {completed}/{total_queries}")
    logger.info(f"  Output: {OUTPUT_PATH}")
    logger.info(f"  Errors: {len(ckpt.state.get('errors', []))}")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Expand PubMed coverage via MeSH disease descriptor queries.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--max-per-query",
        type=int,
        default=50,
        help="Maximum results per PubMed query (default: 50)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint instead of starting fresh",
    )
    args = parser.parse_args()

    run_crawl(max_per_query=args.max_per_query, resume=args.resume)


if __name__ == "__main__":
    main()
