"""Biomedical API crawler — comprehensive sweep of all medical_mcp data sources.

Queries the same public APIs that the medical MCP servers wrap, but broadly
across ALL therapeutic areas and disease categories, not limited to specific
conditions. Goal: find every domain where humanity still has open questions.

Sources: ClinicalTrials.gov, OpenFDA, Open Targets, ChEMBL, UniProt,
PubChem, KEGG, NCBI Datasets.
"""
from __future__ import annotations

import json
import logging
from typing import Iterator

import requests

from .base import BaseCrawler, RawDocument

logger = logging.getLogger(__name__)

# ── Broad ClinicalTrials.gov queries — every major therapeutic area ──
CT_QUERIES = [
    # By unmet-need signal (disease-agnostic)
    "no effective treatment",
    "treatment-resistant",
    "refractory",
    "incurable",
    "undiagnosed",
    "rare disease",
    "orphan disease",
    "first-in-human",
    "novel mechanism",
    "breakthrough therapy",

    # By therapeutic area (comprehensive sweep)
    "oncology neoplasm",
    "neurodegenerative",
    "neuropsychiatric",
    "cardiovascular",
    "metabolic disorder",
    "autoimmune",
    "infectious disease",
    "respiratory disease",
    "gastrointestinal",
    "renal kidney disease",
    "hepatic liver disease",
    "hematologic",
    "musculoskeletal",
    "dermatologic skin",
    "ophthalmologic eye",
    "endocrine hormone",
    "reproductive fertility",
    "pediatric congenital",
    "geriatric aging",
    "psychiatric mental health",
    "pain chronic",
    "genetic inherited",
    "immunodeficiency",
    "transplant rejection",
    "wound healing fibrosis",
    "sleep disorder",
    "substance abuse addiction",
    "nutrition obesity",
    "environmental exposure toxin",
]

# ── OpenFDA — comprehensive drug class adverse event sweep ──
FDA_DRUG_CLASSES = [
    "antibiotics", "antiviral", "antifungal", "antiparasitic",
    "chemotherapy", "immunotherapy", "targeted therapy",
    "monoclonal antibody", "kinase inhibitor", "checkpoint inhibitor",
    "antidepressant", "antipsychotic", "anxiolytic", "anticonvulsant",
    "antihypertensive", "anticoagulant", "statin", "diuretic",
    "insulin", "metformin", "GLP-1", "SGLT2 inhibitor",
    "corticosteroid", "NSAID", "opioid",
    "biologics", "biosimilar", "gene therapy", "cell therapy",
    "vaccine", "immunoglobulin",
    "antihistamine", "bronchodilator",
    "immunosuppressant", "anti-TNF", "JAK inhibitor",
    "hormone replacement", "contraceptive",
    "antiepileptic", "muscle relaxant",
    "proton pump inhibitor", "laxative",
    "bisphosphonate", "DMARD",
    "retinoid", "phototherapy",
    "ophthalmic", "otic",
    "antithrombotic", "thrombolytic",
    "anesthetic", "sedative",
]

# ── Open Targets — ALL therapeutic areas ──
OT_THERAPEUTIC_AREAS = [
    ("neoplasm", "EFO_0000616"),
    ("nervous system disease", "EFO_0000618"),
    ("cardiovascular disease", "EFO_0000319"),
    ("immune system disease", "EFO_0000540"),
    ("respiratory or thoracic disease", "OTAR_0000010"),
    ("infectious disease", "EFO_0005741"),
    ("metabolic disease", "EFO_0000589"),
    ("endocrine system disease", "EFO_0001379"),
    ("gastrointestinal disease", "EFO_0010282"),
    ("musculoskeletal or connective tissue disease", "OTAR_0000006"),
    ("genetic, familial or congenital disease", "OTAR_0000018"),
    ("urinary system disease", "EFO_0009690"),
    ("reproductive system or breast disease", "OTAR_0000014"),
    ("skin disease", "EFO_0010285"),
    ("eye disease", "MONDO_0024458"),
    ("hematologic disease", "EFO_0005803"),
    ("psychiatric disorder", "EFO_0000677"),
    ("liver disease", "EFO_0001421"),
    ("pregnancy or perinatal disease", "OTAR_0000017"),
    ("ear disease", "EFO_0010283"),
    ("nutritional or metabolic disease", "OTAR_0000003"),
]

# ── KEGG — enumerate ALL disease categories ──
KEGG_DISEASE_CATEGORIES = [
    "hsa05010",  # Alzheimer disease pathway
    "hsa05012",  # Parkinson disease pathway
    "hsa05014",  # ALS pathway
    "hsa05016",  # Huntington disease pathway
    "hsa05020",  # Prion disease pathway
    "hsa05200",  # Pathways in cancer
    "hsa05210",  # Colorectal cancer
    "hsa05212",  # Pancreatic cancer
    "hsa05220",  # Chronic myeloid leukemia
    "hsa05221",  # Acute myeloid leukemia
    "hsa05223",  # Non-small cell lung cancer
    "hsa05230",  # Central carbon metabolism in cancer
    "hsa04930",  # Type II diabetes mellitus
    "hsa04940",  # Type I diabetes mellitus
    "hsa05322",  # Systemic lupus erythematosus
    "hsa05323",  # Rheumatoid arthritis
    "hsa05310",  # Asthma
    "hsa05321",  # Inflammatory bowel disease
    "hsa05330",  # Allograft rejection
    "hsa05332",  # Graft-versus-host disease
    "hsa05340",  # Primary immunodeficiency
    "hsa05410",  # Hypertrophic cardiomyopathy
    "hsa05414",  # Dilated cardiomyopathy
    "hsa05418",  # Fluid shear stress and atherosclerosis
    "hsa04932",  # Non-alcoholic fatty liver disease
    "hsa05160",  # Hepatitis C
    "hsa05161",  # Hepatitis B
    "hsa05170",  # HIV infection
    "hsa05171",  # Coronavirus disease
    "hsa05130",  # Pathogenic E. coli infection
    "hsa05150",  # Staphylococcus aureus infection
    "hsa05110",  # Vibrio cholerae infection
    "hsa05100",  # Bacterial invasion of epithelial cells
    "hsa04933",  # AGE-RAGE signaling in diabetic complications
    "hsa04934",  # Cushing syndrome
    "hsa05030",  # Cocaine addiction
    "hsa05031",  # Amphetamine addiction
    "hsa05032",  # Morphine addiction
    "hsa05033",  # Nicotine addiction
    "hsa05034",  # Alcoholism
    "hsa04723",  # Retrograde endocannabinoid signaling
]


class BiomedicalAPICrawler(BaseCrawler):
    """Comprehensively crawls biomedical APIs across ALL therapeutic areas."""

    def __init__(self, output_dir, rate_limit_sec: float = 1.0):
        super().__init__("biomedical_api", output_dir, rate_limit_sec)

    def _crawl_clinicaltrials(self, max_results: int) -> Iterator[RawDocument]:
        """ClinicalTrials.gov v2 API — broad sweep across all therapeutic areas."""
        base_url = "https://clinicaltrials.gov/api/v2/studies"

        per_query = max(max_results // len(CT_QUERIES), 10)
        total = 0

        for query in CT_QUERIES:
            if self.checkpoint.is_query_done(f"ct:{query}"):
                continue
            self.checkpoint.mark_query_started(f"ct:{query}")

            params = {
                "query.cond": query,
                "pageSize": min(per_query, 50),
                "format": "json",
                "fields": "NCTId,BriefTitle,Condition,BriefSummary,Phase,OverallStatus,StartDate,PrimaryCompletionDate",
            }
            try:
                resp = requests.get(base_url, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                self.checkpoint.record_error(f"ClinicalTrials query failed: {e}")
                continue

            studies = data.get("studies", [])
            for study in studies:
                proto = study.get("protocolSection", {})
                ident = proto.get("identificationModule", {})
                desc = proto.get("descriptionModule", {})
                status = proto.get("statusModule", {})
                cond = proto.get("conditionsModule", {})

                nct_id = ident.get("nctId", "")
                if self.checkpoint.is_collected(nct_id):
                    continue

                conditions = cond.get("conditions", [])
                yield RawDocument(
                    source="clinicaltrials",
                    source_id=nct_id,
                    url=f"https://clinicaltrials.gov/study/{nct_id}",
                    title=ident.get("briefTitle", ""),
                    abstract=desc.get("briefSummary", ""),
                    document_type="clinical_trial",
                    keywords=conditions,
                    metadata={
                        "nct_id": nct_id,
                        "phase": proto.get("designModule", {}).get("phases", []),
                        "status": status.get("overallStatus", ""),
                        "conditions": conditions,
                    },
                )
                self.checkpoint.add_collected_id(nct_id)
                total += 1

            self.checkpoint.mark_query_done(f"ct:{query}")
            self.rate_limit()

        logger.info(f"[biomedical_api/clinicaltrials] {total} studies")

    def _crawl_openfda_adverse_events(self, max_results: int) -> Iterator[RawDocument]:
        """OpenFDA — adverse event profiles for ALL major drug classes."""
        base_url = "https://api.fda.gov/drug/event.json"
        total = 0

        for drug_class in FDA_DRUG_CLASSES:
            if self.checkpoint.is_query_done(f"fda:{drug_class}"):
                continue
            self.checkpoint.mark_query_started(f"fda:{drug_class}")

            params = {
                "search": f'patient.drug.medicinalproduct:"{drug_class}"',
                "count": "patient.reaction.reactionmeddrapt.exact",
                "limit": 20,
            }
            try:
                resp = requests.get(base_url, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                self.checkpoint.record_error(f"OpenFDA query failed: {e}")
                continue

            reactions = data.get("results", [])
            if reactions:
                top_reactions = [r["term"] for r in reactions[:10]]
                yield RawDocument(
                    source="openfda",
                    source_id=f"fda_adverse_{drug_class}",
                    url=f"https://api.fda.gov/drug/event.json?search={drug_class}",
                    title=f"Adverse event profile: {drug_class}",
                    abstract=f"Top adverse reactions for {drug_class}: {', '.join(top_reactions)}. "
                             f"Total unique reaction types: {len(reactions)}.",
                    document_type="adverse_event_summary",
                    keywords=top_reactions,
                    metadata={"drug_class": drug_class, "reaction_count": len(reactions)},
                )
                self.checkpoint.add_collected_id(f"fda_adverse_{drug_class}")
                total += 1

            self.checkpoint.mark_query_done(f"fda:{drug_class}")
            self.rate_limit()

        logger.info(f"[biomedical_api/openfda] {total} entries")

    def _crawl_opentargets(self, max_results: int) -> Iterator[RawDocument]:
        """Open Targets Platform GraphQL — ALL therapeutic areas."""
        graphql_url = "https://api.platform.opentargets.org/api/v4/graphql"
        total = 0

        for disease_name, efo_id in OT_THERAPEUTIC_AREAS:
            if self.checkpoint.is_query_done(f"ot:{efo_id}"):
                continue
            self.checkpoint.mark_query_started(f"ot:{efo_id}")

            query = """
            query($efoId: String!) {
              disease(efoId: $efoId) {
                id
                name
                description
                therapeuticAreas { id name }
                knownDrugs { uniqueTargets uniqueDrugs count }
              }
            }
            """
            try:
                resp = requests.post(
                    graphql_url,
                    json={"query": query, "variables": {"efoId": efo_id}},
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json().get("data", {}).get("disease", {})
            except Exception as e:
                self.checkpoint.record_error(f"OpenTargets query failed for {disease_name}: {e}")
                continue

            if data:
                drugs = data.get("knownDrugs", {}) or {}
                areas = [a.get("name", "") for a in data.get("therapeuticAreas", [])]
                yield RawDocument(
                    source="opentargets",
                    source_id=efo_id,
                    url=f"https://platform.opentargets.org/disease/{efo_id}",
                    title=f"Open Targets: {data.get('name', disease_name)}",
                    abstract=f"{data.get('description', '')} "
                             f"Known drugs: {drugs.get('count', 0)}, "
                             f"unique targets: {drugs.get('uniqueTargets', 0)}, "
                             f"therapeutic areas: {', '.join(areas)}.",
                    document_type="drug_target_association",
                    keywords=areas,
                    metadata={
                        "efo_id": efo_id,
                        "known_drugs_count": drugs.get("count", 0),
                        "unique_targets": drugs.get("uniqueTargets", 0),
                    },
                )
                self.checkpoint.add_collected_id(efo_id)
                total += 1

            self.checkpoint.mark_query_done(f"ot:{efo_id}")
            self.rate_limit()

        logger.info(f"[biomedical_api/opentargets] {total} entries")

    def _crawl_kegg_pathways(self, max_results: int) -> Iterator[RawDocument]:
        """KEGG REST API — ALL disease-related pathways."""
        total = 0

        for kegg_id in KEGG_DISEASE_CATEGORIES:
            if self.checkpoint.is_query_done(f"kegg:{kegg_id}"):
                continue
            self.checkpoint.mark_query_started(f"kegg:{kegg_id}")

            try:
                resp = requests.get(f"https://rest.kegg.jp/get/{kegg_id}", timeout=30)
                resp.raise_for_status()
                text = resp.text
            except Exception as e:
                self.checkpoint.record_error(f"KEGG failed for {kegg_id}: {e}")
                continue

            name_line = ""
            for line in text.split("\n"):
                if line.startswith("NAME"):
                    name_line = line.replace("NAME", "").strip()
                    break

            yield RawDocument(
                source="kegg",
                source_id=kegg_id,
                url=f"https://www.kegg.jp/pathway/{kegg_id}",
                title=f"KEGG Pathway: {name_line or kegg_id}",
                abstract=text[:3000],
                full_text=text,
                document_type="disease_pathway",
                metadata={"kegg_id": kegg_id},
            )
            self.checkpoint.add_collected_id(kegg_id)
            self.checkpoint.mark_query_done(f"kegg:{kegg_id}")
            total += 1
            self.rate_limit()

        logger.info(f"[biomedical_api/kegg] {total} entries")

    def _crawl_uniprot_targets(self, max_results: int) -> Iterator[RawDocument]:
        """UniProt REST API — human drug targets, disease-associated proteins, enzymes."""
        search_url = "https://rest.uniprot.org/uniprotkb/search"
        target_queries = [
            # Drug targets
            "(keyword:KW-0798) AND (organism_id:9606) AND (reviewed:true) AND (annotation_score:5)",
            # Disease-associated
            "(keyword:KW-0225) AND (organism_id:9606) AND (reviewed:true)",
            # Proto-oncogene
            "(keyword:KW-0656) AND (organism_id:9606) AND (reviewed:true)",
            # Tumor suppressor
            "(keyword:KW-0043) AND (organism_id:9606) AND (reviewed:true)",
            # Receptor
            "(keyword:KW-0675) AND (organism_id:9606) AND (reviewed:true) AND (annotation_score:5)",
        ]

        total = 0
        per_query = max(max_results // len(target_queries), 20)

        for query in target_queries:
            query_key = f"uniprot:{query[:40]}"
            if self.checkpoint.is_query_done(query_key):
                continue
            self.checkpoint.mark_query_started(query_key)

            params = {
                "query": query,
                "format": "json",
                "size": min(per_query, 100),
                "fields": "accession,id,protein_name,gene_names,organism_name,cc_function,cc_disease",
            }
            try:
                resp = requests.get(search_url, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                self.checkpoint.record_error(f"UniProt query failed: {e}")
                continue

            for entry in data.get("results", []):
                acc = entry.get("primaryAccession", "")
                if self.checkpoint.is_collected(acc):
                    continue

                protein_name = ""
                pn = entry.get("proteinDescription", {}).get("recommendedName", {})
                if pn:
                    protein_name = pn.get("fullName", {}).get("value", "")

                gene_names = []
                for g in entry.get("genes", []):
                    gn = g.get("geneName", {}).get("value", "")
                    if gn:
                        gene_names.append(gn)

                functions = []
                diseases = []
                for comment in entry.get("comments", []):
                    ctype = comment.get("commentType", "")
                    if ctype == "FUNCTION":
                        for t in comment.get("texts", []):
                            functions.append(t.get("value", ""))
                    elif ctype == "DISEASE":
                        d = comment.get("disease", {})
                        if d:
                            diseases.append(d.get("diseaseId", "") + ": " + d.get("description", ""))

                abstract = f"Protein: {protein_name}. Genes: {', '.join(gene_names)}. "
                if functions:
                    abstract += f"Function: {functions[0][:500]}. "
                if diseases:
                    abstract += f"Associated diseases: {'; '.join(diseases[:5])}."

                yield RawDocument(
                    source="uniprot",
                    source_id=acc,
                    url=f"https://www.uniprot.org/uniprot/{acc}",
                    title=f"UniProt: {protein_name} ({', '.join(gene_names)})",
                    abstract=abstract,
                    document_type="protein_target",
                    keywords=gene_names + [protein_name],
                    metadata={"accession": acc, "diseases": diseases},
                )
                self.checkpoint.add_collected_id(acc)
                total += 1

            self.checkpoint.mark_query_done(query_key)
            self.rate_limit()

        logger.info(f"[biomedical_api/uniprot] {total} entries")

    def _crawl_pubchem_compounds(self, max_results: int) -> Iterator[RawDocument]:
        """PubChem PUG REST — compounds across all therapeutic areas."""
        base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
        search_queries = [
            # Broad therapeutic categories
            "anticancer agent",
            "antiviral agent",
            "antibiotic agent",
            "antifungal agent",
            "anti-inflammatory agent",
            "immunosuppressant",
            "antidepressant",
            "antipsychotic",
            "analgesic",
            "antidiabetic agent",
            "cardiovascular agent",
            "antihypertensive agent",
            "neuroprotective agent",
            "bronchodilator",
            "anticoagulant",
            "anticonvulsant",
            "antimalarial",
            "anthelmintic",
            "gene therapy vector",
        ]

        total = 0
        for query in search_queries:
            if self.checkpoint.is_query_done(f"pubchem:{query}"):
                continue
            self.checkpoint.mark_query_started(f"pubchem:{query}")

            try:
                search_resp = requests.get(
                    f"{base_url}/compound/name/{query}/cids/JSON",
                    timeout=30,
                )
                if search_resp.status_code != 200:
                    self.checkpoint.mark_query_done(f"pubchem:{query}")
                    continue
                cids = search_resp.json().get("IdentifierList", {}).get("CID", [])[:5]
            except Exception as e:
                self.checkpoint.record_error(f"PubChem search failed: {e}")
                continue

            for cid in cids:
                if self.checkpoint.is_collected(str(cid)):
                    continue
                try:
                    desc_resp = requests.get(
                        f"{base_url}/compound/cid/{cid}/description/JSON",
                        timeout=15,
                    )
                    if desc_resp.status_code == 200:
                        descs = desc_resp.json().get("InformationList", {}).get("Information", [])
                        description = descs[0].get("Description", "") if descs else ""
                        title = descs[0].get("Title", f"CID:{cid}") if descs else f"CID:{cid}"
                    else:
                        description, title = "", f"CID:{cid}"
                except Exception:
                    description, title = "", f"CID:{cid}"

                yield RawDocument(
                    source="pubchem",
                    source_id=str(cid),
                    url=f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
                    title=f"PubChem: {title}",
                    abstract=description[:2000],
                    document_type="compound",
                    metadata={"cid": cid, "search_query": query},
                )
                self.checkpoint.add_collected_id(str(cid))
                total += 1
                self.rate_limit()

            self.checkpoint.mark_query_done(f"pubchem:{query}")

        logger.info(f"[biomedical_api/pubchem] {total} entries")

    def _crawl_ncbi_datasets(self, max_results: int) -> Iterator[RawDocument]:
        """NCBI Datasets API — genes associated with disease phenotypes."""
        base_url = "https://api.ncbi.nlm.nih.gov/datasets/v2"
        gene_queries = [
            "cancer susceptibility",
            "neurodegeneration",
            "cardiovascular risk",
            "immune deficiency",
            "metabolic disorder",
            "rare genetic disease",
            "pharmacogenomics",
            "epigenetic regulator",
        ]

        total = 0
        for query in gene_queries:
            if self.checkpoint.is_query_done(f"ncbi:{query}"):
                continue
            self.checkpoint.mark_query_started(f"ncbi:{query}")

            try:
                resp = requests.get(
                    f"{base_url}/gene/taxon/9606",
                    params={"search_text": query, "page_size": 10},
                    headers={"Accept": "application/json"},
                    timeout=30,
                )
                if resp.status_code != 200:
                    self.checkpoint.mark_query_done(f"ncbi:{query}")
                    continue
                data = resp.json()
            except Exception as e:
                self.checkpoint.record_error(f"NCBI query failed: {e}")
                continue

            for gene in data.get("genes", []):
                gene_info = gene.get("gene", {})
                gene_id = str(gene_info.get("gene_id", ""))
                if not gene_id or self.checkpoint.is_collected(gene_id):
                    continue

                symbol = gene_info.get("symbol", "")
                description = gene_info.get("description", "")
                full_name = gene_info.get("full_name", "")

                yield RawDocument(
                    source="ncbi_datasets",
                    source_id=gene_id,
                    url=f"https://www.ncbi.nlm.nih.gov/gene/{gene_id}",
                    title=f"NCBI Gene: {symbol} — {full_name}",
                    abstract=f"Gene {symbol} ({full_name}): {description}",
                    document_type="gene_record",
                    keywords=[symbol, full_name],
                    metadata={"gene_id": gene_id, "symbol": symbol, "query": query},
                )
                self.checkpoint.add_collected_id(gene_id)
                total += 1

            self.checkpoint.mark_query_done(f"ncbi:{query}")
            self.rate_limit()

        logger.info(f"[biomedical_api/ncbi] {total} entries")

    def crawl(self, queries: list[str], max_results: int) -> Iterator[RawDocument]:
        per_source = max(max_results // 7, 20)

        logger.info("=== ClinicalTrials.gov ===")
        yield from self._crawl_clinicaltrials(per_source)

        logger.info("=== OpenFDA ===")
        yield from self._crawl_openfda_adverse_events(per_source)

        logger.info("=== Open Targets ===")
        yield from self._crawl_opentargets(per_source)

        logger.info("=== KEGG ===")
        yield from self._crawl_kegg_pathways(per_source)

        logger.info("=== UniProt ===")
        yield from self._crawl_uniprot_targets(per_source)

        logger.info("=== PubChem ===")
        yield from self._crawl_pubchem_compounds(per_source)

        logger.info("=== NCBI Datasets ===")
        yield from self._crawl_ncbi_datasets(per_source)
