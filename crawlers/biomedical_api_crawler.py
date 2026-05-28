"""Biomedical API crawler using the same sources as medical_mcp.

Directly queries the public APIs that the medical MCP servers wrap:
- ClinicalTrials.gov v2 API
- OpenFDA API (drug adverse events, labels)
- Open Targets Platform (GraphQL)
- ChEMBL REST API
- UniProt REST API
- PubChem PUG REST API
- KEGG REST API
- NCBI Datasets API

These are the same data sources used in the MCP benchmark evaluation,
ensuring alignment between training data collection and eval tooling.
"""
from __future__ import annotations

import json
import logging
from typing import Iterator

import requests

from .base import BaseCrawler, RawDocument

logger = logging.getLogger(__name__)


class BiomedicalAPICrawler(BaseCrawler):
    """Crawls biomedical APIs matching the medical_mcp tool ecosystem."""

    def __init__(self, output_dir, rate_limit_sec: float = 1.0):
        super().__init__("biomedical_api", output_dir, rate_limit_sec)

    def _crawl_clinicaltrials(self, max_results: int) -> Iterator[RawDocument]:
        """ClinicalTrials.gov v2 API — studies with unknown/challenging conditions."""
        base_url = "https://clinicaltrials.gov/api/v2/studies"

        condition_queries = [
            "incurable disease",
            "treatment-resistant",
            "refractory cancer",
            "rare disease undiagnosed",
            "no effective treatment",
            "Alzheimer's disease-modifying",
            "ALS amyotrophic lateral sclerosis",
            "glioblastoma recurrent",
            "alopecia areata",
            "androgenetic alopecia regeneration",
            "pancreatic cancer metastatic",
            "prion disease",
            "antibiotic resistant infection",
        ]

        total = 0
        for query in condition_queries:
            if self.checkpoint.is_query_done(f"ct:{query}"):
                continue
            self.checkpoint.mark_query_started(f"ct:{query}")

            params = {
                "query.cond": query,
                "pageSize": min(max_results // len(condition_queries), 50),
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
        """OpenFDA — adverse events for drugs with known safety concerns."""
        base_url = "https://api.fda.gov/drug/event.json"

        drug_queries = [
            "chemotherapy",
            "immunotherapy",
            "biologics",
            "gene therapy",
        ]

        total = 0
        for drug in drug_queries:
            if self.checkpoint.is_query_done(f"fda:{drug}"):
                continue
            self.checkpoint.mark_query_started(f"fda:{drug}")

            params = {
                "search": f'patient.drug.medicinalproduct:"{drug}"',
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
                    source_id=f"fda_adverse_{drug}",
                    url=f"https://api.fda.gov/drug/event.json?search={drug}",
                    title=f"Adverse event profile: {drug}",
                    abstract=f"Top adverse reactions for {drug}: {', '.join(top_reactions)}. "
                             f"Total unique reaction types reported: {len(reactions)}.",
                    document_type="adverse_event_summary",
                    keywords=top_reactions,
                    metadata={"drug": drug, "reaction_count": len(reactions)},
                )
                self.checkpoint.add_collected_id(f"fda_adverse_{drug}")
                total += 1

            self.checkpoint.mark_query_done(f"fda:{drug}")
            self.rate_limit()

        logger.info(f"[biomedical_api/openfda] {total} entries")

    def _crawl_opentargets(self, max_results: int) -> Iterator[RawDocument]:
        """Open Targets Platform GraphQL — diseases with limited drug targets."""
        graphql_url = "https://api.platform.opentargets.org/api/v4/graphql"

        disease_queries = [
            ("Alzheimer's disease", "MONDO_0004975"),
            ("Parkinson's disease", "EFO_0002508"),
            ("Amyotrophic lateral sclerosis", "MONDO_0004976"),
            ("Glioblastoma", "EFO_0000519"),
            ("Pancreatic carcinoma", "EFO_0002618"),
            ("Alopecia areata", "MONDO_0004552"),
            ("Huntington disease", "MONDO_0007739"),
            ("Prion disease", "MONDO_0005652"),
        ]

        total = 0
        for disease_name, efo_id in disease_queries:
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
                drugs = data.get("knownDrugs", {})
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

    def _crawl_kegg_diseases(self, max_results: int) -> Iterator[RawDocument]:
        """KEGG REST API — disease entries and associated pathways."""
        diseases = [
            ("H00056", "Alzheimer disease"),
            ("H00057", "Parkinson disease"),
            ("H00058", "Amyotrophic lateral sclerosis"),
            ("H00042", "Glioma"),
            ("H00019", "Pancreatic cancer"),
            ("H01229", "Alopecia areata"),
            ("H00059", "Huntington disease"),
            ("H00061", "Prion disease"),
            ("H01563", "HIV infection"),
            ("H00020", "Colorectal cancer"),
        ]

        total = 0
        for kegg_id, name in diseases:
            if self.checkpoint.is_query_done(f"kegg:{kegg_id}"):
                continue
            self.checkpoint.mark_query_started(f"kegg:{kegg_id}")

            try:
                resp = requests.get(f"https://rest.kegg.jp/get/{kegg_id}", timeout=30)
                resp.raise_for_status()
                text = resp.text
            except Exception as e:
                self.checkpoint.record_error(f"KEGG failed for {name}: {e}")
                continue

            yield RawDocument(
                source="kegg",
                source_id=kegg_id,
                url=f"https://www.kegg.jp/dbget-bin/www_bget?{kegg_id}",
                title=f"KEGG Disease: {name}",
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
        """UniProt REST API — key drug targets with unresolved biology."""
        search_url = "https://rest.uniprot.org/uniprotkb/search"
        target_queries = [
            "(keyword:KW-0798) AND (organism_id:9606) AND (reviewed:true) AND (annotation_score:5)",  # drug targets
        ]

        total = 0
        for query in target_queries:
            if self.checkpoint.is_query_done(f"uniprot:{query[:30]}"):
                continue
            self.checkpoint.mark_query_started(f"uniprot:{query[:30]}")

            params = {
                "query": query,
                "format": "json",
                "size": min(max_results, 50),
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
                    abstract += f"Associated diseases: {'; '.join(diseases[:3])}."

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

            self.checkpoint.mark_query_done(f"uniprot:{query[:30]}")
            self.rate_limit()

        logger.info(f"[biomedical_api/uniprot] {total} entries")

    def _crawl_pubchem_compounds(self, max_results: int) -> Iterator[RawDocument]:
        """PubChem PUG REST — compounds in clinical development for hard-to-treat diseases."""
        base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
        search_queries = [
            "Alzheimer drug candidate",
            "ALS therapeutic compound",
            "hair growth stimulant",
            "glioblastoma chemotherapy agent",
            "antibiotic resistant bacteria",
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
                        description = ""
                        title = f"CID:{cid}"
                except Exception:
                    description = ""
                    title = f"CID:{cid}"

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

    def crawl(self, queries: list[str], max_results: int) -> Iterator[RawDocument]:
        per_source = max(max_results // 6, 20)

        logger.info("=== ClinicalTrials.gov ===")
        yield from self._crawl_clinicaltrials(per_source)

        logger.info("=== OpenFDA ===")
        yield from self._crawl_openfda_adverse_events(per_source)

        logger.info("=== Open Targets ===")
        yield from self._crawl_opentargets(per_source)

        logger.info("=== KEGG ===")
        yield from self._crawl_kegg_diseases(per_source)

        logger.info("=== UniProt ===")
        yield from self._crawl_uniprot_targets(per_source)

        logger.info("=== PubChem ===")
        yield from self._crawl_pubchem_compounds(per_source)
