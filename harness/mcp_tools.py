"""Thin wrappers for 10 medical REST APIs with OpenAI function-calling schemas."""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_TIMEOUT = 30  # seconds per request
_MAX_RESULT_CHARS = 12000  # truncate response body to this length (fits ~6-8 pubmed abstracts)
_RATE_LIMIT_SEC = 0.34  # ~3 req/s = NCBI keyless ceiling (was 1.0; 3x faster for scale runs)

# Module-level rate-limiter state
_last_call_ts: float = 0.0


def _rate_limit() -> None:
    """Block until at least _RATE_LIMIT_SEC has elapsed since the last call."""
    global _last_call_ts
    now = time.monotonic()
    wait = _RATE_LIMIT_SEC - (now - _last_call_ts)
    if wait > 0:
        time.sleep(wait)
    _last_call_ts = time.monotonic()


def _truncate(text: str, max_chars: int = _MAX_RESULT_CHARS) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars] + "\n... [truncated]", True


def _safe_json(resp: requests.Response) -> Any:
    try:
        return resp.json()
    except Exception:
        body, _ = _truncate(resp.text)
        return {"raw_text": body}


# ===================================================================
# Base class
# ===================================================================


class MCPTool(ABC):
    """Base class for all MCP tool wrappers."""

    name: str
    description: str
    parameters_schema: dict  # JSON Schema for function parameters

    @abstractmethod
    def _execute(self, **kwargs: Any) -> Any:
        """Run the actual API call. Must return serialisable data."""

    def execute(self, **kwargs: Any) -> dict:
        """Public entry: rate-limit, call _execute, wrap result."""
        # Some models (e.g. DeepSeek-V4) emit numeric args as strings ("10"); coerce
        # so downstream slicing/limits don't raise "slice indices must be integers".
        for k in ("max_results", "retmax", "page_size", "pageSize", "limit", "top_k"):
            v = kwargs.get(k)
            if isinstance(v, str) and v.strip().lstrip("-").isdigit():
                kwargs[k] = int(v)
        _rate_limit()
        try:
            results = self._execute(**kwargs)
        except requests.Timeout:
            logger.warning("%s: request timed out", self.name)
            results = {"error": "Request timed out after 30s"}
        except requests.RequestException as exc:
            logger.warning("%s: request failed: %s", self.name, exc)
            results = {"error": str(exc)}

        # Truncate if the serialised result is too long
        import json as _json

        raw = _json.dumps(results, ensure_ascii=False, default=str)
        truncated_str, was_truncated = _truncate(raw)
        if was_truncated:
            # The truncated JSON usually can't be repaired by appending "}" (truncation
            # can land mid-string/array → "Unterminated string"). Try, but fall back to
            # the truncated text so a long tool result never crashes the task.
            body = truncated_str.split("\n... [truncated]")[0]
            try:
                results = _json.loads(body + "}")
            except _json.JSONDecodeError:
                results = {"truncated_text": body, "note": "result truncated; partial text only"}

        return {
            "tool": self.name,
            "query": kwargs,
            "results": results,
            "truncated": was_truncated,
        }

    def openai_schema(self) -> dict:
        """Return an OpenAI function-calling compatible schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }


# ===================================================================
# 1. PubMed (NCBI E-Utilities)
# ===================================================================


class PubMedTool(MCPTool):
    name = "pubmed"
    description = (
        "Search PubMed/MEDLINE biomedical literature (via Europe PMC). "
        "Returns titles, abstracts, and PMIDs."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "PubMed search query"},
            "max_results": {
                "type": "integer",
                "description": "Maximum results to return",
                "default": 5,
            },
        },
        "required": ["query"],
    }

    def _execute(self, *, query: str, max_results: int = 5) -> Any:
        # Europe PMC search: single call returns title + abstract; far more tolerant
        # of concurrency than NCBI E-utils (which 429s under parallel load without a key).
        resp = requests.get(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            params={"query": query, "format": "json", "resultType": "core",
                    "pageSize": max_results},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        results = resp.json().get("resultList", {}).get("result", [])
        articles = []
        for r in results[:max_results]:
            articles.append({
                "pmid": r.get("pmid") or r.get("id", ""),
                "title": r.get("title", ""),
                "abstract": (r.get("abstractText") or "")[:1000],
            })
        return articles


# ===================================================================
# 2. ClinicalTrials.gov (API v2)
# ===================================================================


class ClinicalTrialsGovTool(MCPTool):
    name = "clinicaltrialsgov"
    description = (
        "Search ClinicalTrials.gov for clinical trials using the v2 API. "
        "Returns trial NCT IDs, titles, status, conditions, and interventions."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query for clinical trials"},
            "max_results": {
                "type": "integer",
                "description": "Maximum results to return",
                "default": 5,
            },
        },
        "required": ["query"],
    }

    def _execute(self, *, query: str, max_results: int = 5) -> Any:
        url = "https://clinicaltrials.gov/api/v2/studies"
        params = {
            "query.term": query,
            "pageSize": min(max_results, 20),
            "format": "json",
        }
        resp = requests.get(url, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        studies = data.get("studies", [])
        results = []
        for s in studies[:max_results]:
            proto = s.get("protocolSection", {})
            ident = proto.get("identificationModule", {})
            status = proto.get("statusModule", {})
            conditions = proto.get("conditionsModule", {})
            interventions = proto.get("armsInterventionsModule", {})
            results.append(
                {
                    "nct_id": ident.get("nctId", ""),
                    "title": ident.get("briefTitle", ""),
                    "status": status.get("overallStatus", ""),
                    "conditions": conditions.get("conditions", []),
                    "interventions": [
                        i.get("name", "")
                        for i in interventions.get("interventions", [])
                    ],
                }
            )
        return results


# ===================================================================
# 3. OpenFDA
# ===================================================================


class OpenFDATool(MCPTool):
    name = "openfda"
    description = (
        "Query the openFDA API for drug adverse events or drug labeling information."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "endpoint": {
                "type": "string",
                "enum": ["drug/event", "drug/label"],
                "description": "OpenFDA endpoint to query",
            },
            "query": {"type": "string", "description": "Search query string"},
            "limit": {
                "type": "integer",
                "description": "Maximum results",
                "default": 5,
            },
        },
        "required": ["endpoint", "query"],
    }

    def _execute(self, *, endpoint: str = "drug/event", query: str, limit: int = 5) -> Any:
        url = f"https://api.fda.gov/{endpoint}.json"
        params = {"search": query, "limit": limit}
        resp = requests.get(url, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        return data.get("results", [])


# ===================================================================
# 4. Open Targets (GraphQL)
# ===================================================================


class OpenTargetsTool(MCPTool):
    name = "opentargets"
    description = (
        "Query Open Targets Platform GraphQL API for target, disease, or drug associations."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "query_type": {
                "type": "string",
                "enum": ["target", "disease", "drug"],
                "description": "Type of entity to query",
            },
            "id": {
                "type": "string",
                "description": "Entity ID (e.g. ENSG ID for target, EFO ID for disease, CHEMBL ID for drug)",
            },
        },
        "required": ["query_type", "id"],
    }

    _QUERIES = {
        "target": """
            query($id: String!) {
                target(ensemblId: $id) {
                    id
                    approvedSymbol
                    approvedName
                    biotype
                    functionDescriptions
                }
            }
        """,
        "disease": """
            query($id: String!) {
                disease(efoId: $id) {
                    id
                    name
                    description
                    therapeuticAreas { id name }
                }
            }
        """,
        "drug": """
            query($id: String!) {
                drug(chemblId: $id) {
                    id
                    name
                    drugType
                    mechanismsOfAction { rows { mechanismOfAction } }
                    indications { rows { disease { name } } }
                }
            }
        """,
    }

    def _execute(self, *, query_type: str, id: str) -> Any:
        url = "https://api.platform.opentargets.org/api/v4/graphql"
        gql = self._QUERIES.get(query_type)
        if gql is None:
            return {"error": f"Unknown query_type: {query_type}"}
        payload = {"query": gql, "variables": {"id": id}}
        resp = requests.post(url, json=payload, timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp.json().get("data", {})


# ===================================================================
# 5. ChEMBL
# ===================================================================


class ChEMBLTool(MCPTool):
    name = "chembl"
    description = (
        "Search ChEMBL REST API for molecules, targets, or bioactivity data."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "endpoint": {
                "type": "string",
                "enum": ["molecule", "target", "activity"],
                "description": "ChEMBL resource to query",
            },
            "query": {"type": "string", "description": "Search query or ChEMBL ID"},
        },
        "required": ["endpoint", "query"],
    }

    def _execute(self, *, endpoint: str, query: str) -> Any:
        base = "https://www.ebi.ac.uk/chembl/api/data"
        # If query looks like a ChEMBL ID, fetch directly
        if query.upper().startswith("CHEMBL"):
            url = f"{base}/{endpoint}/{query}.json"
            resp = requests.get(url, timeout=_TIMEOUT)
            resp.raise_for_status()
            return _safe_json(resp)
        else:
            url = f"{base}/{endpoint}/search.json"
            params = {"q": query, "limit": 5}
            resp = requests.get(url, params=params, timeout=_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            return data.get(f"{endpoint}s", data)


# ===================================================================
# 6. UniProt
# ===================================================================


class UniProtTool(MCPTool):
    name = "uniprot"
    description = "Search UniProt for protein information including function, sequence, and annotations."
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "UniProt search query (gene name, protein name, etc.)"},
            "max_results": {
                "type": "integer",
                "description": "Maximum results",
                "default": 5,
            },
        },
        "required": ["query"],
    }

    def _execute(self, *, query: str, max_results: int = 5) -> Any:
        url = "https://rest.uniprot.org/uniprotkb/search"
        params = {
            "query": query,
            "format": "json",
            "size": max_results,
            "fields": "accession,id,protein_name,gene_names,organism_name,length,cc_function",
        }
        resp = requests.get(url, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        results = []
        for entry in data.get("results", []):
            results.append(
                {
                    "accession": entry.get("primaryAccession", ""),
                    "id": entry.get("uniProtkbId", ""),
                    "protein_name": (
                        entry.get("proteinDescription", {})
                        .get("recommendedName", {})
                        .get("fullName", {})
                        .get("value", "")
                    ),
                    "gene_names": [
                        g.get("geneName", {}).get("value", "")
                        for g in entry.get("genes", [])
                    ],
                    "organism": entry.get("organism", {}).get("scientificName", ""),
                    "length": entry.get("sequence", {}).get("length", 0),
                }
            )
        return results


# ===================================================================
# 7. PubChem
# ===================================================================


class PubChemTool(MCPTool):
    name = "pubchem"
    description = "Search PubChem PUG REST for compound information by name, formula, or SMILES."
    parameters_schema = {
        "type": "object",
        "properties": {
            "search_type": {
                "type": "string",
                "enum": ["name", "formula", "smiles"],
                "description": "Type of search input",
            },
            "query": {"type": "string", "description": "Compound name, formula, or SMILES string"},
        },
        "required": ["search_type", "query"],
    }

    def _execute(self, *, search_type: str, query: str) -> Any:
        base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
        namespace_map = {
            "name": "name",
            "formula": "fastformula",
            "smiles": "smiles",
        }
        namespace = namespace_map.get(search_type, "name")

        # Get CIDs first
        url = f"{base}/compound/{namespace}/{requests.utils.quote(query)}/cids/JSON"
        resp = requests.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        cids = resp.json().get("IdentifierList", {}).get("CID", [])[:5]
        if not cids:
            return []

        # Fetch properties for those CIDs
        cid_str = ",".join(str(c) for c in cids)
        prop_url = (
            f"{base}/compound/cid/{cid_str}/property/"
            "MolecularFormula,MolecularWeight,IUPACName,CanonicalSMILES/JSON"
        )
        resp = requests.get(prop_url, timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp.json().get("PropertyTable", {}).get("Properties", [])


# ===================================================================
# 8. KEGG
# ===================================================================


class KEGGTool(MCPTool):
    name = "kegg"
    description = "Query KEGG REST API for pathway, gene, compound, and disease information."
    parameters_schema = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["find", "get", "link"],
                "description": "KEGG API operation",
            },
            "database": {
                "type": "string",
                "description": "KEGG database (e.g. pathway, genes, compound, disease)",
            },
            "query": {"type": "string", "description": "Search term or KEGG ID"},
        },
        "required": ["operation", "database", "query"],
    }

    def _execute(self, *, operation: str, database: str, query: str) -> Any:
        base = "https://rest.kegg.jp"
        if operation == "find":
            url = f"{base}/find/{database}/{requests.utils.quote(query)}"
        elif operation == "get":
            url = f"{base}/get/{requests.utils.quote(query)}"
        elif operation == "link":
            url = f"{base}/link/{database}/{requests.utils.quote(query)}"
        else:
            return {"error": f"Unknown operation: {operation}"}

        resp = requests.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        text, _ = _truncate(resp.text)
        # Parse tab-separated results
        lines = [line for line in text.strip().split("\n") if line.strip()]
        if operation == "get":
            return {"entry": text}
        results = []
        for line in lines:
            parts = line.split("\t")
            if len(parts) >= 2:
                results.append({"id": parts[0], "description": parts[1]})
            else:
                results.append({"raw": line})
        return results


# ===================================================================
# 9. NCBI Datasets
# ===================================================================


class NCBIDatasetsTool(MCPTool):
    name = "ncbi_datasets"
    description = "Query NCBI Datasets API for gene or genome information."
    parameters_schema = {
        "type": "object",
        "properties": {
            "query_type": {
                "type": "string",
                "enum": ["gene", "genome"],
                "description": "Type of dataset to query",
            },
            "query": {"type": "string", "description": "Gene symbol/name or organism name"},
        },
        "required": ["query_type", "query"],
    }

    def _execute(self, *, query_type: str, query: str) -> Any:
        base = "https://api.ncbi.nlm.nih.gov/datasets/v2"
        if query_type == "gene":
            url = f"{base}/gene/symbol/{requests.utils.quote(query)}/taxon/human"
            resp = requests.get(
                url,
                headers={"Accept": "application/json"},
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            genes = data.get("genes", [])
            results = []
            for g in genes[:5]:
                gene_info = g.get("gene", {})
                results.append(
                    {
                        "gene_id": gene_info.get("gene_id", ""),
                        "symbol": gene_info.get("symbol", ""),
                        "description": gene_info.get("description", ""),
                        "type": gene_info.get("type", ""),
                        "chromosomes": gene_info.get("chromosomes", []),
                    }
                )
            return results
        elif query_type == "genome":
            url = f"{base}/genome/taxon/{requests.utils.quote(query)}"
            params = {"page_size": 5}
            resp = requests.get(
                url,
                params=params,
                headers={"Accept": "application/json"},
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            assemblies = data.get("reports", [])
            results = []
            for a in assemblies[:5]:
                results.append(
                    {
                        "accession": a.get("accession", ""),
                        "organism": a.get("organism", {}).get("organism_name", ""),
                        "assembly_level": a.get("assembly_info", {}).get("assembly_level", ""),
                        "assembly_name": a.get("assembly_info", {}).get("assembly_name", ""),
                    }
                )
            return results
        else:
            return {"error": f"Unknown query_type: {query_type}"}


# ===================================================================
# 10. BioMCP (combined PubMed + ClinicalTrials)
# ===================================================================


class BioMCPTool(MCPTool):
    name = "biomcp"
    description = (
        "Combined biomedical query that searches both PubMed and ClinicalTrials.gov "
        "simultaneously and returns merged results."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Biomedical search query"},
        },
        "required": ["query"],
    }

    def __init__(self) -> None:
        self._pubmed = PubMedTool()
        self._ct = ClinicalTrialsGovTool()

    def _execute(self, *, query: str) -> Any:
        pubmed_results = self._pubmed._execute(query=query, max_results=3)
        # Small pause between sub-calls
        time.sleep(0.5)
        ct_results = self._ct._execute(query=query, max_results=3)
        return {
            "pubmed": pubmed_results,
            "clinical_trials": ct_results,
        }


# ===================================================================
# Registry
# ===================================================================

# Canonical tool instances
_ALL_TOOLS: dict[str, MCPTool] = {}


def _ensure_registry() -> dict[str, MCPTool]:
    """Lazily initialise tool registry."""
    if not _ALL_TOOLS:
        for cls in [
            PubMedTool,
            ClinicalTrialsGovTool,
            OpenFDATool,
            OpenTargetsTool,
            ChEMBLTool,
            UniProtTool,
            PubChemTool,
            KEGGTool,
            NCBIDatasetsTool,
            BioMCPTool,
        ]:
            inst = cls()
            _ALL_TOOLS[inst.name] = inst
    return _ALL_TOOLS


class MCPToolRegistry:
    """Central registry for discovering and executing MCP tools."""

    def __init__(self) -> None:
        self._tools = _ensure_registry()

    @property
    def available_tools(self) -> list[str]:
        return list(self._tools.keys())

    def get_tool_schemas(self, tool_names: list[str] | None = None) -> list[dict]:
        """Return OpenAI function-calling schemas for the requested tools.

        Args:
            tool_names: List of tool names. If None, returns schemas for all tools.

        Returns:
            List of OpenAI function schema dicts.
        """
        if tool_names is None:
            tool_names = self.available_tools

        schemas = []
        for name in tool_names:
            # Normalise hyphens
            norm_name = name.replace("-", "_")
            tool = self._tools.get(norm_name)
            if tool is None:
                logger.warning("Unknown tool requested: %s", name)
                continue
            schemas.append(tool.openai_schema())
        return schemas

    def execute(self, tool_name: str, arguments: dict) -> dict:
        """Execute a tool by name with the given arguments.

        Args:
            tool_name: Name of the tool (hyphens or underscores accepted).
            arguments: Keyword arguments for the tool.

        Returns:
            Standard result dict with keys: tool, query, results, truncated.
        """
        norm_name = tool_name.replace("-", "_")
        # DeepSeek-V4 wraps args as {"arguments": <inner>} where inner may be a dict,
        # a JSON string, or RECURSIVELY wrapped again on later rounds
        # (e.g. {"arguments": "{\"arguments\": \"{...}\"}"}). Unwrap+decode in a loop.
        import json as _j
        for _ in range(6):
            if isinstance(arguments, str):
                try:
                    arguments = _j.loads(arguments)
                except Exception:
                    arguments = {}
                    break
            if isinstance(arguments, dict) and set(arguments) <= {"arguments", "name"} \
                    and "arguments" in arguments:
                arguments = arguments["arguments"]
                continue
            break
        if not isinstance(arguments, dict):
            arguments = {}
        tool = self._tools.get(norm_name)
        if tool is None:
            return {
                "tool": tool_name,
                "query": arguments,
                "results": {"error": f"Unknown tool: {tool_name}"},
                "truncated": False,
            }
        return tool.execute(**arguments)
