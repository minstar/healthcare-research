"""Retrieval-grounded status verification for open medical questions.

This module is the missing piece that ResearchMath-14K's Refiner Agent had and our
original single-call refiner lacked: BEFORE the LLM judges whether a question is still
open, we gather *real* evidence from the literature / trial registries, dispatched by
source type. The LLM then judges strictly from that evidence bundle.

Design contract
---------------
- `gather_evidence(question)` is deterministic and tool-grounded (no LLM). It returns an
  `EvidenceBundle` whose `items` carry verifiable IDs (PMID / NCT) and dates.
- The bundle distinguishes *follow-up* evidence (papers that cite the source, or trial
  results posted AFTER the question was raised) from the source itself. Resolution can
  only come from follow-ups.
- `method` records HOW evidence was obtained (retrieval / synthetic / none) so the
  downstream label is auditable and we can measure coverage.

Cost: all calls here are free REST endpoints (NCBI E-utils, ClinicalTrials.gov,
Semantic Scholar). Only the refiner's final judgment is an LLM call.
"""
from __future__ import annotations

import json
import logging
import re
import time
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

NCBI_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
NCBI_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
NCBI_ELINK = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
NCBI_ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
CTGOV = "https://clinicaltrials.gov/api/v2/studies"
S2_PAPER = "https://api.semanticscholar.org/graph/v1/paper"
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"

_TIMEOUT = 30
_CONTROL_CHARS = re.compile(r"[\x00-\x1f]+")


def _safe_json(text: str) -> dict:
    """NCBI E-utils JSON often contains raw control chars that break json.loads."""
    return json.loads(_CONTROL_CHARS.sub(" ", text))


class _RateLimiter:
    """Process-wide spacing between NCBI calls. 3 req/s without key, ~9 with one."""

    def __init__(self, min_interval: float):
        self._min = min_interval
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self):
        with self._lock:
            now = time.monotonic()
            sleep = self._min - (now - self._last)
            if sleep > 0:
                time.sleep(sleep)
            self._last = time.monotonic()


@dataclass
class EvidenceItem:
    id: str               # "PMID:123" / "NCT:456" / "S2:..."
    title: str = ""
    date: str = ""        # ISO-ish; "" if unknown
    snippet: str = ""     # abstract / results summary, truncated
    role: str = "followup"  # "source" | "followup" | "result"


@dataclass
class EvidenceBundle:
    source_id: str
    source_type: str           # pubmed | trial | arxiv | synthetic | unknown
    source_date: str = ""
    method: str = "none"       # retrieval | synthetic | none
    n_followups: int = 0
    items: list[EvidenceItem] = field(default_factory=list)
    notes: str = ""

    def to_prompt_block(self, max_items: int = 8) -> str:
        """Render the evidence as a compact block for the judgment LLM."""
        if not self.items:
            return "NO EXTERNAL EVIDENCE RETRIEVED."
        lines = [f"Source raised: {self.source_date or 'unknown date'} ({self.source_type})"]
        for it in self.items[:max_items]:
            tag = it.role.upper()
            lines.append(f"[{tag} {it.id} | {it.date}] {it.title}\n    {it.snippet[:400]}")
        return "\n".join(lines)

    def evidence_ids(self) -> set[str]:
        return {it.id for it in self.items}


class StatusVerifier:
    def __init__(
        self,
        ncbi_api_key: str | None = None,
        cache_path: str | Path | None = None,
        citing_top_k: int = 6,
        email: str = "minstar@upstage.ai",
        min_interval: float | None = None,
    ):
        self.api_key = ncbi_api_key
        self.top_k = citing_top_k
        self.email = email
        self._limiter = _RateLimiter(min_interval if min_interval is not None
                                     else (0.11 if ncbi_api_key else 0.34))
        self._session = requests.Session()
        self._retries = 3
        self._cache_path = Path(cache_path) if cache_path else None
        self._cache: dict[str, dict] = {}
        if self._cache_path and self._cache_path.exists():
            for line in self._cache_path.open():
                try:
                    rec = json.loads(line)
                    self._cache[rec["source_id"]] = rec["bundle"]
                except Exception:
                    continue

    # ---- public API -------------------------------------------------------
    def gather_evidence(self, question: dict) -> EvidenceBundle:
        sid = str(question.get("source_id", ""))
        if sid in self._cache:
            return EvidenceBundle(**_rehydrate(self._cache[sid]))

        stype = classify_source(sid)
        try:
            if stype == "pubmed":
                bundle = self._verify_pubmed(question, sid)
            elif stype == "trial":
                bundle = self._verify_trial(question, sid)
            elif stype == "arxiv":
                bundle = self._verify_arxiv(question, sid)
            elif stype == "synthetic":
                bundle = EvidenceBundle(sid, "synthetic", method="synthetic",
                                        notes="Templated/API-derived question; not literature-extracted.")
            else:
                bundle = EvidenceBundle(sid, "unknown", method="none")
        except Exception as e:  # network/parse failure must degrade gracefully, not crash
            logger.warning(f"evidence gathering failed for {sid}: {e}")
            bundle = EvidenceBundle(sid, stype, method="none", notes=f"error: {e}")

        self._persist(bundle)
        return bundle

    def screen(self, question: dict) -> dict:
        """Fast, LLM-free contamination screen (no abstract fetching).

        Returns deterministic signals + a `contamination` verdict computed against the
        item's CURRENT open_status label. Used by the full-corpus audit.
        """
        sid = str(question.get("source_id", ""))
        label = question.get("open_status", "")
        stype = classify_source(sid)
        out = {"source_id": sid, "source_type": stype, "current_label": label,
               "method": "none", "n_followups": 0, "src_year": "", "flag": "ok", "detail": ""}
        try:
            if stype == "synthetic":
                out.update(method="synthetic", flag="synthetic_template",
                           detail="KEGG/UniProt-derived; not literature-extracted")
            elif stype == "trial":
                nct = sid.split(":", 1)[1] if ":" in sid else sid
                self._limiter.wait()
                r = self._session.get(f"{CTGOV}/{nct}",
                    params={"fields": "OverallStatus,CompletionDate,ResultsFirstPostDate,HasResults"},
                    timeout=_TIMEOUT)
                if r.status_code == 200:
                    sm = r.json().get("protocolSection", {}).get("statusModule", {})
                    status = sm.get("overallStatus", "")
                    results = bool(r.json().get("hasResults")) or bool(sm.get("resultsFirstPostDateStruct"))
                    out.update(method="retrieval", src_year=sm.get("completionDateStruct", {}).get("date", ""),
                               n_followups=1 if results else 0, detail=f"{status}; results={results}")
                    if status in ("TERMINATED", "WITHDRAWN", "SUSPENDED"):
                        out["flag"] = "trial_dead"  # will never answer; weak provenance
                    elif status == "COMPLETED" and results and label == "open":
                        out["flag"] = "trial_resolved_but_open"
                    elif status == "COMPLETED":
                        out["flag"] = "trial_completed"  # answerable from publications
                    # RECRUITING / ACTIVE / NOT_YET_RECRUITING → genuinely result-pending
                else:
                    out["detail"] = f"ctgov {r.status_code}"
            elif stype == "pubmed":
                pmid = sid.split(":", 1)[1] if ":" in sid else sid
                cites = self._epmc_citations(pmid, self.top_k, fetch_abstracts=False)
                out.update(method="retrieval", n_followups=len(cites))
                if cites:
                    out["detail"] = f"latest citing yr={cites[0].get('date','')}"
                    if len(cites) >= self.top_k:
                        out["flag"] = "heavy_followup_recheck"  # lots of later work → re-judge candidate
                else:
                    out["flag"] = "no_followup_evidence"
            elif stype == "arxiv":
                # arXiv here are curated expert open-problem papers (valid sources);
                # they just lack a cheap PMID citation path. Neutral, not contamination.
                out.update(method="none", flag="arxiv_citation_check_needed",
                           detail="arXiv open-problem paper; resolution via Semantic Scholar (Stage 2)")
            else:
                out["flag"] = "unverifiable_source"
        except Exception as e:
            out["detail"] = f"error: {e}"
        return out

    # ---- source-type strategies ------------------------------------------
    def _verify_pubmed(self, question: dict, sid: str) -> EvidenceBundle:
        pmid = sid.split(":", 1)[1] if ":" in sid else sid

        # Source date: prefer Europe PMC (one call, returns date), fall back to NCBI efetch.
        src_date = self._epmc_pubyear(pmid) or self._pubmed_dates([pmid]).get(pmid, "")
        bundle = EvidenceBundle(sid, "pubmed", source_date=src_date, method="retrieval")

        # 1) Papers that CITE the source — resolution can only come from these.
        #    Europe PMC's citations index is far more reliable than NCBI elink citedin.
        citing = self._epmc_citations(pmid, self.top_k)
        if not citing:  # fallback to NCBI elink, then enrich via efetch
            ids = [p for p in self._elink_citedin(pmid) if p != pmid][: self.top_k]
            meta = self._pubmed_meta(ids)
            citing = [{"id": p, "title": meta.get(p, {}).get("title", ""),
                       "date": meta.get(p, {}).get("date", ""),
                       "abstract": meta.get(p, {}).get("abstract", "")} for p in ids]
        bundle.n_followups = len(citing)

        for c in citing[: self.top_k]:
            bundle.items.append(EvidenceItem(
                id=f"PMID:{c['id']}", title=c.get("title", ""), date=str(c.get("date", "")),
                snippet=c.get("abstract", ""), role="followup",
            ))
        if not bundle.items:
            bundle.notes = "No citing or follow-up papers found."
        return bundle

    # ---- Europe PMC primitives (reliable citation index + abstracts) ------
    def _epmc_pubyear(self, pmid: str) -> str:
        try:
            self._limiter.wait()
            r = self._session.get(f"{EPMC}/search",
                params={"query": f"EXT_ID:{pmid} AND SRC:MED", "format": "json", "resultType": "lite"},
                timeout=_TIMEOUT)
            res = r.json().get("resultList", {}).get("result", [])
            return res[0].get("firstPublicationDate", res[0].get("pubYear", "")) if res else ""
        except Exception:
            return ""

    def _epmc_citations(self, pmid: str, k: int, fetch_abstracts: bool = True) -> list[dict]:
        """Citing papers, most-recent-first. Set fetch_abstracts=False for fast audit."""
        try:
            self._limiter.wait()
            r = self._session.get(f"{EPMC}/MED/{pmid}/citations",
                params={"format": "json", "pageSize": max(k * 3, 25)}, timeout=_TIMEOUT)
            cites = r.json().get("citationList", {}).get("citation", [])
        except Exception:
            return []
        cites = [c for c in cites if c.get("source") == "MED" and c.get("id")]
        cites.sort(key=lambda c: c.get("pubYear", 0), reverse=True)
        cites = cites[:k]
        abstracts = self._epmc_abstracts([c["id"] for c in cites]) if fetch_abstracts else {}
        return [{"id": c["id"], "title": c.get("title", ""), "date": c.get("pubYear", ""),
                 "abstract": abstracts.get(c["id"], "")} for c in cites]

    def _epmc_abstracts(self, pmids: list[str]) -> dict[str, str]:
        if not pmids:
            return {}
        q = " OR ".join(f"EXT_ID:{p}" for p in pmids)
        try:
            self._limiter.wait()
            r = self._session.get(f"{EPMC}/search",
                params={"query": f"({q}) AND SRC:MED", "format": "json",
                        "resultType": "core", "pageSize": len(pmids)}, timeout=_TIMEOUT)
            res = r.json().get("resultList", {}).get("result", [])
        except Exception:
            return {}
        return {x.get("pmid", x.get("id", "")): (x.get("abstractText") or "")[:800] for x in res}

    def _verify_trial(self, question: dict, sid: str) -> EvidenceBundle:
        nct = sid.split(":", 1)[1] if ":" in sid else sid
        self._limiter.wait()
        r = self._session.get(
            f"{CTGOV}/{nct}",
            params={"fields": "OverallStatus,CompletionDate,ResultsFirstPostDate,HasResults"},
            timeout=_TIMEOUT,
        )
        if r.status_code != 200:
            return EvidenceBundle(sid, "trial", method="none", notes=f"ctgov {r.status_code}")
        d = r.json()
        sm = d.get("protocolSection", {}).get("statusModule", {})
        status = sm.get("overallStatus", "")
        comp = sm.get("completionDateStruct", {}).get("date", "")
        results = bool(d.get("hasResults")) or bool(sm.get("resultsFirstPostDateStruct"))
        bundle = EvidenceBundle(sid, "trial", source_date=comp, method="retrieval")
        bundle.items.append(EvidenceItem(
            id=f"NCT:{nct}", title=f"Trial status: {status}", date=comp,
            snippet=f"overallStatus={status}; completion={comp}; results_posted={results}",
            role="result",
        ))
        bundle.n_followups = 1 if results else 0
        bundle.notes = "completed_with_results" if (status == "COMPLETED" and results) else status
        return bundle

    def _verify_arxiv(self, question: dict, sid: str) -> EvidenceBundle:
        arxiv_id = sid.split("v")[0]
        self._limiter.wait()
        try:
            r = self._session.get(
                f"{S2_PAPER}/arXiv:{arxiv_id}",
                params={"fields": "title,citationCount,citations.title,citations.year,citations.abstract"},
                timeout=_TIMEOUT,
            )
            if r.status_code != 200:
                return EvidenceBundle(sid, "arxiv", method="none", notes=f"s2 {r.status_code}")
            d = r.json()
            bundle = EvidenceBundle(sid, "arxiv", method="retrieval")
            bundle.n_followups = d.get("citationCount", 0)
            for c in (d.get("citations") or [])[: self.top_k]:
                bundle.items.append(EvidenceItem(
                    id=f"S2:{c.get('title','')[:40]}", title=c.get("title", ""),
                    date=str(c.get("year", "")), snippet=(c.get("abstract") or "")[:400],
                    role="followup",
                ))
            return bundle
        except Exception as e:
            return EvidenceBundle(sid, "arxiv", method="none", notes=f"error: {e}")

    # ---- NCBI primitives --------------------------------------------------
    def _elink_citedin(self, pmid: str) -> list[str]:
        r = self._ncbi_get(NCBI_ELINK, {
            "dbfrom": "pubmed", "db": "pubmed", "id": pmid,
            "linkname": "pubmed_pubmed_citedin", "retmode": "json",
        })
        if r is None or r.status_code != 200:
            return []
        linksets = _safe_json(r.text).get("linksets") or [{}]
        ls = linksets[0].get("linksetdbs") or []
        ids = ls[0].get("links", []) if ls else []
        return list(reversed(ids))  # NCBI returns oldest-first; we want most recent first

    def _esearch_since(self, terms: str, since_date: str) -> list[str]:
        date_filter = ""
        m = re.search(r"(\d{4})", since_date or "")
        if m:
            date_filter = f' AND ("{m.group(1)}"[Date - Publication] : "3000"[Date - Publication])'
        r = self._ncbi_get(NCBI_ESEARCH, {
            "db": "pubmed", "term": terms + date_filter, "retmax": self.top_k,
            "retmode": "json", "sort": "relevance",
        })
        if r is None or r.status_code != 200:
            return []
        return _safe_json(r.text).get("esearchresult", {}).get("idlist", [])

    def _pubmed_dates(self, pmids: list[str]) -> dict[str, str]:
        return {k: v.get("date", "") for k, v in self._pubmed_meta(pmids).items()}

    def _pubmed_meta(self, pmids: list[str]) -> dict[str, dict]:
        if not pmids:
            return {}
        r = self._ncbi_get(NCBI_EFETCH, {
            "db": "pubmed", "id": ",".join(pmids), "retmode": "xml", "rettype": "abstract",
        })
        if r is None or r.status_code != 200:
            return {}
        import xml.etree.ElementTree as ET
        out: dict[str, dict] = {}
        try:
            root = ET.fromstring(r.text)
        except ET.ParseError:
            return {}
        for art in root.findall(".//PubmedArticle"):
            pid = art.findtext(".//PMID", "")
            title = art.findtext(".//ArticleTitle", "") or ""
            abstract = " ".join(e.text or "" for e in art.findall(".//AbstractText"))
            y = art.findtext(".//PubDate/Year", "") or art.findtext(".//PubMedPubDate/Year", "")
            mo = art.findtext(".//PubDate/Month", "")
            out[pid] = {"title": title, "abstract": abstract[:800], "date": f"{y}-{mo}".strip("-")}
        return out

    def _ncbi_get(self, url: str, params: dict) -> requests.Response | None:
        """NCBI E-utils returns transient 500s frequently; retry with backoff."""
        r = None
        for attempt in range(self._retries):
            self._limiter.wait()
            try:
                r = self._session.get(url, params=self._ncbi_params(params), timeout=_TIMEOUT)
            except requests.RequestException:
                r = None
            # NCBI intermittently returns 500 or an empty 200 body under load; retry both.
            if r is not None and r.status_code == 200 and r.text.strip():
                return r
            time.sleep(0.6 * (attempt + 1))
        return r

    def _ncbi_params(self, p: dict) -> dict:
        p = dict(p)
        p["email"] = self.email
        if self.api_key:
            p["api_key"] = self.api_key
        return p

    def _persist(self, bundle: EvidenceBundle):
        self._cache[bundle.source_id] = asdict(bundle)
        if self._cache_path:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self._cache_path.open("a") as f:
                f.write(json.dumps({"source_id": bundle.source_id, "bundle": asdict(bundle)},
                                   ensure_ascii=False) + "\n")


# ---- helpers --------------------------------------------------------------
def classify_source(sid: str) -> str:
    if sid.startswith("PMID"):
        return "pubmed"
    if sid.startswith("NCT"):
        return "trial"
    if re.match(r"\d{4}\.\d{4,5}", sid):
        return "arxiv"
    if sid.startswith("hsa") or re.match(r"[OPQ][0-9][A-Z0-9]{3}[0-9]", sid) or sid.startswith("map"):
        return "synthetic"  # KEGG pathway / UniProt accession → templated
    return "unknown"


def _key_terms(text: str, k: int = 6) -> str:
    """Cheap keyword extraction for the follow-up search (no LLM)."""
    stop = {"what", "which", "how", "does", "the", "and", "for", "are", "with", "can",
            "of", "in", "to", "is", "be", "that", "this", "remain", "open", "question",
            "patients", "disease", "such", "why", "underpinning", "mechanistic"}
    words = re.findall(r"[A-Za-z][A-Za-z\-]{3,}", text.lower())
    seen, out = set(), []
    for w in words:
        if w in stop or w in seen:
            continue
        seen.add(w)
        out.append(w)
        if len(out) >= k:
            break
    return " AND ".join(out)


def _rehydrate(d: dict) -> dict:
    d = dict(d)
    d["items"] = [EvidenceItem(**it) for it in d.get("items", [])]
    return d
