"""Refiner agent: enriches extracted questions with taxonomy, status verification, and MCP tool mapping.

Analogous to ResearchMath-14K's Refiner Agent. Takes extracted questions and:
1. Rewrites them to be fully self-contained
2. Assigns 3-level taxonomy labels
3. Verifies open/resolved status by searching recent literature
4. Maps relevant MCP tools that could help answer the question
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path

from openai import OpenAI

from .taxonomy import TAXONOMY, get_level1_categories, get_all_level2_flat

logger = logging.getLogger(__name__)

TAXONOMY_STR = "\n".join(
    f"- {l1}: {', '.join(l2s)}"
    for l1, l2s in {k: list(v.keys()) for k, v in TAXONOMY.items()}.items()
)

MCP_TOOLS_DESC = """\
Available MCP tools for answering medical questions:
- pubmed: Search PubMed literature database
- clinicaltrialsgov: Search ClinicalTrials.gov for trials
- openfda: Query FDA drug labels, adverse events, recalls
- opentargets: Drug-target-disease associations (Open Targets)
- chembl: Compound bioactivity, targets, drug development data (ChEMBL)
- uniprot: Protein sequences, function, structure (UniProt)
- pubchem: Molecular properties, bioassays, compound data (PubChem)
- kegg: Metabolic pathways, genes, reactions, diseases (KEGG)
- ncbi-datasets: Genomic data, gene info, taxonomy (NCBI)
- biomcp: Integrated biomedical queries (genes, variants, trials, drugs)
- healthcare: FDA, ICD-10, medRxiv, NCBI Bookshelf, BMI\
"""

SYSTEM_PROMPT = f"""\
You are an expert medical research curator. Your task is to refine and enrich \
open medical questions for a benchmark dataset.

For each question, you must:

1. REWRITE the question to be fully self-contained:
   - Define all medical abbreviations on first use
   - Include necessary clinical context (condition, population, intervention)
   - Make it understandable to a medical professional without the source paper
   - Expand from the original ~100-300 chars to ~500-1500 chars

2. CLASSIFY using this medical taxonomy:
{TAXONOMY_STR}

3. ASSESS the resolution status:
   - "open": question is explicitly unresolved, no definitive answer exists
   - "partially_answered": some progress but incomplete understanding
   - "answered": the question has been definitively answered in recent literature
   - "unknown": cannot determine current status

4. MAP relevant MCP tools that could help investigate this question:
{MCP_TOOLS_DESC}

5. RATE difficulty on three axes (1-5 each):
   - clinical_knowledge: how specialized the medical background needed
   - research_depth: how many papers/trials need to be consulted
   - multi_step_reasoning: how many data sources must be integrated

6. VERIFY via academic venues:
   - Search for workshops, conferences, or journal special issues where this problem \
     has been discussed as an open challenge
   - Look for WHO/NIH priority lists, Cochrane review conclusions, or clinical guideline \
     gaps that reference this topic
   - Note any relevant medical conferences (e.g., ASCO, AHA, RSNA, NeurIPS Health, \
     MICCAI, AMIA) or workshops where this is listed as an open problem

Output a JSON object with these fields:
- "self_contained_problem": the rewritten, self-contained question (string)
- "taxonomy_level_1": primary domain from the taxonomy (string)
- "taxonomy_level_2": sub-domain (string)
- "taxonomy_level_3": specific topic tag (string)
- "open_status": one of ["open", "partially_answered", "answered", "unknown"]
- "status_reasoning": why you assigned this status (1-3 sentences)
- "status_evidence": any evidence for resolution (string, can be empty)
- "verification_venues": list of conferences/workshops/journals/organizations that \
  discuss this as an open problem (list of strings)
- "verification_notes": how/where you found corroborating evidence (string)
- "relevant_mcp_tools": list of tool names useful for investigating this question
- "mcp_tool_rationale": brief explanation of how each tool helps
- "difficulty_clinical_knowledge": integer 1-5
- "difficulty_research_depth": integer 1-5
- "difficulty_multi_step_reasoning": integer 1-5\
"""

USER_TEMPLATE = """\
Source: {source} ({source_id})
Source Title: {source_title}
Original Question: {original_question}
Current Self-Contained Version: {self_contained_question}
Question Type: {question_type}
Clinical Domain: {clinical_domain}
Why Open: {why_open}
Publication Date: {publication_date}

Refine this question.\
"""


@dataclass
class RefinedQuestion:
    source: str
    source_id: str
    source_url: str
    source_title: str
    original_question: str
    self_contained_problem: str
    taxonomy_level_1: str
    taxonomy_level_2: str
    taxonomy_level_3: str
    open_status: str
    status_reasoning: str
    status_evidence: str
    verification_venues: list[str] = field(default_factory=list)
    verification_notes: str = ""
    relevant_mcp_tools: list[str] = field(default_factory=list)
    mcp_tool_rationale: str = ""
    difficulty_clinical_knowledge: int = 3
    difficulty_research_depth: int = 3
    difficulty_multi_step_reasoning: int = 3
    question_type: str = ""
    why_open: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class QuestionRefiner:
    def __init__(
        self,
        model: str = "gpt-4.1",
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ):
        kwargs = {}
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key
        self.client = OpenAI(**kwargs)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def refine_question(self, question: dict) -> RefinedQuestion | None:
        user_msg = USER_TEMPLATE.format(
            source=question.get("source", ""),
            source_id=question.get("source_id", ""),
            source_title=question.get("source_title", ""),
            original_question=question.get("original_question", ""),
            self_contained_question=question.get("self_contained_question", ""),
            question_type=question.get("question_type", ""),
            clinical_domain=question.get("clinical_domain", ""),
            why_open=question.get("why_open", ""),
            publication_date=question.get("metadata", {}).get("publication_date", ""),
        )

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            logger.error(f"Refinement failed for {question.get('source_id')}: {e}")
            return None

        text = resp.choices[0].message.content
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning(f"JSON parse failed for {question.get('source_id')}")
            return None

        return RefinedQuestion(
            source=question.get("source", ""),
            source_id=question.get("source_id", ""),
            source_url=question.get("source_url", ""),
            source_title=question.get("source_title", ""),
            original_question=question.get("original_question", ""),
            self_contained_problem=data.get("self_contained_problem", ""),
            taxonomy_level_1=data.get("taxonomy_level_1", ""),
            taxonomy_level_2=data.get("taxonomy_level_2", ""),
            taxonomy_level_3=data.get("taxonomy_level_3", ""),
            open_status=data.get("open_status", "unknown"),
            status_reasoning=data.get("status_reasoning", ""),
            status_evidence=data.get("status_evidence", ""),
            verification_venues=data.get("verification_venues", []),
            verification_notes=data.get("verification_notes", ""),
            relevant_mcp_tools=data.get("relevant_mcp_tools", []),
            mcp_tool_rationale=data.get("mcp_tool_rationale", ""),
            difficulty_clinical_knowledge=data.get("difficulty_clinical_knowledge", 3),
            difficulty_research_depth=data.get("difficulty_research_depth", 3),
            difficulty_multi_step_reasoning=data.get("difficulty_multi_step_reasoning", 3),
            question_type=question.get("question_type", ""),
            why_open=question.get("why_open", ""),
            metadata=question.get("metadata", {}),
        )

    def refine_batch(self, questions: list[dict], output_path: Path) -> list[RefinedQuestion]:
        results = []
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for i, q in enumerate(questions):
                refined = self.refine_question(q)
                if refined:
                    f.write(json.dumps(refined.to_dict(), ensure_ascii=False) + "\n")
                    results.append(refined)

                if (i + 1) % 10 == 0:
                    logger.info(f"Refined {i+1}/{len(questions)}, {len(results)} successful")

        logger.info(f"Refinement complete: {len(results)}/{len(questions)} questions refined")
        return results
