"""Extractor agent: identifies and extracts open/unsolved questions from medical documents.

Analogous to ResearchMath-14K's Extractor Agent. Processes raw documents and
outputs structured open question records.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator

from openai import OpenAI

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an expert medical research analyst. Your task is to extract open, unsolved, \
or unanswered questions from medical/biomedical/clinical research documents.

For each document, identify questions that are:
1. EXPLICITLY stated as open, unsolved, unresolved, or requiring further research
2. IMPLICITLY open — where the document describes knowledge gaps, insufficient evidence, \
   or areas needing investigation
3. Questions from "Future Directions" or "Limitations" sections

For each question, provide:
- The verbatim quote or close paraphrase from the source
- A clean, self-contained rewrite that can be understood without the source document
- Whether the question is about: diagnosis, treatment, mechanism, epidemiology, methodology, or other
- Difficulty estimate: how specialized the knowledge needed is (1-5)

Output a JSON array of objects with these fields:
- "original_question": verbatim or close paraphrase from source
- "self_contained_question": rewritten to be standalone (include necessary medical context, \
  define abbreviations, specify the condition/population)
- "question_type": one of ["diagnosis", "treatment", "mechanism", "epidemiology", \
  "methodology", "prognosis", "prevention", "other"]
- "clinical_domain": the medical specialty or field
- "why_open": brief explanation of why this is an open question (1-2 sentences)
- "difficulty": integer 1-5 (1=medical student level, 5=frontier research)

ACCEPT vs REJECT (be selective — quality over quantity):
- ACCEPT: a specific, researchable question where the field genuinely lacks an answer
  (mechanism unknown, no validated method, conflicting/insufficient evidence).
- REJECT: vague aspirations ("more research is needed"), rhetorical/boilerplate "future
  directions" with no concrete question, questions already answered in the same document,
  or implementation asks with known solutions.

WORKED EXAMPLES.
Source: "Although AQP4 mislocalization is observed in AD, whether it is a cause or
consequence of glymphatic failure remains undefined; future work should clarify this."
-> ACCEPT: {"original_question":"whether AQP4 mislocalization is a cause or consequence of
glymphatic failure remains undefined","self_contained_question":"Is aquaporin-4 (AQP4)
mislocalization a cause or a consequence of glymphatic clearance failure in Alzheimer's
disease?","question_type":"mechanism","clinical_domain":"Neurology","why_open":"Observed
association but causal direction not established.","difficulty":4}
Source: "Further studies with larger sample sizes are warranted." -> REJECT (boilerplate).

If the document contains no identifiable open questions, return an empty array [].
Prefer 5-15 HIGH-QUALITY open questions over many weak ones.\
"""

USER_TEMPLATE = """\
Document Source: {source} ({source_id})
Title: {title}

--- Abstract / Content ---
{content}
---

Extract all open/unsolved medical questions from this document.\
"""


@dataclass
class ExtractedQuestion:
    source: str
    source_id: str
    source_url: str
    source_title: str
    original_question: str
    self_contained_question: str
    question_type: str
    clinical_domain: str
    why_open: str
    difficulty: int
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class QuestionExtractor:
    def __init__(
        self,
        model: str = "gpt-4.1",
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 8192,
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

    def extract_from_document(self, doc: dict) -> list[ExtractedQuestion]:
        content = doc.get("full_text") or doc.get("abstract", "")
        if not content or len(content.strip()) < 100:
            return []

        # Truncate very long documents
        if len(content) > 15000:
            content = content[:15000] + "\n[... truncated ...]"

        user_msg = USER_TEMPLATE.format(
            source=doc["source"],
            source_id=doc["source_id"],
            title=doc["title"],
            content=content,
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
            logger.error(f"LLM extraction failed for {doc['source_id']}: {e}")
            return []

        text = resp.choices[0].message.content
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                items = parsed.get("questions", parsed.get("items", []))
            else:
                items = parsed
        except json.JSONDecodeError:
            logger.warning(f"JSON parse failed for {doc['source_id']}")
            return []

        results = []
        for item in items:
            if not isinstance(item, dict):
                continue
            results.append(ExtractedQuestion(
                source=doc["source"],
                source_id=doc["source_id"],
                source_url=doc.get("url", ""),
                source_title=doc["title"],
                original_question=item.get("original_question", ""),
                self_contained_question=item.get("self_contained_question", ""),
                question_type=item.get("question_type", "other"),
                clinical_domain=item.get("clinical_domain", ""),
                why_open=item.get("why_open", ""),
                difficulty=item.get("difficulty", 3),
                metadata={
                    "authors": doc.get("authors", []),
                    "publication_date": doc.get("publication_date", ""),
                    "keywords": doc.get("keywords", []),
                },
            ))

        logger.info(f"Extracted {len(results)} questions from {doc['source_id']}")
        return results

    def extract_batch(
        self, docs: list[dict], output_path: Path
    ) -> list[ExtractedQuestion]:
        all_questions = []
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for i, doc in enumerate(docs):
                questions = self.extract_from_document(doc)
                for q in questions:
                    f.write(json.dumps(q.to_dict(), ensure_ascii=False) + "\n")
                all_questions.extend(questions)

                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i+1}/{len(docs)} docs, {len(all_questions)} questions so far")

        logger.info(f"Extraction complete: {len(all_questions)} questions from {len(docs)} documents")
        return all_questions
