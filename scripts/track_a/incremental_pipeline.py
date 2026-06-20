#!/usr/bin/env python3
"""End-to-end incremental pipeline: raw documents -> extracted + filtered + deduped + refined questions.

Takes new raw documents (e.g. from MeSH expansion), processes them through the full
pipeline (extract -> quality filter -> dedup -> refine), and merges into the main dataset.

Steps:
  1. Read new docs from input JSONL
  2. Batch extraction via Claude CLI Haiku (30 docs per batch, 5 docs per Claude call)
  3. Quality filtering (reuses classify_removal from filter_quality.py)
  4. Incremental dedup against existing corpus (sentence-transformers cosine sim >= 0.90)
  5. Refinement via Claude CLI Haiku
  6. Merge into main dataset

Usage:
    python scripts/track_a/incremental_pipeline.py \\
        --input data/raw/pubmed_mesh/documents.jsonl \\
        [--existing data/extracted/all_questions_refined.jsonl] \\
        [--workers 10]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path("/data/project/private/minstar/workspace/healthcare-research")
DEFAULT_EXISTING = BASE_DIR / "data" / "extracted" / "all_questions_refined.jsonl"
OUTPUT_DIR = BASE_DIR / "data" / "expanded"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompts (identical to scripts/extract_batch.py and scripts/refine_batch.py)
# ---------------------------------------------------------------------------
EXTRACT_SYSTEM_PROMPT = """You are a medical research analyst. Given a scientific document (title + abstract), extract ALL open/unsolved questions, problems, or challenges mentioned or implied.

For each open question found, output a JSON object with these fields:
- source_id: the document's source_id
- source_url: the document's url
- source_title: the document's title
- original_question: the exact text from the document that signals an open question
- self_contained_question: a reformulated, self-contained version of the question (define all abbreviations, include clinical context)
- question_type: one of [mechanism, treatment, diagnosis, prevention, epidemiology, methodology, prognosis]
- clinical_domain: the medical specialty (e.g., Neurology, Oncology, Cardiology)
- why_open: brief explanation of why this question remains unsolved
- difficulty: 1-5 rating (5 = hardest)

If no open questions are found, output nothing for that document.
Output ONLY valid JSON objects, one per line (JSONL format). No markdown, no explanations."""

REFINE_SYSTEM_PROMPT = """You are an expert medical research curator. For each question below, output a JSON object with:

- "idx": the question index (integer, as given)
- "taxonomy_l1": primary domain from: Clinical Medicine, Oncology, Neuroscience & Psychiatry, Infectious Disease & Immunology, Cardiovascular Medicine, Genomics & Precision Medicine, Pharmacology & Drug Discovery, Public Health & Epidemiology, Rare & Orphan Diseases, Surgical Sciences, Medical AI & Informatics, Other
- "taxonomy_l2": sub-domain (e.g., "Cancer Biology", "Neurodegeneration", "Antimicrobial Resistance")
- "taxonomy_l3": specific topic tag (e.g., "tumor microenvironment", "CRISPR therapeutics")
- "open_status": one of ["open", "partially_answered", "answered", "unknown"]
- "status_reasoning": 1-2 sentences on why this status
- "verification_venues": list of conferences/journals/workshops where this is discussed as open (e.g., ["ASCO", "Nature Medicine", "WHO Priority Pathogens"])
- "relevant_mcp_tools": list from [pubmed, clinicaltrialsgov, openfda, opentargets, chembl, uniprot, pubchem, kegg, ncbi-datasets, biomcp]
- "difficulty_clinical_knowledge": 1-5
- "difficulty_research_depth": 1-5
- "difficulty_multi_step_reasoning": 1-5

Output ONLY valid JSON objects, one per line (JSONL). No markdown, no explanations."""


# ---------------------------------------------------------------------------
# Quality filter (reuse from scripts/filter_quality.py)
# ---------------------------------------------------------------------------
# Import classify_removal. We add the scripts dir to sys.path so we can import it.
sys.path.insert(0, str(BASE_DIR / "scripts"))
try:
    from filter_quality import classify_removal
except ImportError:
    logger.warning("Could not import classify_removal from filter_quality.py, using inline copy")

    # Inline fallback: minimal version of the quality filter
    KEGG_TEMPLATES = ["rate-limiting or bottleneck", "response heterogeneity to drugs targeting", "functional interactions between"]
    NCT_PROTOCOL_PHRASES = ["the investigators plan", "subjects will be recruited", "the study will take place", "this study is", "this trial is"]
    LAZY_WHY_STARTERS = ["Research article", "Literature explicitly", "Explicitly identified", "Listed as unsolved"]
    META_PATTERNS = ["what are the key unresolved questions regarding", "what aspects of", "what are the unresolved questions", "what are the open questions", "remain unsolved or unclear", "this manuscript provides"]
    COCHRANE_FRAGMENTS = ["insufficient evidence", "no evidence that", "uncertain whether", "unclear or high", "findings of this review"]
    GENERIC_TEMPLATES = ["what treatments are needed to address", "why treatment resistance a leading", "what remains remains"]
    GARBLE_PATTERNS = ["in the context of", "what are the barriers to effective", "cochrane reviews", "an overview of cochrane"]

    def classify_removal(q: dict) -> str | None:
        sid = q.get("source_id", "")
        url = q.get("source_url", "")
        text = q.get("self_contained_question", "")
        text_lower = text.lower()
        why = q.get("why_open", "")
        difficulty = q.get("difficulty", 0)
        if len(text) < 30:
            return "short (<30 chars)"
        if sid.startswith("fda_"):
            return "FDA adverse event"
        if sid.startswith("hsa"):
            if any(t in text_lower for t in KEGG_TEMPLATES):
                return "KEGG template"
            if difficulty == 1:
                return "KEGG D1 fragment"
        if sid.startswith("NCT"):
            if difficulty <= 2:
                return "NCT difficulty <= 2"
            if any(t in text_lower for t in NCT_PROTOCOL_PHRASES):
                return "NCT protocol text"
        if "stackexchange" in url:
            return "StackExchange"
        if "?" not in text and len(text) < 80:
            return "non-question fragment"
        if "?" not in text:
            return "missing question mark"
        if text.startswith("How can we better understand"):
            return "HCWBU prefix"
        if any(why.startswith(s) for s in LAZY_WHY_STARTERS):
            return "lazy extraction"
        if any(p in text_lower for p in META_PATTERNS):
            return "meta-question"
        if any(p in text_lower for p in COCHRANE_FRAGMENTS):
            return "Cochrane fragment"
        if any(p in text_lower for p in GENERIC_TEMPLATES):
            return "generic template"
        if text[0].islower():
            return "lowercase fragment"
        if text.rstrip("?").rstrip().endswith("."):
            return "title-in-question"
        if any(text_lower.startswith(p) or p in text_lower for p in GARBLE_PATTERNS):
            return "garbled pattern"
        if len(text) < 60:
            words = text.split()
            if len(words) < 6 or text.count("?") > 1:
                return "short broken"
        return None


# ---------------------------------------------------------------------------
# State management (resumable pipeline)
# ---------------------------------------------------------------------------
class PipelineState:
    """Track which pipeline steps have completed for resumability."""

    def __init__(self, state_path: Path):
        self.path = state_path
        self.state: dict = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            with open(self.path) as f:
                return json.load(f)
        return {
            "started_at": datetime.now().isoformat(),
            "steps_completed": [],
            "extraction_batches_done": 0,
            "total_extracted": 0,
            "total_after_filter": 0,
            "total_after_dedup": 0,
            "total_after_refine": 0,
            "total_merged": 0,
        }

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.state, f, indent=2, ensure_ascii=False)

    def is_step_done(self, step: str) -> bool:
        return step in self.state.get("steps_completed", [])

    def mark_step_done(self, step: str):
        if step not in self.state["steps_completed"]:
            self.state["steps_completed"].append(step)
        self.save()

    def update(self, key: str, value):
        self.state[key] = value
        self.save()


# ---------------------------------------------------------------------------
# Step 1: Read input documents
# ---------------------------------------------------------------------------
def load_documents(input_path: Path) -> list[dict]:
    """Load raw documents from a JSONL file."""
    docs = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    docs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    logger.info(f"Loaded {len(docs)} documents from {input_path}")
    return docs


# ---------------------------------------------------------------------------
# Step 2: Extraction via Claude CLI Haiku
# ---------------------------------------------------------------------------
def extract_from_docs_chunk(docs: list[dict]) -> list[dict]:
    """Use Claude CLI to extract questions from a chunk of documents (max ~5 docs)."""
    prompt_parts = ["Extract all open/unsolved medical questions from these documents:\n"]
    for i, doc in enumerate(docs):
        prompt_parts.append(f"--- Document {i+1} ---")
        prompt_parts.append(f"source_id: {doc.get('source_id', '')}")
        prompt_parts.append(f"url: {doc.get('url', '')}")
        prompt_parts.append(f"title: {doc.get('title', '')}")
        abstract = doc.get("abstract", "")[:3000]
        prompt_parts.append(f"abstract: {abstract}")
        prompt_parts.append("")

    prompt = "\n".join(prompt_parts)

    try:
        result = subprocess.run(
            ["claude", "--print", "--model", "haiku", "--system-prompt", EXTRACT_SYSTEM_PROMPT],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=300,
        )
        output = result.stdout.strip()
    except subprocess.TimeoutExpired:
        logger.warning("Claude CLI timed out during extraction")
        return []
    except Exception as e:
        logger.warning(f"Claude CLI error during extraction: {e}")
        return []

    questions = []
    for line in output.split("\n"):
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        try:
            q = json.loads(line)
            if isinstance(q, dict) and "source_id" in q:
                questions.append(q)
        except json.JSONDecodeError:
            continue

    return questions


def run_extraction(docs: list[dict], work_dir: Path, state: PipelineState, workers: int) -> list[dict]:
    """Run extraction over all documents in batches of 30, with 5 docs per Claude call."""
    extracted_path = work_dir / "extracted.jsonl"

    if state.is_step_done("extraction"):
        logger.info("Extraction already done, loading from cache")
        questions = []
        with open(extracted_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    questions.append(json.loads(line))
        return questions

    batch_size = 30
    chunk_size = 5
    all_questions: list[dict] = []

    # Load partial progress if any
    batches_done = state.state.get("extraction_batches_done", 0)
    if batches_done > 0 and extracted_path.exists():
        with open(extracted_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    all_questions.append(json.loads(line))
        logger.info(f"Resuming extraction from batch {batches_done}, {len(all_questions)} questions so far")

    total_batches = (len(docs) + batch_size - 1) // batch_size

    for batch_idx in range(batches_done, total_batches):
        batch_start = batch_idx * batch_size
        batch_docs = docs[batch_start : batch_start + batch_size]

        # Split batch into chunks of 5 docs
        chunks = [batch_docs[i : i + chunk_size] for i in range(0, len(batch_docs), chunk_size)]
        batch_questions: list[dict] = []

        # Process chunks in parallel (up to `workers` threads)
        effective_workers = min(workers, len(chunks))
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = {executor.submit(extract_from_docs_chunk, chunk): ci for ci, chunk in enumerate(chunks)}
            for future in as_completed(futures):
                ci = futures[future]
                try:
                    result = future.result()
                    batch_questions.extend(result)
                except Exception as e:
                    logger.error(f"  Chunk {ci} failed: {e}")

        all_questions.extend(batch_questions)

        # Append to file incrementally
        with open(extracted_path, "a") as f:
            for q in batch_questions:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")

        state.update("extraction_batches_done", batch_idx + 1)
        state.update("total_extracted", len(all_questions))

        logger.info(
            f"  Batch {batch_idx+1}/{total_batches}: {len(batch_docs)} docs -> "
            f"{len(batch_questions)} questions (total: {len(all_questions)})"
        )

    state.mark_step_done("extraction")
    logger.info(f"Extraction complete: {len(all_questions)} questions from {len(docs)} documents")
    return all_questions


# ---------------------------------------------------------------------------
# Step 3: Quality filtering
# ---------------------------------------------------------------------------
def run_quality_filter(questions: list[dict], work_dir: Path, state: PipelineState) -> list[dict]:
    """Apply quality filters, same rules as scripts/filter_quality.py."""
    filtered_path = work_dir / "filtered.jsonl"

    if state.is_step_done("quality_filter"):
        logger.info("Quality filter already done, loading from cache")
        filtered = []
        with open(filtered_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    filtered.append(json.loads(line))
        return filtered

    kept = []
    removed_counts: dict[str, int] = {}

    for q in questions:
        reason = classify_removal(q)
        if reason:
            removed_counts[reason] = removed_counts.get(reason, 0) + 1
        else:
            kept.append(q)

    total_removed = sum(removed_counts.values())
    logger.info(f"Quality filter: {len(questions)} -> {len(kept)} ({total_removed} removed)")
    for reason, count in sorted(removed_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {reason}: {count}")

    with open(filtered_path, "w") as f:
        for q in kept:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    state.update("total_after_filter", len(kept))
    state.mark_step_done("quality_filter")
    return kept


# ---------------------------------------------------------------------------
# Step 4: Incremental dedup (embedding-based)
# ---------------------------------------------------------------------------
def load_existing_questions(existing_path: Path) -> list[dict]:
    """Load existing question corpus."""
    questions = []
    if existing_path.exists():
        with open(existing_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        questions.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return questions


def compute_embeddings(texts: list[str], model, batch_size: int = 256) -> np.ndarray:
    """Compute sentence embeddings in batches."""
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embs = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
        all_embs.append(embs)
    return np.vstack(all_embs) if all_embs else np.zeros((0, 384))


def run_dedup(
    new_questions: list[dict],
    existing_path: Path,
    work_dir: Path,
    state: PipelineState,
    sim_threshold: float = 0.90,
) -> list[dict]:
    """Deduplicate new questions against existing corpus using cosine similarity."""
    deduped_path = work_dir / "deduped.jsonl"

    if state.is_step_done("dedup"):
        logger.info("Dedup already done, loading from cache")
        deduped = []
        with open(deduped_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    deduped.append(json.loads(line))
        return deduped

    if not new_questions:
        state.update("total_after_dedup", 0)
        state.mark_step_done("dedup")
        return []

    logger.info("Loading sentence-transformers model for dedup...")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Load existing corpus questions
    existing_qs = load_existing_questions(existing_path)
    existing_texts = [q.get("self_contained_question", "") for q in existing_qs]
    logger.info(f"Existing corpus: {len(existing_texts)} questions")

    # Check for cached existing embeddings
    emb_cache_path = work_dir / "existing_embeddings.npy"
    if emb_cache_path.exists() and len(existing_texts) > 0:
        logger.info("Loading cached existing embeddings...")
        existing_embs = np.load(str(emb_cache_path))
        # Validate cache matches current corpus size
        if existing_embs.shape[0] != len(existing_texts):
            logger.info("Cache size mismatch, recomputing existing embeddings...")
            existing_embs = compute_embeddings(existing_texts, model)
            np.save(str(emb_cache_path), existing_embs)
    elif existing_texts:
        logger.info(f"Computing embeddings for {len(existing_texts)} existing questions...")
        existing_embs = compute_embeddings(existing_texts, model)
        np.save(str(emb_cache_path), existing_embs)
    else:
        existing_embs = np.zeros((0, 384))

    # Compute new question embeddings
    new_texts = [q.get("self_contained_question", "") for q in new_questions]
    logger.info(f"Computing embeddings for {len(new_texts)} new questions...")
    new_embs = compute_embeddings(new_texts, model)

    # Also deduplicate within the new batch itself
    kept_indices: list[int] = []
    kept_embs_list: list[np.ndarray] = []

    # Combined reference embeddings: existing + already-kept new ones
    if existing_embs.shape[0] > 0:
        ref_embs = existing_embs.copy()
    else:
        ref_embs = np.zeros((0, 384))

    duplicates = 0
    for i in range(len(new_questions)):
        emb_i = new_embs[i : i + 1]  # (1, dim)

        is_dup = False
        if ref_embs.shape[0] > 0:
            # Cosine similarity (embeddings are already normalized)
            sims = emb_i @ ref_embs.T  # (1, n_ref)
            max_sim = float(sims.max())
            if max_sim >= sim_threshold:
                is_dup = True
                duplicates += 1

        if not is_dup:
            kept_indices.append(i)
            kept_embs_list.append(new_embs[i])
            # Add to reference set for intra-batch dedup
            ref_embs = np.vstack([ref_embs, new_embs[i : i + 1]])

    deduped = [new_questions[i] for i in kept_indices]
    logger.info(
        f"Dedup: {len(new_questions)} -> {len(deduped)} "
        f"({duplicates} duplicates removed, threshold={sim_threshold})"
    )

    with open(deduped_path, "w") as f:
        for q in deduped:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    state.update("total_after_dedup", len(deduped))
    state.mark_step_done("dedup")
    return deduped


# ---------------------------------------------------------------------------
# Step 5: Refinement via Claude CLI Haiku
# ---------------------------------------------------------------------------
def refine_question_batch(questions: list[dict]) -> list[dict]:
    """Refine a batch of questions using Claude CLI."""
    prompt_parts = ["Refine these medical questions:\n"]
    for i, doc in enumerate(questions):
        prompt_parts.append(f"--- Question {i} ---")
        prompt_parts.append(f"source_id: {doc.get('source_id', '')}")
        prompt_parts.append(f"question: {doc.get('self_contained_question', '')}")
        prompt_parts.append(f"type: {doc.get('question_type', '')}")
        prompt_parts.append(f"domain: {doc.get('clinical_domain', '')}")
        prompt_parts.append(f"why_open: {doc.get('why_open', '')}")
        prompt_parts.append("")

    prompt = "\n".join(prompt_parts)

    try:
        result = subprocess.run(
            ["claude", "--print", "--model", "haiku", "--system-prompt", REFINE_SYSTEM_PROMPT],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=300,
        )
        output = result.stdout.strip()
    except subprocess.TimeoutExpired:
        logger.warning("Claude CLI timed out during refinement")
        return questions  # Return unrefined
    except Exception as e:
        logger.warning(f"Claude CLI error during refinement: {e}")
        return questions

    refinements = []
    for line in output.split("\n"):
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        try:
            r = json.loads(line)
            if isinstance(r, dict) and "taxonomy_l1" in r:
                refinements.append(r)
        except json.JSONDecodeError:
            continue

    # Merge refinements back into questions
    ref_by_idx = {r.get("idx", -1): r for r in refinements}
    merged = []
    for i, doc in enumerate(questions):
        ref = ref_by_idx.get(i, {})
        doc["taxonomy_l1"] = ref.get("taxonomy_l1", "")
        doc["taxonomy_l2"] = ref.get("taxonomy_l2", "")
        doc["taxonomy_l3"] = ref.get("taxonomy_l3", "")
        doc["open_status"] = ref.get("open_status", "unknown")
        doc["status_reasoning"] = ref.get("status_reasoning", "")
        doc["verification_venues"] = ref.get("verification_venues", [])
        doc["relevant_mcp_tools"] = ref.get("relevant_mcp_tools", [])
        doc["difficulty_clinical_knowledge"] = ref.get("difficulty_clinical_knowledge", doc.get("difficulty", 3))
        doc["difficulty_research_depth"] = ref.get("difficulty_research_depth", doc.get("difficulty", 3))
        doc["difficulty_multi_step_reasoning"] = ref.get("difficulty_multi_step_reasoning", doc.get("difficulty", 3))
        merged.append(doc)

    return merged


def run_refinement(
    questions: list[dict], work_dir: Path, state: PipelineState, workers: int
) -> list[dict]:
    """Run refinement over all questions in batches."""
    refined_path = work_dir / "refined.jsonl"

    if state.is_step_done("refinement"):
        logger.info("Refinement already done, loading from cache")
        refined = []
        with open(refined_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    refined.append(json.loads(line))
        return refined

    if not questions:
        state.update("total_after_refine", 0)
        state.mark_step_done("refinement")
        return []

    # Refine in batches of 30 (same size as refine_batch.py uses)
    refine_batch_size = 30
    all_refined: list[dict] = []
    total_batches = (len(questions) + refine_batch_size - 1) // refine_batch_size

    for batch_idx in range(total_batches):
        batch_start = batch_idx * refine_batch_size
        batch = questions[batch_start : batch_start + refine_batch_size]

        refined_batch = refine_question_batch(batch)
        all_refined.extend(refined_batch)

        logger.info(
            f"  Refine batch {batch_idx+1}/{total_batches}: "
            f"{len(batch)} questions processed (total: {len(all_refined)})"
        )

    with open(refined_path, "w") as f:
        for q in all_refined:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    state.update("total_after_refine", len(all_refined))
    state.mark_step_done("refinement")
    logger.info(f"Refinement complete: {len(all_refined)} questions")
    return all_refined


# ---------------------------------------------------------------------------
# Step 6: Merge into main dataset
# ---------------------------------------------------------------------------
def run_merge(
    new_questions: list[dict],
    existing_path: Path,
    work_dir: Path,
    state: PipelineState,
) -> Path:
    """Merge new refined questions into the expanded dataset."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "all_questions_expanded.jsonl"

    if state.is_step_done("merge"):
        logger.info(f"Merge already done: {output_path}")
        return output_path

    # Load existing corpus
    existing = load_existing_questions(existing_path)
    logger.info(f"Existing corpus: {len(existing)} questions")

    # Combine
    combined = existing + new_questions
    logger.info(f"Combined: {len(existing)} + {len(new_questions)} = {len(combined)} questions")

    # Write combined output
    with open(output_path, "w") as f:
        for q in combined:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    # Also write just the new questions separately
    new_only_path = OUTPUT_DIR / "new_questions_only.jsonl"
    with open(new_only_path, "w") as f:
        for q in new_questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    state.update("total_merged", len(combined))
    state.mark_step_done("merge")

    logger.info(f"Merged dataset written to {output_path}")
    logger.info(f"New questions only written to {new_only_path}")
    return output_path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_pipeline(
    input_path: Path,
    existing_path: Path,
    workers: int,
):
    """Run the full incremental pipeline."""
    # Work directory for intermediate files
    work_dir = OUTPUT_DIR / "_pipeline_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    state_path = work_dir / "pipeline_state.json"
    state = PipelineState(state_path)

    logger.info("=" * 70)
    logger.info("Incremental Pipeline")
    logger.info(f"  Input:    {input_path}")
    logger.info(f"  Existing: {existing_path}")
    logger.info(f"  Output:   {OUTPUT_DIR}")
    logger.info(f"  Workers:  {workers}")
    logger.info(f"  State:    {state_path}")
    if state.state["steps_completed"]:
        logger.info(f"  Resuming: steps done = {state.state['steps_completed']}")
    logger.info("=" * 70)

    # Step 1: Load documents
    logger.info("\n--- Step 1: Load input documents ---")
    docs = load_documents(input_path)
    if not docs:
        logger.error("No documents found in input file")
        return

    # Step 2: Extract questions
    logger.info("\n--- Step 2: Extract open questions ---")
    extracted = run_extraction(docs, work_dir, state, workers)
    if not extracted:
        logger.warning("No questions extracted, stopping pipeline")
        return
    logger.info(f"  -> {len(extracted)} questions extracted")

    # Step 3: Quality filter
    logger.info("\n--- Step 3: Quality filtering ---")
    filtered = run_quality_filter(extracted, work_dir, state)
    logger.info(f"  -> {len(filtered)} questions after filtering")

    # Step 4: Dedup
    logger.info("\n--- Step 4: Incremental dedup ---")
    deduped = run_dedup(filtered, existing_path, work_dir, state)
    logger.info(f"  -> {len(deduped)} questions after dedup")

    # Step 5: Refinement
    logger.info("\n--- Step 5: Refinement ---")
    refined = run_refinement(deduped, work_dir, state, workers)
    logger.info(f"  -> {len(refined)} questions refined")

    # Step 6: Merge
    logger.info("\n--- Step 6: Merge into dataset ---")
    output_path = run_merge(refined, existing_path, work_dir, state)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("Pipeline Complete")
    logger.info(f"  Input documents:     {len(docs)}")
    logger.info(f"  Questions extracted:  {len(extracted)}")
    logger.info(f"  After quality filter: {len(filtered)}")
    logger.info(f"  After dedup:          {len(deduped)}")
    logger.info(f"  After refinement:     {len(refined)}")
    logger.info(f"  Final merged corpus:  {state.state.get('total_merged', '?')}")
    logger.info(f"  Output:               {output_path}")
    logger.info("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="End-to-end incremental pipeline: raw docs -> expanded question dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input JSONL file with raw documents",
    )
    parser.add_argument(
        "--existing",
        type=Path,
        default=DEFAULT_EXISTING,
        help=f"Path to existing question corpus (default: {DEFAULT_EXISTING})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of parallel workers for Claude CLI calls (default: 10)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    if not args.existing.exists():
        logger.warning(f"Existing corpus not found: {args.existing} (will proceed without dedup baseline)")

    run_pipeline(
        input_path=args.input,
        existing_path=args.existing,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
