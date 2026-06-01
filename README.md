# Healthcare Research: Open Medical Questions Benchmark

A pipeline for systematically collecting, curating, and benchmarking unsolved medical/biomedical/clinical questions from the scientific literature.

## Overview

1. **Crawl** — Harvest documents from PubMed, MedRxiv, arXiv, Cochrane, Nature, OpenFDA, and biomedical APIs
2. **Extract** — LLM-based extraction of open research questions from harvested documents
3. **Filter** — 15-rule quality filter removing garbled text, templates, non-questions, and answered items
4. **Refine** — Taxonomy classification, difficulty scoring (3-axis), MCP tool mapping
5. **Status Verification** — *Retrieval-grounded* open/answered judgment: gather real follow-up evidence (citing papers, trial results) per source, then judge status from that evidence (not model memory)
6. **Contamination Audit** — Full-corpus scan that flags synthetic templates, dead/completed trials, and mislabeled items against real evidence
7. **Gold Answers** — Reference answer generation with citation verification
8. **Benchmark Harness** — Standalone evaluation framework with 10 medical MCP tool wrappers and LLM-as-judge

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total curated questions | 1,969 |
| Taxonomy L1 categories | 12 |
| Unique clinical domains | 200+ |
| Difficulty distribution | 96% at level 3-5 |
| Gold answer coverage | 150 (expanding) |
| PMID existence (gold citations) | 100% real IDs / 0.0% invalid (NCBI) |
| PMID correctness (gold citations) | ⚠️ ~74% point to the WRONG paper (LLM-judge, n=360; 12/12 sampled mismatches verified) — existence ≠ correctness; gold PMIDs are NOT used as ground truth |

## Project Structure

```
├── crawlers/                  # Document harvesters
│   ├── pubmed_crawler.py      #   PubMed (NCBI E-utils)
│   ├── medrxiv_crawler.py     #   MedRxiv preprints
│   ├── arxiv_crawler.py       #   arXiv biomedical
│   ├── cochrane_crawler.py    #   Cochrane systematic reviews
│   ├── nature_crawler.py      #   Nature journals
│   ├── biomedical_api_crawler.py  # OpenTargets, ChEMBL, UniProt, etc.
│   └── web_crawler.py         #   General web sources
│
├── pipeline/                  # Core processing
│   ├── extractor.py           #   LLM question extraction
│   ├── dedup.py               #   Embedding-based deduplication (MiniLM, cosine ≥ 0.90)
│   ├── refiner.py             #   Taxonomy & metadata refinement
│   ├── status_verifier.py     #   Retrieval-grounded evidence gathering (Europe PMC / CT.gov / S2)
│   └── taxonomy.py            #   12-category taxonomy definitions
│
├── scripts/
│   ├── filter_quality.py      # 15-rule quality filter
│   ├── audit_contamination.py # Full-corpus retrieval-grounded contamination audit (no LLM)
│   ├── cleanup_v2.py          # Deterministic cleanup: normalize labels, quarantine, relabel
│   ├── refine_batch.py        # Batch refinement via Claude CLI
│   ├── track_a/               # Data expansion
│   │   ├── pubmed_mesh_expansion.py   # 112 MeSH terms × 3 query templates
│   │   └── incremental_pipeline.py    # Extract → filter → dedup → refine → merge
│   └── track_b/               # Gold answer generation
│       ├── prepare_gold_batches.py    # Split into batches of 3
│       ├── generate_gold_answers.py   # Claude CLI batch generation (Opus)
│       ├── merge_gold_answers.py      # Merge + auto-validation
│       └── validate_gold_answers.py   # PMID/NCT verification + completeness criteria
│
├── harness/                   # Standalone benchmark evaluation
│   ├── task_loader.py         #   JSONL loader with taxonomy/difficulty/tool filters
│   ├── mcp_tools.py           #   10 medical API wrappers (direct REST, no Docker)
│   ├── completion_runner.py   #   Multi-turn tool-use runner (OpenAI/LiteLLM/Claude)
│   ├── judge.py               #   5-dimension LLM-as-judge scoring
│   ├── metrics.py             #   Aggregation by taxonomy, difficulty, tool usage
│   └── run.py                 #   CLI entry point
│
├── data/
│   ├── raw/                   # Crawled documents
│   ├── extracted/             # Extracted questions (pre-filter)
│   ├── refined/               # Refined batches with taxonomy
│   ├── gold_batches/          # Input batches for gold answer generation
│   ├── gold_answers/          # Generated gold answers with citations
│   └── export/                # Final benchmark files
│       ├── mcp_benchmark.jsonl
│       ├── mcp_benchmark_with_gold.jsonl
│       └── validation_report.json
│
└── run.py                     # Main CLI (crawl, extract, filter, refine)
```

## Gold Answer Schema

Each question has a reference answer with the following structure:

```json
{
  "current_knowledge": "2-3 paragraphs on what IS currently known",
  "unknown_aspects": "1-2 paragraphs on what remains unknown or debated",
  "evidence_landscape": "Brief description of evidence quality (RCTs, preclinical, etc.)",
  "key_citations": [
    {"type": "PMID", "id": "12345678", "relevance": "one sentence"}
  ],
  "mcp_tool_plan": [
    {"tool": "pubmed", "query": "search query", "purpose": "what this retrieves"}
  ],
  "answer_summary": "2-4 paragraph synthesis for a researcher",
  "self_completeness": 0.45
}
```

> **Note**: `self_completeness` is the model's self-assessed epistemic difficulty score — how completely the question can be answered with current evidence. This is distinct from the judge's `completeness` dimension (which evaluates a model's response thoroughness). Values are useful for relative comparison between questions, not as absolute measures.

## Gold Answer Validation

Post-generation validation runs automatically via `validate_gold_answers.py`:

### Structural Checks
- **PMID verification**: Batch lookup against NCBI efetch (existence check)
- **NCT verification**: ClinicalTrials.gov API check
- **Completeness criteria**:
  - `current_knowledge` >= 200 chars
  - `unknown_aspects` >= 100 chars
  - `answer_summary` >= 200 chars
  - `key_citations` >= 2 entries
  - `mcp_tool_plan` >= 1 entry
  - `self_completeness` in [0.0, 1.0]

### Semantic Checks
- **Type-Token Ratio (TTR)**: Flags answers with TTR < 0.3 (highly repetitive text)
- **Question-Answer Entity Overlap**: Flags answers where < 20% of question entities appear in the answer
- **Citation-Content Alignment**: Flags answers where < 50% of citations have keyword overlap with the answer text

### Citation Relevance Verification (`--check-relevance`)
- Fetches paper title/abstract from NCBI for each PMID
- Computes Jaccard keyword similarity between citation `relevance` text and actual paper content
- Flags citations with score < 0.1 as "weak citations"

```bash
# Standalone validation
python scripts/track_b/validate_gold_answers.py --report data/export/validation_report.json

# With auto-fix (removes invalid citations)
python scripts/track_b/validate_gold_answers.py --fix

# Merge triggers validation automatically
python scripts/track_b/merge_gold_answers.py
python scripts/track_b/merge_gold_answers.py --fix-citations   # merge + fix
python scripts/track_b/merge_gold_answers.py --skip-validation  # merge only
```

## Status Verification & Contamination Audit

The original refiner assigned `open_status` from a single tool-less LLM call, so labels were
ungrounded (the corpus had **zero** `answered`/`unknown` items — confirmation bias toward the
source's framing). The verification stage replaces this with retrieval-grounded evidence,
mirroring ResearchMath-14K's refiner (which reads up to 10 citing papers to determine status).

### Stage 1 — Evidence gathering (`pipeline/status_verifier.py`, no LLM)

Dispatched by source type; all calls are free REST endpoints:

| Source | Evidence | Resolution signal |
|--------|----------|-------------------|
| PubMed (PMID) | Europe PMC citations API → citing papers + abstracts (NCBI elink as fallback) | Later papers that resolve the question |
| Trial (NCT) | ClinicalTrials.gov v2 → `OverallStatus`, completion date, results posted | COMPLETED + results ⇒ no longer open |
| arXiv | Semantic Scholar citations | Follow-up work |
| KEGG/UniProt | none | Flagged `synthetic` (templated, not literature-extracted) |

> NCBI `elink` citedin proved unreliable (intermittent empty/500 responses); Europe PMC's
> citation index is used as the primary follow-up source for PMIDs.

### Stage 2 — Grounded judgment (LLM, evidence-constrained)

The LLM judges `open_status` **only** from the gathered evidence and must cite evidence IDs
that exist in the bundle (`status_evidence_ids ⊆ retrieved IDs`); hallucinated citations are
rejected and the item falls back to `unknown`. `answered`/`unknown` are now reachable.

### Contamination Audit (`scripts/audit_contamination.py`)

Runs the fast, LLM-free `screen()` over the full corpus and categorizes each item:

```bash
python scripts/audit_contamination.py \
  --data data/expanded/all_questions_combined.jsonl \
  --out data/audit_full --workers 12
```

Flags: `trial_resolved_but_open`, `trial_dead` (terminated/withdrawn), `trial_completed`,
`synthetic_template`, `heavy_followup_recheck` (re-judge candidate), `no_followup_evidence`.

### Deterministic Cleanup → v2 (`scripts/cleanup_v2.py`)

Applies auditable, non-judgment fixes (nothing deleted in place — removals are quarantined):

1. **Label normalization** — 11 observed `open_status` strings (`partially_resolved`,
   `mostly_resolved`, `partially resolved`, `closed`, …) → 4 canonical
   (`open`, `partially_answered`, `answered`, `unknown`)
2. **Quarantine** — synthetic templates + dead trials → `data/export/quarantine_v2.jsonl`
3. **Relabel** — completed-with-results trials labeled `open` → `partially_answered`
4. **Provenance** — attach `status_method`, `audit_flag`, `n_followups`, `src_year`

```bash
python scripts/cleanup_v2.py \
  --data data/expanded/all_questions_combined.jsonl \
  --audit data/audit_full/audit_per_item.jsonl \
  --out-dir data/export
# → data/export/mcp_benchmark_v2.jsonl + quarantine_v2.jsonl + cleanup_v2_report.json
```

## Benchmark Harness

Evaluate any LLM's ability to answer open medical questions using real biomedical APIs.

### MCP Tools (10 medical APIs)

| Tool | API | Description |
|------|-----|-------------|
| pubmed | NCBI E-utils | Literature search |
| clinicaltrialsgov | ClinicalTrials.gov v2 | Trial search |
| openfda | OpenFDA | Drug adverse events |
| opentargets | Open Targets GraphQL | Drug-target associations |
| chembl | ChEMBL REST | Compound/target data |
| uniprot | UniProt REST | Protein information |
| pubchem | PubChem PUG REST | Chemical properties |
| kegg | KEGG REST | Pathway data |
| ncbi_datasets | NCBI Datasets v2 | Gene/genome data |
| biomcp | BioMCP composite | Multi-source biomedical |

### Judge Dimensions

| Dimension | Weight | Description |
|-----------|--------|-------------|
| Coverage | 0.25 | How much of the gold answer is addressed |
| Evidence Quality | 0.20 | Citation accuracy and evidence level |
| Tool Usage | 0.25 | Appropriate tool selection and query quality |
| Reasoning | 0.15 | Logical consistency, uncertainty acknowledgment |
| Completeness | 0.15 | Overall answer thoroughness |

Pass threshold: `main_score >= 0.60`

### Usage

```bash
python harness/run.py \
  --data data/export/mcp_benchmark_with_gold.jsonl \
  --model "openai::http://localhost:8000/v1::my-model" \
  --judge "gemini/gemini-2.5-pro" \
  --taxonomy-filter "Oncology" \
  --difficulty-min 3 \
  --limit 50 \
  --output results/

# Model spec format: backend::base_url::model_name
# Backends: openai, litellm, claude

# Multi-judge ensemble (comma-separated) — computes median scores + Cohen's kappa
python harness/run.py \
  --data data/export/mcp_benchmark_with_gold.jsonl \
  --model "openai::http://localhost:8000/v1::my-model" \
  --judge "gemini/gemini-2.5-pro,gpt-4o,claude-opus-4-20250514" \
  --output results/
```

## Data Expansion

```bash
# PubMed MeSH expansion (112 disease terms × 3 templates)
python scripts/track_a/pubmed_mesh_expansion.py --max-per-query 50

# Incremental pipeline (new docs → extract → filter → dedup → refine → merge)
python scripts/track_a/incremental_pipeline.py \
  --input data/raw/pubmed_mesh/documents.jsonl \
  --existing data/extracted/all_questions_refined.jsonl \
  --workers 10
```

## Taxonomy (12 L1 Categories)

Clinical Medicine, Oncology, Neuroscience & Psychiatry, Infectious Disease & Immunology, Cardiovascular Medicine, Genomics & Precision Medicine, Pharmacology & Drug Discovery, Public Health & Epidemiology, Rare & Orphan Diseases, Surgical Sciences, Medical AI & Informatics, Other
