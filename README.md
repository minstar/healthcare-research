# Healthcare Research: Open Medical Questions Benchmark (난제)

A pipeline for collecting, curating, and benchmarking **unsolved** medical / biomedical /
clinical research questions ("난제") from authoritative literature, and evaluating LLMs on
them with **agentic tool use** and **per-question checklist rubrics**.

The benchmark is designed around two ideas the early version lacked:

1. **Retrieval-grounded openness.** A question's `open_status` is judged from real
   follow-up evidence (citing papers, trial results), not from a model's memory of the
   source's framing.
2. **Empirical difficulty.** Difficulty is not self-rated. Each question is run through
   **three strong models** with full tool access; the pass/fail pattern defines the
   difficulty bucket. Self-rated difficulty was measured to be uncorrelated with model
   performance and is deprecated.

## Pipeline

```
crawl ─▶ extract ─▶ refine ─▶ dedup ─▶ export        (build the question corpus)
                                  │
        status-verify ◀───────────┤                  (retrieval-grounded open/answered)
        contamination-audit ◀──────┤                  (LLM-free screen vs real evidence)
                                  │
   gen-rubrics ─▶ agentic-eval ─▶ checklist-judge ─▶ difficulty-buckets   (empirical labels)
```

1. **Crawl** — harvest documents from authoritative sources (see *Tracks* below).
2. **Extract** — Claude-CLI extraction of open research questions from each document
   (one document can raise several distinct questions).
3. **Refine** — taxonomy (12 L1 categories), 3-axis difficulty hints, MCP-tool mapping,
   `open_status` + reasoning.
4. **Dedup** — exact-question-text dedup (a paper's multiple questions are *kept*; only
   identical text collapses).
5. **Status verification** — `pipeline/status_verifier.py` gathers real follow-up evidence;
   `scripts/stage2_judge.py` re-judges `open_status` constrained to that evidence.
6. **Contamination audit** — `scripts/audit_contamination.py` (no LLM) flags synthetic
   templates, completed/dead trials, and items with no follow-up.
7. **Rubric generation** — `scripts/gen_rubrics.py` writes a frozen per-question checklist
   (`must_mention / must_acknowledge / must_ground / must_avoid`, 5-8 criteria).
8. **Agentic eval** — `harness/run.py` runs a model against each question with real MCP
   tools (multi-round tool use), producing answer traces.
9. **Checklist judging** — `scripts/checklist_judge.py` grades each answer against its
   frozen rubric (met / partial / not_met → weighted score).
10. **Difficulty buckets** — `scripts/compute_buckets.py` turns the 3-model scores into
    difficulty labels.

## Corpus versions

The corpus is built additively; each version layers empirical labels and new tracks onto
the text-deduped v3 base.

| Version | Rows | What it adds |
|---------|------|--------------|
| `mcp_benchmark_v3` | 12,553 | Clean base: `retrieval_verified` (6,648) + `expert_consensus` (5,905, JLA/NICE) |
| `mcp_benchmark_v3.2` | 12,553 | 3-model empirical labels on the gold-bearing subset; all-pass audit; still-open audit |
| `mcp_benchmark_v3.3` | 13,078 | + **priority_setting** track (525 questions) with empirical labels |
| `mcp_benchmark_v3.4` | 13,561 | + **expand** track (483); **question-granular** relabel of the whole corpus *(being finalized)* |

`mcp_benchmark_with_gold.jsonl` (1,969) is the gold-answer-bearing slice used for rubric
generation and agentic eval.

### Tracks (`corpus_track`)

| Track | Source | Openness grounding |
|-------|--------|--------------------|
| `retrieval_verified` | PubMed / trials / arXiv | Retrieval-based status (citing papers, trial results) |
| `expert_consensus` | JLA Priority Setting Partnerships, NICE research recs | Open by expert/consensus declaration |
| `priority_setting` | Society agendas, WHO/CHNRI/NASEM/PCORI, Delphi/consensus "research priorities" (Europe PMC) | Authoritative priority-setting documents |
| `expand_priority_lit` | Additional priority-setting + Cochrane research-gap literature | Same as above |

> **Note on preprints:** medRxiv abstracts were trialed as a source and **rejected** — they
> report results rather than frame open questions, so extraction yields ~nothing. Crawling
> is focused on priority-setting / research-gap documents.

## Empirical difficulty (3-model buckets)

Each question is answered, with tools, by **GLM-5.1**, **Qwen3.6**, and **DeepSeek-V4-Flash**,
then graded by the GLM-5.1 checklist judge. A model "fails" a question if its checklist
score `< 0.5`. The pass/fail pattern across the three models gives:

| Bucket | `empirical_difficulty` | Meaning |
|--------|------------------------|---------|
| all-3-fail | `core_nanje` (`nanje_core=true`) | Genuine open problem — no strong model can answer it |
| split | `discriminating` | Discriminating question |
| all-3-pass | `easy` | Not a real 난제 — quarantined unless a per-item audit keeps it |

Observed distribution is stable across scales (~50 % core / ~45 % split / ~5 % easy at the
1969, 525, and 300-item scales). On the finalized core set, all three strong models score
~0.27-0.30 average with **≈0 % pass@0.5** — i.e. the core set is genuinely hard
(`results/core_nanje_baseline.json`).

All-3-pass questions are individually audited (`scripts/classify_allpass.py`) into
`vague_aspirational / already_resolved / broad_goal / genuine_hard`; only `genuine_hard` is
kept, the rest are `quarantine=true`.

### Question granularity (`task_id`)

`task_id` is **unique per question** (`{source_id}#{k}`). This matters because extraction
produces several questions per source paper; keying eval/rubrics/judging on `source_id`
alone would silently collapse them to source-paper granularity. v3.4 re-keys the whole
corpus (including the 1969 gold slice) to question granularity.

## Status verification & contamination audit

The original refiner assigned `open_status` from a single tool-less LLM call (the corpus had
**zero** `answered`/`unknown` items — confirmation bias toward the source). The verification
stage replaces this with retrieval-grounded evidence.

### Stage 1 — evidence gathering (`pipeline/status_verifier.py`, no LLM)

| Source | Evidence | Resolution signal |
|--------|----------|-------------------|
| PubMed (PMID) | Europe PMC citations → citing papers + abstracts | Later papers resolving the question |
| Trial (NCT) | ClinicalTrials.gov v2 status / completion / results | COMPLETED + results ⇒ likely no longer fully open |
| arXiv | Semantic Scholar citations | Follow-up work |
| KEGG / UniProt | none | Flagged `synthetic` |

> NCBI `elink` citedin proved unreliable; Europe PMC's citation index is the primary
> follow-up source for PMIDs.

### Stage 2 — grounded judgment (`scripts/stage2_judge.py`, Claude CLI)

The LLM judges `open_status` **only** from the gathered evidence and must cite evidence IDs
that exist in the bundle; hallucinated citations are rejected and the item falls back to
`unknown`. On the still-open audit of the 657 core_nanje (the trial-completed / no-followup
flagged subset), **0 were confirmed-resolved** — the core set survives grounded re-judgment.

### Contamination audit (`scripts/audit_contamination.py`, no LLM)

```bash
python scripts/audit_contamination.py --data data/export/expand_questions.jsonl \
  --out results/audit_expand --workers 12
```

Reports `hard_contamination` (synthetic templates / dead trials) and flags
`trial_completed`, `heavy_followup_recheck`, `no_followup_evidence`. The core and expand
sets audit at **0 % hard contamination**.

## Benchmark harness

Evaluate any LLM's ability to answer open medical questions using real biomedical APIs.

### MCP tools (10 medical APIs, direct REST — no Docker)

`pubmed` · `clinicaltrialsgov` · `openfda` · `opentargets` · `chembl` · `uniprot` ·
`pubchem` · `kegg` · `ncbi_datasets` · `biomcp`

### Scoring — per-question checklist rubric (current method)

The free-form 5-dimension judge (`harness/judge.py`) was found to have high variance on open
questions (the same answer scored 0.50 by one judge and 0.30 by another). It is superseded by
**checklist judging**: a frozen per-question rubric is graded criterion-by-criterion, which
raised judge agreement from Spearman ~0.35 to ~0.82.

```
score = Σ(weight · v) / Σ(weight),   v ∈ {met:1.0, partial:0.5, not_met:0.0}
```

### Usage

```bash
# 1. Generate frozen rubrics (once per dataset)
python scripts/gen_rubrics.py --data data/export/mcp_benchmark_with_gold.jsonl \
  --model glm-5.1 --base http://<glm-node>:8000/v1 --out data/eval_samples/rubrics_1969_uid.jsonl

# 2. Agentic completion (real tool use; resumable checkpoint to <output>/traces.jsonl)
OPENAI_API_BASE=http://<node>:8000/v1 python harness/run.py \
  --data data/export/mcp_benchmark_with_gold.jsonl \
  --model "openai::http://<node>:8000/v1::my-model" --no-judge \
  --workers 12 --output results/my_model_run

# 3. Checklist judge (resumable; appends + fsync per item)
python scripts/checklist_judge.py --traces results/my_model_run/traces.jsonl \
  --rubrics data/eval_samples/rubrics_1969_uid.jsonl \
  --judge openai/glm-5.1 --base http://<glm-node>:8000/v1 \
  --out results/my_model_run/checklist_glm.jsonl

# Model spec format: backend::base_url::model_name  (backends: openai, litellm, claude)
```

## Serving & orchestration

The three judge/eval models run as vLLM OpenAI-compatible endpoints on **B200 (Blackwell)**:

| Model | Script | Notes |
|-------|--------|-------|
| GLM-5.1-FP8 | `serving/serve_glm51.slurm` | TP8; arch needs source-built DeepGEMM in the `kimi` env |
| Qwen3.6 | `serving/serve_qwen36.slurm` | `qwen3_xml` tool parser |
| DeepSeek-V4-Flash | `settings/serving/deepseek_v4_flash_vllm.sh` | docker image `vllm-openai:deepseekv4-cu130`, DP4 |

**Preemptible orchestration.** Serving runs on the `preemptible` partition and is scancelled
when done. Because an 8-GPU job can be preempted mid-run, every long step is **resumable**
(append + `os.fsync`, skip already-done `task_id`s): `gen_rubrics.py`, `checklist_judge.py`,
and the harness completion loop (`harness/completion_runner.py`, checkpoint = `traces.jsonl`).
`scripts/run_nextbatch.sh` wraps this in an outer loop that re-establishes endpoints and
re-runs resumable steps until both tracks complete — so a preemption is just another
iteration. Submit-once-and-wait is preferred over cancel+resubmit (which resets queue
priority).

## Project structure

```
├── crawlers/                       # Document harvesters
│   ├── priority_sources_crawler.py #   Authoritative priority-setting (Europe PMC)
│   ├── jla_crawler.py / nice_crawler.py   # Expert-consensus open questions
│   ├── cochrane_crawler.py         #   Systematic-review research gaps
│   ├── pubmed_crawler.py / arxiv_crawler.py / medrxiv_crawler.py / nature_crawler.py
│   └── biomedical_api_crawler.py   #   OpenTargets, ChEMBL, UniProt, …
│
├── pipeline/
│   ├── extractor.py / refiner.py / dedup.py / taxonomy.py
│   └── status_verifier.py          #   Retrieval-grounded evidence (Europe PMC / CT.gov / S2)
│
├── scripts/
│   ├── extract_batch.py / refine_batch.py        # Claude-CLI batch extract & refine
│   ├── export_priority.py / export_expand.py     # Track export (dedup vs existing)
│   ├── gen_rubrics.py                            # Per-question checklist rubrics (resumable)
│   ├── checklist_judge.py                        # Rubric-based judge (resumable)
│   ├── compute_buckets.py                        # 3-model difficulty buckets
│   ├── classify_allpass.py                       # Audit all-3-pass questions
│   ├── stage2_judge.py                           # Grounded open_status re-judgment
│   ├── audit_contamination.py                    # LLM-free contamination screen
│   ├── apply_still_open_flags.py                 # Attach still-open audit flags
│   ├── build_v3_3.py / build_v3_4.py             # Version assembly (question-granular)
│   ├── run_nextbatch.sh / wait_and_judge_priority.sh  # Preempt-tolerant orchestrators
│   └── track_b/                                  # Gold answer generation + validation
│
├── harness/                        # Standalone benchmark eval
│   ├── task_loader.py / mcp_tools.py / completion_runner.py
│   ├── judge.py                    #   (legacy 5-dim judge; superseded by checklist_judge)
│   ├── metrics.py / run.py
│
├── data/
│   ├── raw/                        # Crawled documents (per-track subdirs)
│   ├── extracted/ , refined/       # Pipeline intermediates
│   └── export/                     # Benchmark files (mcp_benchmark_v3*.jsonl, *_questions.jsonl)
│
├── serving/                        # vLLM serve scripts + job-id / node state
└── results/                        # Eval traces, checklist scores, audits, run logs
```

## Question record (key fields)

```json
{
  "task_id": "PMID:38345416#0",
  "source_id": "PMID:38345416",
  "self_contained_question": "…",
  "corpus_track": "retrieval_verified",
  "taxonomy_l1": "Neuroscience & Psychiatry",
  "open_status": "open",
  "empirical_difficulty": "core_nanje",
  "nanje_core": true,
  "model_scores_glmjudge": {"glm": 0.30, "qwen": 0.29, "deepseek_v4": 0.27},
  "n_models_fail": 3,
  "difficulty_source": "3model_empirical_1969_uid",
  "still_open_flag": "confirmed_open",
  "quarantine": false
}
```

## Gold answer schema

```json
{
  "current_knowledge": "what IS currently known",
  "unknown_aspects": "what remains unknown / debated",
  "evidence_landscape": "evidence quality (RCTs, preclinical, …)",
  "key_citations": [{"type": "PMID", "id": "12345678", "relevance": "…"}],
  "mcp_tool_plan": [{"tool": "pubmed", "query": "…", "purpose": "…"}],
  "answer_summary": "synthesis for a researcher",
  "self_completeness": 0.45
}
```

> `self_completeness` is the model's self-assessed answerability of the question — useful for
> relative comparison, not as an absolute measure, and distinct from the empirical difficulty.

> Gold PMIDs are verified to **exist** (NCBI) but were measured to often point to the wrong
> paper (~74 % mismatch, LLM-judge n=360). Gold citations are therefore **not** used as
> ground truth; scoring relies on the per-question checklist rubric.

## Taxonomy (12 L1 categories)

Clinical Medicine · Oncology · Neuroscience & Psychiatry · Infectious Disease & Immunology ·
Cardiovascular Medicine · Genomics & Precision Medicine · Pharmacology & Drug Discovery ·
Public Health & Epidemiology · Rare & Orphan Diseases · Surgical Sciences ·
Medical AI & Informatics · Other
