# OpenBioRQ — Open Biomedical Research Questions

A pipeline for collecting, curating, and benchmarking **open (currently-unresolved)**
biomedical / clinical research questions from authoritative literature and trial records,
and evaluating LLMs on them with **agentic tool use** and **per-question checklist rubrics**.

"Open" is a **provenance** claim — every question is sourced from a genuinely unresolved
research front (JLA/NICE priority-setting, expert consensus, or retrieval-verified open
literature). It is a set of **hard, open-ended biomedical questions**, not an AI grand
challenge: the strongest frontier agents do solve a meaningful fraction (see *Findings*).

The benchmark is designed around two ideas the early version lacked:

1. **Retrieval-grounded openness.** A question's `open_status` is judged from real
   follow-up evidence (citing papers, trial results), not from a model's memory of the
   source's framing.
2. **Empirical difficulty.** Difficulty is not self-rated. Each question is run through
   strong models with full tool access; the pass/fail pattern defines the difficulty.
   Self-rated difficulty was measured uncorrelated with model performance and is deprecated.

> **Paper status.** This repo backs the **OpenBioRQ** paper. The single source of truth for
> every number is `paper_writing/FACTS.md`; the headline figures below are transcribed from
> it. The paper is built on the **v3 / 12,553-question base** plus the **frozen core** — the
> later `v3.4` whole-corpus relabel was **not adopted** (decided 2026-06-19).

## Pipeline

```
crawl ─▶ extract ─▶ refine ─▶ dedup ─▶ export        (build the question corpus)
                                  │
        status-verify ◀───────────┤                  (retrieval-grounded open/answered)
        contamination-audit ◀──────┤                  (LLM-free screen vs real evidence)
                                  │
   gen-rubrics ─▶ agentic-eval ─▶ checklist-judge ─▶ leaderboard / buckets   (empirical eval)
```

1. **Crawl** — harvest documents from authoritative sources (see *Tracks* below).
2. **Extract** — Claude-CLI extraction of open research questions from each document
   (one document can raise several distinct questions).
3. **Refine** — taxonomy (12 L1 categories), 3-axis difficulty hints, MCP-tool mapping,
   `open_status` + reasoning. The refine pass also makes each question **self-contained**:
   LLM-judged self-containment rises **51.6% → 85.4%** (+33.8 pts, n=500 each;
   `results/self_containment_before.jsonl` → `results/self_containment.jsonl`).
4. **Dedup** — MiniLM-L6 cosine ≥ 0.90 (`pipeline/dedup.py`); a paper's multiple distinct
   questions are *kept*, only near-identical text collapses.
5. **Status verification** — `pipeline/status_verifier.py` gathers real follow-up evidence;
   `scripts/stage2_judge.py` re-judges `open_status` constrained to that evidence.
6. **Contamination audit** — `scripts/audit_contamination.py` (no LLM) flags synthetic
   templates, completed/dead trials, and items with no follow-up.
7. **Rubric generation** — `scripts/gen_rubrics.py` writes a frozen per-question checklist
   (`must_mention / must_acknowledge / must_ground / must_avoid`, 5-8 weighted criteria).
8. **Agentic eval** — `harness/run.py` runs a model against each question with real MCP
   tools (multi-round tool use), producing answer traces.
9. **Checklist judging** — `scripts/checklist_judge.py` grades each answer against its
   frozen rubric (met / partial / not_met → weighted score).
10. **Leaderboard / buckets** — `scripts/build_leaderboard_t0.py` (T=0 leaderboard) and
    `scripts/compute_buckets.py` (3-model difficulty buckets) turn scores into labels.

## Corpus versions

The corpus is built additively; each version layers empirical labels and new tracks onto
the text-deduped v3 base.

| Version | Rows | What it adds |
|---------|------|--------------|
| `mcp_benchmark_v3` | 12,553 | **Paper base.** `retrieval_verified` (6,648) + `expert_consensus` (5,905, JLA/NICE) |
| `mcp_benchmark_v3.2` | 12,553 | 3-model empirical labels on the gold-bearing subset; all-pass + still-open audits |
| `mcp_benchmark_v3.3` | 13,078 | + **priority_setting** track (525 questions) with empirical labels |
| `mcp_benchmark_v3.4` | 13,561 | + **expand** track (483); question-granular whole-corpus relabel — **NOT ADOPTED** (kept unused; paper stays on the v3 base + frozen core) |

`mcp_benchmark_with_gold.jsonl` (**1,969**) is the gold-answer-bearing slice used for rubric
generation and agentic eval. The **657-question core** (`data/eval_samples/core_eval.jsonl`)
and its **423-question frozen core** (`robust_core_eval.jsonl`) are the evaluation sets.

### Tracks (`corpus_track`)

| Track | Source | Openness grounding |
|-------|--------|--------------------|
| `retrieval_verified` | PubMed / trials / arXiv | Retrieval-based status (citing papers, trial results) |
| `expert_consensus` | JLA Priority Setting Partnerships, NICE research recs | Open by expert/consensus declaration |
| `priority_setting` | Society agendas, WHO/CHNRI/NASEM/PCORI, Delphi/consensus "research priorities" (Europe PMC) | Authoritative priority-setting documents |
| `expand_priority_lit` | Additional priority-setting + Cochrane research-gap literature | Same as above |

> **Note on preprints:** medRxiv abstracts were trialed and **rejected** — they report
> results rather than frame open questions, so extraction yields ~nothing. Crawling is
> focused on priority-setting / research-gap documents.

## The core sets — full core (657) and frozen core (423)

Difficulty is decoding-sensitive, and the core is defined in two stages:

- **3-model buckets (T=0.3).** Each gold-slice question is answered with tools by
  **GLM-5.1**, **Qwen3.6**, **DeepSeek-V4-Flash**, graded by the GLM-5.1 checklist judge
  (fail = score < 0.5). The pass/fail pattern gives `all-3-fail` (`core`, ~50%),
  `split` (`discriminating`, ~45%), `all-3-pass` (`easy`, ~5%) — stable across the 1969,
  525, and 300 scales. All-3-pass items are individually audited
  (`scripts/classify_allpass.py`); only `genuine_hard` is kept.
- **Full core (657).** The all-3-fail set carried forward as the primary eval set
  (`core_eval.jsonl`, with `gold_answer`).
- **Frozen core (423).** The earlier "pass@0.5 ≈ 0%" was a **T=0.3 artifact** — at **T=0**
  the same set is much more solvable. We therefore re-define the **frozen core = the 423/657
  questions that all three roster models still fail (<0.5) at T=0**
  (`robust_core_eval.jsonl`, ids in `results/robust_core_ids.json`).

> **Frozen-core reproducibility (honest caveat).** The frozen core is a **single-T=0
> snapshot, not a seed-stable partition.** Re-decoding shows membership churn concentrated at
> the score boundary (~46% boundary flip; 34.7% of the riskiest deep-failures also flip on
> re-decode), driven by live-API drift and erratic agentic tool use. *Comparisons are
> unaffected* — all models are scored on the **same** 423 — and the frontier gradient is
> unchanged on the churn-stable subset (346: Gemini 24.6 / Opus 32.7 / GPT-5.5 55.5 vs
> full-423 28.8 / 37.8 / 59.6). Seed-averaged core is future work.
> Sources: `results/redecode_*`, `results/gradient_resampling.json`.

## Findings (transcribed from `paper_writing/FACTS.md`)

### Leaderboard — frozen core (423), T=0, ten-tool harness, GLM-5.1 checklist judge

| Model | role | frozen-core solve@0.5 [95% CI] |
|-------|------|-------------------------------:|
| GLM-5.1 / Qwen3.6 / DeepSeek-V4 | roster | 0% (frozen = all-roster-fail by construction) |
| Qwen3-235B-A22B | held-out (older gen) | 2.1 [1.1, 4.0] |
| GLM-5 | held-out | 16.6 [13.3, 20.4] |
| Qwen3.5-397B-A17B | held-out | 16.8 [13.6, 20.7] |
| Gemini-3-Pro | frontier (Google) | 28.8 [24.7, 33.3] |
| Opus-4.7 | frontier (Anthropic) | 37.8 [33.3, 42.5] |
| **GPT-5.5** | frontier (OpenAI) | **59.6 [54.8, 64.1]** |

(Full-core (657) solve@0.5: Gemini 37.4, Opus 48.6, GPT-5.5 **66.7**, GPT-5.5-no-tool 60.8.
Sources: `results/leaderboard_core_t0.json`, `results/fullcore_combined.json`.)

- **Hard but not unsolvable, and not saturated.** A clean capability gradient
  (Gemini < Opus < GPT-5.5) — the best single agent still leaves ~40% of the frozen core
  unsolved. The earlier "~1 in 6" was a *same-lineage held-out* property, not a frontier one.
- **Single-lineage circularity resolved** (three independent lineages: Google / Anthropic /
  OpenAI), but the resolution cuts *against* an "AI grand challenge" framing.
- **Tools confer no *measurable* advantage.** No-tool GLM-5.1 full-core 30.8 [27.3,34.4] vs
  with-tool 26.6 [23.4,30.1] → CIs overlap; replicated across lineage (GPT-5.5 59.6 vs
  no-tool 55.6, CIs overlap). The claim is "no measurable advantage," not "tools don't help."

### Discriminates where MedQA saturates (`results/medqa_ortho/`)

On closed-form MedQA/PubMedQA/MedMCQA the same models compress into a **~6-pt band
(89.9–96.2)**, while OpenBioRQ spreads **0 → 59.6**. Spearman(MedQA, OpenBioRQ) = **0.14**.
Killer pair: DeepSeek-V4 and GLM-5 are within 0.2 pt on MedQA yet **4× apart** on OpenBioRQ
(6.2 vs 26.1). OpenBioRQ exposes heterogeneity that saturated MC benchmarks hide.

### Construction integrity

- **Contamination: 0% hard** on both core (657) and expand (483)
  (`scripts/audit_contamination.py`), measured at **full sensitivity** — a positive control
  recovers injected positives **34/34** with **0/12** false positives
  (`scripts/contamination_positive_control.py`).
- **No memorization signature.** Verbatim membership: token Jaccard vs source median 0.23,
  86.5% of frozen-core questions share ≤5 consecutive verbatim words with their source
  (`scripts/membership_verbatim.py`). Perplexity: frozen-core questions are *higher*
  perplexity than long-public MedQA under the same model (`scripts/membership_perplexity.py`).

### Citation factuality — existence ≠ correctness (`results/cite_audit*`)

Agent trace citations almost always **exist** (Qwen 99.6% / DeepSeek-V4 99.8%; overall
fabrication ≈ **0.7%**) — but ~**1 in 7 to 1 in 10** of the real citations are **wrong-paper**
(exists but does not support the claim): definitive **15.9%** (primary GLM judge) /
**10.6%** (independent Opus judge). Existence ≠ correctness by **15–23×**. The wrong-paper
verdict survives an independent different-family judge (cross-family Cohen κ = **0.755**).

## Status verification & contamination audit

The original refiner assigned `open_status` from a single tool-less LLM call (the corpus had
**zero** `answered`/`unknown` items — confirmation bias toward the source). The verification
stage replaces this with retrieval-grounded evidence (status ablation on n=200: 56.5% flip
rate, `unknown` reachable 14%, was 0% under source-framing).

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

The LLM judges `open_status` **only** from gathered evidence and must cite evidence IDs that
exist in the bundle; hallucinated citations are rejected and the item falls back to
`unknown`. On the still-open audit of the 657 core (trial-completed / no-followup flagged
subset), **0 were confirmed-resolved** — the core survives grounded re-judgment.

### Contamination audit (`scripts/audit_contamination.py`, no LLM)

```bash
python scripts/audit_contamination.py --data data/export/expand_questions.jsonl \
  --out results/audit_expand --workers 12
```

Reports `hard_contamination` (synthetic templates / dead trials) and flags
`trial_completed`, `heavy_followup_recheck`, `no_followup_evidence`. Core and expand sets
audit at **0% hard contamination** (positive control: recall 34/34, FP 0/12).

## Benchmark harness

Evaluate any LLM's ability to answer open medical questions using real biomedical APIs.

### MCP tools (10 medical APIs, direct REST — no Docker)

`pubmed` · `clinicaltrialsgov` · `openfda` · `opentargets` · `chembl` · `uniprot` ·
`pubchem` · `kegg` · `ncbi_datasets` · `biomcp`

### Scoring — per-question checklist rubric (current method)

The free-form 5-dimension judge (`harness/judge.py`) had high variance on open questions
(the same answer scored 0.50 by one judge and 0.30 by another). It is superseded by
**checklist judging**: a frozen per-question rubric is graded criterion-by-criterion, raising
inter-judge agreement from Spearman ~0.35 to ~0.82 (cross-family wrong-paper κ = 0.755;
a non-expert human spot-check, n=50, gives κ 0.29 vs GLM / 0.51 vs Opus — domain-expert
validation is deferred).

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
  --data data/eval_samples/core_eval.jsonl \
  --model "openai::http://<node>:8000/v1::my-model" --no-judge \
  --workers 12 --output results/my_model_run

# 3. Checklist judge (resumable; appends + fsync per item)
python scripts/checklist_judge.py --traces results/my_model_run/traces.jsonl \
  --rubrics data/eval_samples/rubrics_1969_uid.jsonl \
  --judge openai/glm-5.1 --base http://<glm-node>:8000/v1 \
  --out results/my_model_run/checklist_glm.jsonl

# 3b. Or judge via OpenRouter (no local GPU): scripts/checklist_judge_or.py
# 4. T=0 leaderboard from a set of checklist files: scripts/build_leaderboard_t0.py

# Model spec format: backend::base_url::model_name  (backends: openai, litellm, claude)
```

## Serving & orchestration

The roster judge/eval models run as vLLM OpenAI-compatible endpoints on **B200 (Blackwell)**:

| Model | Script | Notes |
|-------|--------|-------|
| GLM-5.1-FP8 | `serving/serve_glm51.slurm` | TP8; arch needs source-built DeepGEMM in the `kimi` env |
| Qwen3.6 | `serving/serve_qwen36.slurm` | `qwen3_xml` tool parser |
| DeepSeek-V4-Flash | `settings/serving/deepseek_v4_flash_vllm.sh` | docker image `vllm-openai:deepseekv4-cu130`, DP4 |

Independent-lineage frontier models (Gemini-3-Pro / Opus-4.7 / GPT-5.5) are run via
**OpenRouter** with the provider pinned to the first-party upstream.

**Preemptible orchestration.** Serving runs on the `preemptible` partition and is scancelled
when done. Because an 8-GPU job can be preempted mid-run, every long step is **resumable**
(append + `os.fsync`, skip already-done `task_id`s): `gen_rubrics.py`, `checklist_judge.py`,
and the harness completion loop (`harness/completion_runner.py`, checkpoint = `traces.jsonl`).
`scripts/run_nextbatch.sh` wraps this in an outer loop that re-establishes endpoints and
re-runs resumable steps until both tracks complete. Submit-once-and-wait is preferred over
cancel+resubmit (which resets queue priority).

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
│   ├── checklist_judge.py / checklist_judge_or.py # Rubric judge (local / OpenRouter)
│   ├── build_leaderboard_t0.py                   # T=0 frozen/full-core leaderboard
│   ├── compute_buckets.py                        # 3-model difficulty buckets
│   ├── classify_allpass.py                       # Audit all-3-pass questions
│   ├── stage2_judge.py                           # Grounded open_status re-judgment
│   ├── audit_contamination.py                    # LLM-free contamination screen
│   ├── contamination_positive_control.py         # Sensitivity check for the audit
│   ├── membership_verbatim.py / membership_perplexity.py  # Memorization probes
│   ├── exp_behavioral.py                         # Agentic-collapse / tool-use behavior
│   ├── exp_cite_audit.py / l2_claude_judge.py    # Citation factuality (L1 existence / L2 support)
│   ├── analysis_afg.py                           # Per-domain + no-tool criterion decomposition
│   ├── roster_robustness.py / redecode_compare.py # Frozen-core robustness & reproducibility
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
│   ├── export/                     # Corpus files (mcp_benchmark_v3*.jsonl, *_questions.jsonl)
│   ├── eval_samples/               # Core/frozen eval sets + rubrics
│   └── hf_release/openbiorq/       # HuggingFace release (frozen 423 / full 657 / rubrics)
│
├── serving/                        # vLLM serve scripts + job-id / node state
├── results/                        # Eval traces, checklist scores, leaderboards, audits
└── paper_writing/                  # OpenBioRQ paper + FACTS.md (number source of truth)
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
  "empirical_difficulty": "core",
  "model_scores_glmjudge": {"glm": 0.30, "qwen": 0.29, "deepseek_v4": 0.27},
  "n_models_fail": 3,
  "difficulty_source": "3model_empirical_1969_uid",
  "still_open_flag": "confirmed_open",
  "quarantine": false
}
```

> **`task_id` is unique per question** (`{source_id}#{k}`). Extraction produces several
> questions per source paper; keying eval/rubrics/judging on `source_id` alone would silently
> collapse them to source-paper granularity. The gold slice (1,969) and all eval sets use
> question-granular `task_id` (`rubrics_1969_uid.jsonl`).

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

> `self_completeness` is the model's self-assessed answerability — useful for relative
> comparison, not as an absolute measure, distinct from the empirical difficulty.

> Gold citations exist (>99% real PMIDs, NCBI-verified) but ~**74%** point to the wrong paper
> (LLM-judge n≈360; cross-family confirmed 72.8%, κ=0.99). Gold citations are therefore
> **not** used as ground truth — scoring relies on the per-question checklist rubric.

## Taxonomy (12 L1 categories)

Clinical Medicine · Oncology · Neuroscience & Psychiatry · Infectious Disease & Immunology ·
Cardiovascular Medicine · Genomics & Precision Medicine · Pharmacology & Drug Discovery ·
Public Health & Epidemiology · Rare & Orphan Diseases · Surgical Sciences ·
Medical AI & Informatics · Other
