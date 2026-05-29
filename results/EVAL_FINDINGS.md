# Evaluation findings & difficulty recalibration (2026-05-29/30)

First end-to-end model evaluation of the open-medical-questions benchmark, on a
300-item stratified sample of the gold-bearing corpus, with two served models
(GLM-5.1-FP8, Qwen3.6-35B-A3B) using real MCP tools, scored by per-question
checklist rubrics with two judges (GLM + Qwen).

## Headline results (n=300; n=222 with all 4 judgings)

| metric | value |
|---|---|
| Judge agreement (checklist), GLM answers | Spearman **0.79** |
| Judge agreement (checklist), Qwen answers | Spearman **0.86** |
| Fair model score (2-judge mean): GLM-5.1 | **0.419** |
| Fair model score: Qwen3.6 | **0.326** |
| Per-item GLM-vs-Qwen difficulty corr | Spearman **0.33** |
| Empirical difficulty spread (hard→easy thirds) | 0.221 → 0.523 |
| Nanje buckets | both-fail **55%** / split **35%** / both-pass **10%** |

## What the investigation established (in order)

1. **Self-rated difficulty is invalid.** LLM-assigned difficulty (clinical_knowledge/
   research_depth/multi_step) does not correlate with model performance — self-"hard"
   items scatter across empirical hard/med/easy; self-"easy" are often empirically hard.
   → DEPRECATED. Use empirical (pass-based) difficulty.

2. **Free-form judge dimensions were unreliable.** Same answers scored 0.50 (GLM judge)
   vs 0.30 (Qwen judge); pass-rate flipped 0%↔51% by judge; coverage/completeness were
   the noise source (Spearman 0.19). Root cause: judges disagreed on what "complete"
   means — GLM rewarded comprehensive prose, Qwen penalized ungrounded/parametric answers
   (GLM bailed to memory on 60%+ of items, falsely claiming "tools returned nothing").

3. **Per-question checklist rubrics fixed it.** Instance-specific criteria
   (must_mention / must_acknowledge / must_ground / must_avoid), judged met/partial/
   not_met, lifted judge agreement to Spearman 0.79–0.86 and removed the strictness
   offset. Both judges now grade the SAME concrete checklist. (HealthBench-style.)

4. **Fair comparison + clean empirical difficulty** then become possible (above table).

## Open caveat — needs ≥3 models

GLM-vs-Qwen per-item difficulty correlate only Spearman ~0.33: item difficulty is
partly model-specific, so a 2-model pass average is a START but not a stable label.
Recommend adding a 3rd (and ideally 4th) model before freezing difficulty labels.

## v3 difficulty recalibration plan (12,553 items)

The empirical-difficulty methodology is validated; applying it to all of v3 requires
running models, which is expensive. Recommended staging:

1. **Deprecate self-rated difficulty in v3** (it is noise). Keep it only as metadata.
2. **Empirical labels where available**: 300 items now have judge-reliable empirical
   difficulty + nanje_bucket (`results/empirical_difficulty_checklist_300.jsonl`).
   These are all from the gold-bearing subset.
3. **Expand on the gold subset first** (1,969): generate rubrics + run ≥3 models +
   checklist-judge → empirical labels for the full evaluable set. Cost driver is model
   generation (Europe PMC removed the NCBI 429 bottleneck; ~hours/model on 1 node).
4. **Curated tracks (JLA/NICE, ~5,905)** lack gold → need gold generation before they
   can be checklist-judged the same way (or rubrics drawn directly from the JLA/NICE
   criteria themselves, which are already structured).
5. **Freeze**: an item is a "core nanje" if ≥N strong models fail it (both-fail bucket);
   demote both-pass items; split items are the most discriminating — keep as the eval core.

## Reusable pipeline (all committed)
- serving: `serve_glm51.slurm` (DeepGEMM v2.1.1 built for torch 2.10), `serve_qwen36.slurm`
- `scripts/gen_rubrics.py` → per-question checklists
- `scripts/checklist_judge.py` → reliable scoring (any judge)
- harness PubMed tool → Europe PMC (concurrency-safe), rate 0.34s
- analysis recipe in this file's headline query (per_task.csv + checklist_*.jsonl)
