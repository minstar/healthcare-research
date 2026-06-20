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

---

# Extended phase: verification at scale, paper applications, data growth (2026-05-31 → 06-01)

Served GLM-5.1 + Qwen3.6 + DeepSeek-V4-Flash on B200 (cluster is Blackwell, not the
H200 in CLAUDE.md). Ran verification, trajectory-SFT, and scale-up tracks in parallel.

## Verification vs ResearchMath-14K (Appendix H) — sufficiency check

Our verification machinery is paper-grade (some parts stronger), but scale-execution
lagged; this phase closed the gaps and surfaced a major one.

| dimension | paper | us | finding |
|---|---|---|---|
| self-containment audit | 67→94.2% (500) | **85.4% fully** (500): retrieval .975 / consensus .852 | solid; consensus track terser |
| citation correctness | agent-judge web-verify | LLM-judge of PMID↔claim (360) | 🔴 **~74% of gold citation PMIDs are MISMATCHED** (point to unrelated papers; old "0% hallucination" only checked existence) |
| source concentration | top-50 = 25.78% | **top-50 = 10.7%** (3,798 docs) | 🟢 **we are MORE diverse than the paper** |
| open_status grounding | refiner reads citing papers, all items | status_verifier (Europe PMC/CT.gov) all; Stage-2 LLM re-judge | ⚠️ Stage-2 at-scale FAILED (GLM contention, 98% llm_fail) → re-run pending |
| difficulty | LLM tournament (Elo) | empirical 3-model pass-rate | 🟢 stronger; cross-benchmark Elo positioning still TODO |

## Paper elements applied

- **Trajectory → SFT flywheel (H.3):** eval traces ARE agentic trajectories.
  `build_sft_trajectories.py` filters them (non-attempt / fake-citation in-trace /
  quality gate) into SFT messages. 300×3: 900→58 kept (6%, matches paper 220K→5K
  selectivity). non-attempt drops 71% (the parametric-fallback problem). **Best teacher
  = DeepSeek-V4** (42/58); GLM bails most (3/58). At 1969: ~1,140 candidates pre-quality
  (qwen 25% / dsv4 33%).
- **Citation-fabrication metric (H.4):** `fabrication_rate.py`. Models cite sparsely
  under the old prompt, but ~90% of citations are ungrounded (cited but not retrieved
  in-trace) = the paper's memory-citation pattern.
- **Contamination top-N concentration:** computed (10.7% top-50, better than paper).

## Negative results caught by A/B (verify-before-trust)

- **P0 agent-prompt upgrade FAILED.** The verbose open-question/structured prompt
  collapsed DeepSeek-V4 tool use: 0-tool-call traces 2% → **91%**, answers truncated
  mid-plan. Reverted to the minimal prompt. P0 needs a tool-preserving redesign.
- **Jaccard citation metric was an artifact** (97.5% "weak" = short-note-vs-abstract
  Jaccard ≈ 0) — replaced by the LLM citation-judge (which then found the real 74%
  PMID-mismatch problem).
- **Stage-2 at scale** failed on GLM-node contention (4 concurrent jobs → judge
  timeouts). Re-run scheduled with a freed GLM + timeout 120→300.

## Data growth

Authoritative priority-setting crawler (`priority_sources_crawler.py`, Europe PMC):
581 docs (consensus/scientific statements, society research agendas, top-10 priorities,
CHNRI, Delphi) → 387 extracted → **corpus 6,733 → 7,115 (+382)**. (RePORTER 2.9M grants
and AHRQ Bookshelf reconned but deferred as lower-precision / lower-yield.)

## Pending (final phase)

- 1969 3-model agentic eval (glm-eval finishing) → checklist-judge all 3 → empirical
  difficulty at scale (coverage 2.2% → ~16%) → v3.2 recalibration.
- Stage-2 re-run on freed GLM.
- SFT flywheel + fabrication at full 1969 scale.
- TODO: gold-citation regeneration (74% mismatch); tournament/Elo cross-benchmark
  positioning; P0 v2 tool-preserving redesign.
