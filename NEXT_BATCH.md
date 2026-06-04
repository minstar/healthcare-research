# Next eval batch — GPU-serving queue (deferred 2026-06-02)

> **AUTONOMY (user-granted 2026-06-02):** When the next batch starts, spin up the
> serving stack (PREEMPTIBLE partition, priority-preserving wait, auto-scancel when done)
> and process the queued items — no further approval needed. Standard safety still applies:
> scancel ONLY my own serving jobs by explicit ID; NEVER touch training jobs.

## ⭐ NEXT-BATCH PRIORITY QUEUE (user-confirmed 2026-06-03)
1. **Expand 483 empirical labeling** → needs FULL 3-model agentic eval (GLM+Qwen+DSV4) since
   `expand_questions.jsonl` has NO traces yet (unlike priority which we salvaged). Pipeline:
   gen_rubrics(task_id) → 3-model agentic eval (harness, resumable checkpoint) → checklist
   judge → compute_buckets → merge into v3.4. Reuse `scripts/run_priority_labeling.sh` pattern
   but with --data data/export/expand_questions.jsonl (already has unique task_id).
2. **v3.2 core_nanje QUESTION-level relabel** → core_nanje currently 657 at source-paper
   granularity. Assign unique task_id to ALL v3 questions, re-key the 1969 traces/rubrics by
   position (verify positional match first like priority), re-judge, recompute buckets. This
   makes the whole benchmark question-granular and consistent with priority/expand.
- Both are GLM-judge heavy; expand also needs Qwen+DSV4 for the agentic completions.
- Use the priority-preserving orchestrator pattern: `scripts/wait_and_judge_priority.sh`
  (submit once on preemptible, wait without cancel-on-timeout, resume on real preemption).

## ⚠️ SYSTEMIC BUG found 2026-06-02: non-unique task_id
- task_id defaulted to source_id, but extraction yields MULTIPLE questions per source paper
  (priority: 525 questions from 223 papers; 1969 → ~1396 distinct). So eval/rubric/judge/buckets
  collapse to source-paper granularity — empirical labels undercount distinct questions.
- FIXED for priority+expand: assigned unique `task_id` = `{source_id}#{k}`; traces remapped by
  position (verified 525/525), `.bak` saved; gen_rubrics now prefers task_id field (+ resumable).
- TODO: re-apply unique task_id + RE-LABEL v3.2 core_nanje at QUESTION granularity (next batch).
  Until then, v3.2 core_nanje=657 is at ~source-paper granularity (still valid, just coarse).

## STATUS (2026-06-02 batch run)
- ✅ **Task 1 DONE** — stage2 re-judgment of 85 flags: 0 confirmed-resolved, core_nanje stays 657. (verdicts merged into v3.2: `stage2_status`, `still_open_flag`)
- ✅ **Task 3 DONE** — core_nanje agentic baseline (subset of 1969 eval): GLM 0.303 / Qwen 0.295 / DSV4 0.272, pass@0.5≈0%. `results/core_nanje_baseline.json`
- 🔶 **Task 2 PARTIAL** — agentic eval DONE & SAVED (`results/priority_{glm,qwen,dsv4}/traces.jsonl`, 525 each, real answers+tools), rubrics DONE (`data/eval_samples/rubrics_priority525.jsonl`). **Checklist judging FAILED**: serving jobs 23556/57/58 were cancelled (by user, freeing GPU) at 10:59 right after eval, so all 525×3 judge calls got "Connection error".
  - **RESUME (1 command, needs only GLM judge):** `sbatch serving/serve_glm51.slurm` → wait ~15min → `bash scripts/resume_priority_judge.sh <new_GLM_jobid>`. Does judging → buckets → v3.3. NO need to re-run the expensive agentic eval.
  - Note: `data/export/mcp_benchmark_v3.3.jsonl` currently exists but with 0 priority labels (placeholder); resume regenerates it correctly.

All three items below need the 3-model serving stack up (GLM-5.1 + Qwen3.6 + DeepSeek-V4
on B200). Batch them together in one serving window.

## 1. LLM stage2 re-judgment of still-open flags (85 core_nanje)
- Input: `data/export/mcp_benchmark_v3.2.jsonl` rows where `still_open_flag in {high_risk, recheck}`
  - high_risk 11 (completed trial + results posted), recheck 74
- Detail file: `results/still_open_audit_core.jsonl`
- Action: grounded re-judgment (scripts/stage2_judge.py) — confirm whether the broad
  research question is actually resolved. Only LLM-confirmed-resolved -> quarantine.
- Goal: finalize core_nanje set (currently 657, hard_contamination=0%, 100% eval-ready).

## 1b. (NEW 2026-06-02) Expansion track — 483 questions, needs empirical labeling
- `data/export/expand_questions.jsonl` (483 new: priority2 496 + cochrane2 114 PMID crawl → extract → refine → export). open 352 / partial 131, 12 domains balanced.
- Contamination audit done: hard_contamination 0%, verifiable 100% (`results/audit_expand/`).
- medRxiv preprints (400) yielded 0 questions — results-reporting abstracts, NOT open-question docs. Preprints confirmed unsuitable as nanje source; future crawl = priority-setting / research-gap only.
- TODO (next labeling batch): same 3-model agentic eval + checklist + buckets as priority, merge into v3.3/v3.4. Reuse `scripts/run_priority_labeling.sh` pattern with --data expand_questions.jsonl + gen_rubrics.

## 2. Priority 525 empirical labeling -> v3.3
- Input: `data/export/priority_questions.jsonl` (525 new, open 445 / partial 80)
- Action: 3-model checklist eval (harness/run.py --no-judge on glm/qwen/dsv4) -> buckets
  (all3_fail/split/all3_pass @0.5) -> empirical_difficulty, same as v3.2 recalibration.
- Then merge into main -> `data/export/mcp_benchmark_v3.3.jsonl`.
- Reuse: scripts/recalibrate_v3_2.py pattern (bucket source_id -> question text join).

## 3. Actual agentic eval (C) on core_nanje
- 657 core_nanje already eval-ready (gold + rubric >=3 criteria, 100%).
- Rubrics: `data/eval_samples/rubrics_1969.jsonl`; gold: `data/export/mcp_benchmark_with_gold.jsonl`.
- Run candidate model(s) (Solar etc.) through harness/run.py with checklist judging.

## Serving launch reference
- `scripts/launch_verification_on_glm.sh` (waits for GLM endpoint, fans out tracks)
- Backup before any v3.2 mutation: `/tmp/v3.2.bak.jsonl` (current good state)
