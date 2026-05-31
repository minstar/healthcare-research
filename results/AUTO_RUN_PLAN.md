# Autonomous run plan (user away ~10h, started 2026-05-29 ~23:xx)

Serving jobs to SCANCEL only at the very end: 21831 (Qwen c36f-i04-2-gpu164), 21737 (GLM c36f-g04-1-gpu115).
Endpoints: GLM serving/.serving_node, Qwen serving/.qwen_node.

## Pipeline (sequential, self-heal on errors)
1. [RUNNING] rubric gen 300 -> data/eval_samples/rubrics_300.jsonl  (/tmp/rub300.log)
2. [RUNNING] GLM eval 300 (--no-judge) -> results/baseline_glm_300  (results/baseline_glm_300.log, DONE_ marker)
3. [RUNNING] Qwen eval 300 (--no-judge) -> results/baseline_qwen_300 (results/baseline_qwen_300.log, DONE_ marker)
4. [ ] checklist-judge GLM answers w/ GLM judge + Qwen judge -> results/baseline_glm_300/checklist_{glm,qwen}.jsonl
5. [ ] checklist-judge Qwen answers w/ GLM judge + Qwen judge -> results/baseline_qwen_300/checklist_{glm,qwen}.jsonl
6. [ ] analysis: judge agreement (Spearman), fair model comparison, empirical difficulty (4-judging mean),
       nanje buckets (both_fail/split/both_pass) -> results/empirical_difficulty_checklist_300.jsonl + commit
7. [ ] v3 difficulty recalibration design: map empirical difficulty methodology; write a recalibration note + commit
8. [ ] FINAL: scancel 21831 21737; commit any remaining; summary report for the user

## Commands (reuse)
- checklist judge: python scripts/checklist_judge.py --traces <run>/traces.jsonl --rubrics data/eval_samples/rubrics_300.jsonl --judge openai/<m> --base http://<node>:8000/v1 --out <run>/checklist_<m>.jsonl
- judge=GLM base=$(cat serving/.serving_node):8000, judge=Qwen base=$(cat serving/.qwen_node):8000
- env: conda activate kimi; OPENAI_API_KEY=dummy

## Error-handling notes
- empty tool results / parser issues already fixed. NCBI rate 0.34s.
- if an endpoint died (job preempted/OOM), check squeue; GLM=21737 Qwen=21831. Resubmit serving/serve_*.slurm, update .serving_node/.qwen_node, recompute affected step.
- checklist judge needs litellm? No — uses openai client directly. judge max_tokens 4096 (thinking).

## FINAL PHASE (when comprehensive monitor bui4ygs1g fires: all 7 tracks done)
1. Aggregate 7-point report: (1) priority 581-doc new-question count + final corpus size,
   (2) P0 retention (vs old 6%) + P0 fabrication rate, (3) 1969 3-model difficulty buckets
   + empirical coverage (2.2%->~16%), (4) Stage2 open_status distribution (answered/unknown),
   (5) verification (self-cont 85% / citation 74% mismatch / contamination 10.7%),
   (6) paper applications, (7) v3 recalibration -> v3.2.
2. Build SFT trajectories from P0 + 1969 traces; run fabrication_rate on P0 traces.
3. GPU: run any remaining useful GPU work; if none, SCANCEL ONLY my serving jobs by EXPLICIT ID:
   scancel 23035 23020 23021   (GLM / Qwen / V4)
   DO NOT touch 21567_* / 21568_* (pt2-minstar-tau3 training — NOT mine; they also match 'glm51').
   NEVER scancel by name/pattern.

## STAGE2 RE-RUN (caught: 98% llm_fail from GLM contention)
Stage2 at-scale FAILED (4553/4646 llm_fail = GLM judge timeouts under 4-job node load).
In FINAL PHASE, BEFORE scancel, RE-RUN Stage2 once the GLM node is freed (other GLM
tracks done) with a higher timeout:
  - edit scripts/stage2_judge.py call_claude timeout 120->300, OR run when GLM idle.
  python scripts/stage2_judge.py --data data/export/mcp_benchmark_v2.jsonl --sample 100000 \
    --model glm-5.1 --workers 4 --out data/stage2_full
This is legitimate "remaining GPU work" to do before cancelling serving jobs.
