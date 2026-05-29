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
