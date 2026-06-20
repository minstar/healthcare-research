#!/usr/bin/env bash
# Parallel GLM-5.1 judging for the 3 configs not already being judged (opus is running
# as an orphan from judge_fullcore.sh). checklist_judge resumes (skips scored), so gpt55
# only re-tries its 2 errored traces. Same endpoint/rubric/threshold as the sequential run.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "${CONDA_ENV:-kimi}"
RUB=data/eval_samples/rubrics_1969_uid.jsonl
GN="${GLM_NODE:?set GLM_NODE=<GLM-5.1 serving node host>}"
LOG=results/judge_par.log
: > "$LOG"
for tag in api_gpt55_fullcore api_gemini3pro_fullcore api_gpt55_notool_fullcore; do
  OPENAI_API_KEY=dummy python scripts/checklist_judge.py --traces results/$tag/traces.jsonl \
    --rubrics "$RUB" --judge openai/glm-5.1 --base "http://$GN:8000/v1" \
    --out results/$tag/checklist_glm.jsonl --workers 8 >> "$LOG" 2>&1 &
done
wait
echo "JUDGE_PAR_DONE" | tee -a "$LOG"
