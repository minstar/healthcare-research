#!/usr/bin/env bash
# RESUME Task2 after serving was cancelled mid-judging.
# The expensive agentic eval is DONE (results/priority_{glm,qwen,dsv4}/traces.jsonl, 525 each).
# Only checklist judging + buckets + v3.3 remain — these need ONLY the GLM judge endpoint.
#
# Usage:
#   1) re-serve GLM:  sbatch serving/serve_glm51.slurm   (wait ~15min for load)
#   2) bash scripts/resume_priority_judge.sh <GLM_JOBID>
set -u
cd /data/project/private/minstar/workspace/healthcare-research
source /data/project/private/minstar/miniconda3/etc/profile.d/conda.sh
conda activate kimi
export OPENAI_API_KEY=dummy

GJOB=${1:?usage: resume_priority_judge.sh <GLM_JOBID>}
# wait for GLM endpoint healthy
for i in $(seq 1 80); do
  GN=$(squeue -j $GJOB -h -o '%N' 2>/dev/null)
  [ -z "$(squeue -j $GJOB -h -o '%T' 2>/dev/null)" ] && { echo "GLM job $GJOB gone"; exit 1; }
  if [ -n "$GN" ] && curl -s --max-time 4 http://$GN:8000/v1/models 2>/dev/null | grep -q glm-5.1; then
    echo "$GN" > serving/.node_glm; echo "GLM UP @ $GN"; break
  fi
  sleep 30
done
GN=$(cat serving/.node_glm)
RUB=data/eval_samples/rubrics_priority525.jsonl

echo "=== checklist judging (3 models, sequential on GLM judge) ==="
for m in glm qwen dsv4; do
  python scripts/checklist_judge.py --traces results/priority_${m}/traces.jsonl \
    --rubrics $RUB --judge openai/glm-5.1 --base http://$GN:8000/v1 \
    --out results/priority_${m}/checklist_glm.jsonl --workers 8 2>&1 | grep -ivE httpx | tail -1
done

echo "=== buckets ==="
python scripts/compute_buckets.py \
  --glm results/priority_glm/checklist_glm.jsonl \
  --qwen results/priority_qwen/checklist_glm.jsonl \
  --dsv4 results/priority_dsv4/checklist_glm.jsonl \
  --out /tmp/buckets_priority.json --threshold 0.5

echo "=== build v3.3 ==="
python scripts/build_v3_3.py
echo "RESUME_PRIORITY_DONE"
