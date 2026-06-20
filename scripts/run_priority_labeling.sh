#!/usr/bin/env bash
# Task 2: empirical 3-model labeling of the priority 525.
# Prereq: serving up; node names in serving/.node_{glm,qwen,dsv4}.
set -u
cd /data/project/private/minstar/workspace/healthcare-research
source /data/project/private/minstar/miniconda3/etc/profile.d/conda.sh
conda activate kimi
export OPENAI_API_KEY=dummy

GN=$(cat serving/.node_glm); QN=$(cat serving/.node_qwen); VN=$(cat serving/.node_dsv4)
DATA=data/export/priority_questions.jsonl
RUB=data/eval_samples/rubrics_priority525.jsonl
echo "[nodes] glm=$GN qwen=$QN dsv4=$VN"

# --- 1. rubrics (once, GLM) ---
if [ ! -s "$RUB" ]; then
  echo "[1/4] generating rubrics for 525..."
  python scripts/gen_rubrics.py --data $DATA --model glm-5.1 --base http://$GN:8000/v1 \
    --out $RUB --workers 8 > /tmp/prio_rubrics.log 2>&1
fi
echo "[1/4] rubrics: $(wc -l < $RUB)"

# --- 2. agentic eval, 3 models in parallel (distinct nodes) ---
echo "[2/4] agentic eval (3 models parallel)..."
run_eval(){ # node served_name out workers
  OPENAI_API_BASE=http://$1:8000/v1 python harness/run.py --data $DATA \
    --model "openai::http://$1:8000/v1::$2" --no-judge --workers $4 \
    --output results/priority_$3 > results/priority_$3.log 2>&1; echo "DONE_$3_$?" >> results/priority_$3.log
}
run_eval $GN glm-5.1          glm  10 &
run_eval $QN qwen-3.6         qwen 12 &
run_eval $VN deepseek-v4-flash dsv4 12 &
wait
echo "[2/4] eval done: glm=$(tail -1 results/priority_glm.log) qwen=$(tail -1 results/priority_qwen.log) dsv4=$(tail -1 results/priority_dsv4.log)"

# --- 3. checklist judge (GLM judge, sequential to avoid overload) ---
echo "[3/4] checklist judging..."
for m in glm qwen dsv4; do
  python scripts/checklist_judge.py --traces results/priority_${m}/traces.jsonl \
    --rubrics $RUB --judge openai/glm-5.1 --base http://$GN:8000/v1 \
    --out results/priority_${m}/checklist_glm.jsonl --workers 8 2>&1 | grep -ivE httpx | tail -1
done

# --- 4. buckets ---
echo "[4/4] buckets..."
python scripts/compute_buckets.py \
  --glm results/priority_glm/checklist_glm.jsonl \
  --qwen results/priority_qwen/checklist_glm.jsonl \
  --dsv4 results/priority_dsv4/checklist_glm.jsonl \
  --out /tmp/buckets_priority.json --threshold 0.5
echo "PRIORITY_LABELING_DONE"
