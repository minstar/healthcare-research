#!/usr/bin/env bash
# Task2 resume, preemption-resilient: serve GLM on PREEMPTIBLE, judge (resumable),
# and if the job is preempted mid-judge, re-serve and continue from the fsync'd partial.
# When all 3 models are fully judged -> buckets -> v3.3, then scancel GLM (turn off).
#
# Usage: bash scripts/serve_and_judge_priority.sh
set -u
cd /data/project/private/minstar/workspace/healthcare-research
source /data/project/private/minstar/miniconda3/etc/profile.d/conda.sh
conda activate kimi
export OPENAI_API_KEY=dummy
RUB=data/eval_samples/rubrics_priority525.jsonl
TOTAL=$(wc -l < data/export/priority_questions.jsonl)
LOG=results/priority_judge_run.log
echo "$(date '+%F %T') START serve_and_judge_priority (TOTAL=$TOTAL)" | tee -a $LOG

scored() {  # $1=model -> count of distinct task_ids with a real checklist_score
  python - "$1" <<'PY'
import json,sys
m=sys.argv[1]; f=f"results/priority_{m}/checklist_glm.jsonl"
try: d={json.loads(l)["task_id"] for l in open(f) if json.loads(l).get("checklist_score") is not None}
except FileNotFoundError: d=set()
print(len(d))
PY
}
all_done() { for m in glm qwen dsv4; do [ "$(scored $m)" -lt "$TOTAL" ] && return 1; done; return 0; }

# one-time: drop checklist files that are 100% errors (no resumable value)
for m in glm qwen dsv4; do
  f=results/priority_$m/checklist_glm.jsonl
  [ -f "$f" ] && [ "$(scored $m)" -eq 0 ] && { rm -f "$f"; echo "cleared all-error $f" | tee -a $LOG; }
done

ATTEMPT=0
while ! all_done && [ $ATTEMPT -lt 20 ]; do
  ATTEMPT=$((ATTEMPT+1))
  echo "$(date '+%F %T') [attempt $ATTEMPT] glm=$(scored glm) qwen=$(scored qwen) dsv4=$(scored dsv4) /$TOTAL" | tee -a $LOG
  JID=$(sbatch --partition=preemptible --parsable serving/serve_glm51.slurm)
  echo "$(date '+%F %T') submitted GLM(preemptible) job $JID" | tee -a $LOG
  GN=""
  for i in $(seq 1 100); do
    st=$(squeue -j $JID -h -o '%T' 2>/dev/null)
    [ -z "$st" ] && { echo "job $JID vanished while waiting" | tee -a $LOG; break; }
    n=$(squeue -j $JID -h -o '%N' 2>/dev/null)
    if [ -n "$n" ] && curl -s --max-time 4 http://$n:8000/v1/models 2>/dev/null | grep -q glm-5.1; then GN=$n; break; fi
    sleep 30
  done
  [ -z "$GN" ] && { echo "$(date '+%F %T') no endpoint (preempted/slow); retry" | tee -a $LOG; scancel $JID 2>/dev/null; continue; }
  echo "$(date '+%F %T') GLM up @ $GN -> rubrics + judging (resumable)" | tee -a $LOG
  # rubrics keyed by UNIQUE task_id (resumable); regenerate if incomplete
  python scripts/gen_rubrics.py --data data/export/priority_questions.jsonl \
    --model glm-5.1 --base http://$GN:8000/v1 --out $RUB --workers 8 2>&1 | grep -ivE httpx | tee -a $LOG
  for m in glm qwen dsv4; do
    python scripts/checklist_judge.py --traces results/priority_${m}/traces.jsonl \
      --rubrics $RUB --judge openai/glm-5.1 --base http://$GN:8000/v1 \
      --out results/priority_${m}/checklist_glm.jsonl --workers 8 2>&1 | grep -ivE httpx | tee -a $LOG
  done
  scancel $JID 2>/dev/null; echo "$(date '+%F %T') scancel GLM $JID (pass $ATTEMPT done)" | tee -a $LOG
done

if all_done; then
  echo "$(date '+%F %T') ALL JUDGED -> buckets + v3.3" | tee -a $LOG
  python scripts/compute_buckets.py --glm results/priority_glm/checklist_glm.jsonl \
    --qwen results/priority_qwen/checklist_glm.jsonl --dsv4 results/priority_dsv4/checklist_glm.jsonl \
    --out /tmp/buckets_priority.json --threshold 0.5 2>&1 | tee -a $LOG
  python scripts/build_v3_3.py 2>&1 | tee -a $LOG
  echo "PRIORITY_FULLY_DONE" | tee -a $LOG
else
  echo "$(date '+%F %T') INCOMPLETE after $ATTEMPT attempts (glm=$(scored glm) qwen=$(scored qwen) dsv4=$(scored dsv4))" | tee -a $LOG
fi
