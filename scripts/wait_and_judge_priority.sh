#!/usr/bin/env bash
# Task2 finish on PREEMPTIBLE, priority-preserving: reuse the already-queued GLM job and
# WAIT (no cancel-on-timeout — cancelling would reset queue priority). When it lands and the
# endpoint is healthy: gen rubrics (unique task_id, resumable) + judge 3 models (resumable)
# + buckets + v3.3, then scancel. Only resubmit a new job if the running one is PREEMPTED
# (vanishes) before all judging is done — the fsync'd partials let it resume.
#
# Usage: bash scripts/wait_and_judge_priority.sh [existing_glm_jobid]
set -u
cd /data/project/private/minstar/workspace/healthcare-research
source /data/project/private/minstar/miniconda3/etc/profile.d/conda.sh
conda activate kimi
export OPENAI_API_KEY=dummy
RUB=data/eval_samples/rubrics_priority525.jsonl
TOTAL=$(wc -l < data/export/priority_questions.jsonl)
LOG=results/priority_judge_run.log
echo "$(date '+%F %T') START wait_and_judge_priority (TOTAL=$TOTAL, preemptible, priority-preserving)" | tee -a $LOG

scored() { python - "$1" <<'PY'
import json,sys
m=sys.argv[1]; f=f"results/priority_{m}/checklist_glm.jsonl"
try: d={json.loads(l)["task_id"] for l in open(f) if json.loads(l).get("checklist_score") is not None}
except FileNotFoundError: d=set()
print(len(d))
PY
}
rubrics_done() { [ -f "$RUB" ] && [ "$(python -c "import json;print(sum(1 for l in open('$RUB') if json.loads(l).get('criteria')))" 2>/dev/null||echo 0)" -ge "$TOTAL" ]; }
all_done() { rubrics_done || return 1; for m in glm qwen dsv4; do [ "$(scored $m)" -lt "$TOTAL" ] && return 1; done; return 0; }

JID="${1:-}"
[ -n "$JID" ] && squeue -j "$JID" -h -o '%T' >/dev/null 2>&1 || JID=""

while ! all_done; do
  # ensure exactly one GLM job exists in queue/run
  if [ -z "$JID" ] || ! squeue -j "$JID" -h -o '%T' >/dev/null 2>&1 || [ -z "$(squeue -j $JID -h -o '%T' 2>/dev/null)" ]; then
    JID=$(sbatch --partition=preemptible --parsable serving/serve_glm51.slurm)
    echo "$(date '+%F %T') submitted GLM(preemptible) job $JID (resume on preemption)" | tee -a $LOG
  fi
  # wait (indefinitely) for this job to run + endpoint healthy; do NOT cancel on slowness
  GN=""
  while :; do
    st=$(squeue -j $JID -h -o '%T' 2>/dev/null)
    if [ -z "$st" ]; then echo "$(date '+%F %T') job $JID left queue (started+ended or preempted)" | tee -a $LOG; break; fi
    if [ "$st" = "RUNNING" ]; then
      n=$(squeue -j $JID -h -o '%N' 2>/dev/null)
      if [ -n "$n" ] && curl -s --max-time 4 http://$n:8000/v1/models 2>/dev/null | grep -q glm-5.1; then GN=$n; break; fi
    fi
    sleep 60
  done
  [ -z "$GN" ] && { JID=""; continue; }   # vanished before healthy -> resubmit

  echo "$(date '+%F %T') GLM up @ $GN -> rubrics + judging (resumable)" | tee -a $LOG
  python scripts/gen_rubrics.py --data data/export/priority_questions.jsonl \
    --model glm-5.1 --base http://$GN:8000/v1 --out $RUB --workers 8 2>&1 | grep -ivE httpx | tee -a $LOG
  for m in glm qwen dsv4; do
    python scripts/checklist_judge.py --traces results/priority_${m}/traces.jsonl \
      --rubrics $RUB --judge openai/glm-5.1 --base http://$GN:8000/v1 \
      --out results/priority_${m}/checklist_glm.jsonl --workers 8 2>&1 | grep -ivE httpx | tee -a $LOG
  done
  echo "$(date '+%F %T') pass done: rubrics=$(rubrics_done && echo ok || echo partial) glm=$(scored glm) qwen=$(scored qwen) dsv4=$(scored dsv4) /$TOTAL" | tee -a $LOG
  if all_done; then scancel $JID 2>/dev/null; echo "$(date '+%F %T') scancel GLM $JID (all done)" | tee -a $LOG; fi
done

echo "$(date '+%F %T') ALL JUDGED -> buckets + v3.3" | tee -a $LOG
python scripts/compute_buckets.py --glm results/priority_glm/checklist_glm.jsonl \
  --qwen results/priority_qwen/checklist_glm.jsonl --dsv4 results/priority_dsv4/checklist_glm.jsonl \
  --out /tmp/buckets_priority.json --threshold 0.5 2>&1 | tee -a $LOG
python scripts/build_v3_3.py 2>&1 | tee -a $LOG
echo "PRIORITY_FULLY_DONE" | tee -a $LOG
