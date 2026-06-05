#!/usr/bin/env bash
# Re-measure roster Qwen3.6 + DeepSeek-V4 on core 657 at TEMPERATURE 0.0 (GLM-5.1@T0 already
# exists as the with-tools control). PREEMPTIBLE + patient: wait indefinitely for endpoints;
# if a serve job is preempted, resubmit and resume (harness checkpoint). Per-model scancel on
# completion. Then judge, re-define the robust core, build the temperature-0 leaderboard.
set -u
cd /data/project/private/minstar/workspace/healthcare-research
source /data/project/private/minstar/miniconda3/etc/profile.d/conda.sh
conda activate kimi
export OPENAI_API_KEY=dummy EVAL_TEMPERATURE=0.0
LOG=results/roster_temp0_run.log
CORE=data/eval_samples/core_eval.jsonl
RUB=data/eval_samples/rubrics_1969_uid.jsonl
NCORE=$(wc -l < $CORE)
DSV4_SCRIPT=/data/project/private/minstar/settings/serving/deepseek_v4_flash_vllm.sh
log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG" >&2; }
jid_of(){ squeue -u minstar -h -n "$1" -o '%i' 2>/dev/null|head -1; }
cnt(){ python - "$1" <<'PY'
import json,sys
try: print(len({json.loads(l)["task_id"] for l in open(sys.argv[1]) if json.loads(l).get("checklist_score") is not None}))
except FileNotFoundError: print(0)
PY
}
traces(){ wc -l < "results/baseline_$1/traces.jsonl" 2>/dev/null || echo 0; }

# patient + preempt-resilient eval; resubmits its own serve job if preempted/missing
eval_roster(){ # jobname served tag serve_script
  local jn=$1 sv=$2 tag=$3 script=$4 jid st node
  while [ "$(traces $tag)" -lt "$NCORE" ]; do
    jid=$(jid_of "$jn")
    if [ -z "$jid" ]; then
      jid=$(sbatch --partition=preemptible --parsable "$script")
      log "[$tag] (re)submitted $jn -> $jid"; sleep 20; continue
    fi
    st=$(squeue -j "$jid" -h -o '%T' 2>/dev/null)
    if [ "$st" = "RUNNING" ]; then
      node=$(squeue -j "$jid" -h -o '%N' 2>/dev/null)
      if [ -n "$node" ] && curl -s --max-time 4 "http://$node:8000/v1/models" 2>/dev/null | grep -q "$sv"; then
        log "[eval//] $tag @ $node (T=0, resumable)"
        OPENAI_API_BASE=http://$node:8000/v1 python harness/run.py --data $CORE \
          --model "openai::http://$node:8000/v1::$sv" --no-judge --workers 12 \
          --output results/baseline_${tag} >> "$LOG" 2>&1
        continue   # re-check trace count (handles mid-eval preemption -> resume)
      fi
    fi
    sleep 60   # pending or loading -> wait patiently
  done
  jid=$(jid_of "$jn"); [ -n "$jid" ] && scancel "$jid" && log "[gpu freed] scancel $jn ($tag done: $(traces $tag) traces)"
}

log "=== roster Qwen3.6 + DSV4 @ T=0 (preemptible, patient) ==="
eval_roster serve-qwen36     qwen-3.6          roster_qwen36_t0 serving/serve_qwen36.slurm &
eval_roster serve-dsv4flash  deepseek-v4-flash roster_dsv4_t0   "$DSV4_SCRIPT" &
wait
log "=== judging roster@T0 ==="
# resolve/keep GLM-5.1 judge (resubmit if preempted)
GN=""
while [ -z "$GN" ]; do
  jid=$(jid_of serve-glm51); [ -z "$jid" ] && { jid=$(sbatch --partition=preemptible --parsable serving/serve_glm51.slurm); log "resubmitted GLM judge $jid"; sleep 20; continue; }
  node=$(squeue -j "$jid" -h -o '%N' 2>/dev/null); [ "$(squeue -j $jid -h -o '%T')" = RUNNING ] && [ -n "$node" ] && curl -s --max-time 4 "http://$node:8000/v1/models" 2>/dev/null | grep -q glm-5.1 && GN=$node || sleep 60
done
for tag in roster_qwen36_t0 roster_dsv4_t0; do
  [ "$(cnt results/baseline_${tag}/checklist_glm.jsonl)" -lt "$NCORE" ] || continue
  log "[judge] $tag"
  python scripts/checklist_judge.py --traces results/baseline_${tag}/traces.jsonl \
    --rubrics $RUB --judge openai/glm-5.1 --base http://$GN:8000/v1 \
    --out results/baseline_${tag}/checklist_glm.jsonl --workers 8 2>&1 | grep -ivE httpx >> "$LOG"
done
conda activate base
python scripts/build_leaderboard_t0.py 2>&1 | tee -a "$LOG"
jid=$(jid_of serve-glm51); [ -n "$jid" ] && scancel "$jid" && log "scancel GLM-5.1 judge (all T=0 done)"
echo "ROSTER_TEMP0_DONE" | tee -a "$LOG"
