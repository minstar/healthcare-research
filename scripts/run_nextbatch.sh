#!/usr/bin/env bash
# Next batch (preempt-tolerant): (1) expand 483 full 3-model labeling, (2) v3.2/1969
# question-granular relabel, (3) build v3.4. Outer loop re-establishes endpoints and
# re-runs RESUMABLE steps until both tracks are complete, so GLM/DSV4 preemption just
# means another loop iteration. wait_ep logs to stderr/file only (stdout = node, clean).
set -u
cd /data/project/private/minstar/workspace/healthcare-research
source /data/project/private/minstar/miniconda3/etc/profile.d/conda.sh
conda activate kimi
export OPENAI_API_KEY=dummy
LOG=results/nextbatch_run.log
EXP=data/export/expand_questions.jsonl
RUBE=data/eval_samples/rubrics_expand483.jsonl
RUB1969=data/eval_samples/rubrics_1969_uid.jsonl
GOLD=data/export/mcp_benchmark_with_gold.jsonl
DSV4_SCRIPT=/data/project/private/minstar/settings/serving/deepseek_v4_flash_vllm.sh
NE=$(wc -l < $EXP); N1969=$(wc -l < $GOLD)
. serving/.batch_jobids2

log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG" >&2; }   # NEVER to stdout

wait_ep(){ # $1=jobvar $2=served $3=script ; echoes node to STDOUT only
  local jv=$1 served=$2 script=$3 jid n st
  jid=$(eval echo \$$jv)
  while :; do
    st=$(squeue -j "$jid" -h -o '%T' 2>/dev/null)
    if [ -z "$st" ]; then
      jid=$(sbatch --partition=preemptible --parsable "$script")
      eval "$jv=$jid"; sed -i "s/^$jv=.*/$jv=$jid/" serving/.batch_jobids2
      log "resubmitted $served -> $jid (preempted/failed)"; sleep 20; continue
    fi
    if [ "$st" = "RUNNING" ]; then
      n=$(squeue -j "$jid" -h -o '%N' 2>/dev/null)
      if [ -n "$n" ] && curl -s --max-time 4 "http://$n:8000/v1/models" 2>/dev/null | grep -q "$served"; then
        echo "$n"; return 0
      fi
    fi
    sleep 60
  done
}

cnt(){ python - "$1" <<'PY'
import json,sys
try: print(len({json.loads(l)["task_id"] for l in open(sys.argv[1]) if json.loads(l).get("checklist_score") is not None}))
except FileNotFoundError: print(0)
PY
}
rubn(){ python - "$1" <<'PY'
import json,sys
try: print(sum(1 for l in open(sys.argv[1]) if json.loads(l).get("criteria")))
except FileNotFoundError: print(0)
PY
}
tracen(){ wc -l < "$1" 2>/dev/null || echo 0; }
expand_done(){ for m in glm qwen dsv4; do [ "$(cnt results/expand_$m/checklist_glm.jsonl)" -lt "$NE" ] && return 1; done; return 0; }
relabel_done(){ for m in glm qwen dsv4; do [ "$(cnt results/baseline_${m}_1969/checklist_uid.jsonl)" -lt "$N1969" ] && return 1; done; return 0; }

ATT=0
while { ! expand_done || ! relabel_done; } && [ $ATT -lt 40 ]; do
  ATT=$((ATT+1)); log "===== attempt $ATT (expand_done=$(expand_done && echo y || echo n) relabel_done=$(relabel_done && echo y || echo n)) ====="
  GN=$(wait_ep GLM glm-5.1 serving/serve_glm51.slurm)
  QN=$(wait_ep QWEN qwen-3.6 serving/serve_qwen36.slurm)
  VN=$(wait_ep DSV4 deepseek-v4-flash "$DSV4_SCRIPT")
  log "endpoints glm=$GN qwen=$QN dsv4=$VN"

  # ---------- TRACK 1: expand 483 (full 3-model) ----------
  if ! expand_done; then
    [ "$(rubn $RUBE)" -lt "$NE" ] && { log "[expand] gen_rubrics ($(rubn $RUBE)/$NE)"; \
      python scripts/gen_rubrics.py --data $EXP --model glm-5.1 --base http://$GN:8000/v1 --out $RUBE --workers 8 >>"$LOG" 2>&1; }
    declare -A NODE=( [glm]=$GN [qwen]=$QN [dsv4]=$VN ) SV=( [glm]=glm-5.1 [qwen]=qwen-3.6 [dsv4]=deepseek-v4-flash ) W=( [glm]=10 [qwen]=12 [dsv4]=12 )
    for m in glm qwen dsv4; do
      if [ "$(tracen results/expand_$m/traces.jsonl)" -lt "$NE" ]; then
        log "[expand] agentic eval $m ($(tracen results/expand_$m/traces.jsonl)/$NE)"
        OPENAI_API_BASE=http://${NODE[$m]}:8000/v1 python harness/run.py --data $EXP \
          --model "openai::http://${NODE[$m]}:8000/v1::${SV[$m]}" --no-judge --workers ${W[$m]} \
          --output results/expand_$m >>"$LOG" 2>&1 &
      fi
    done
    wait
    for m in glm qwen dsv4; do
      [ "$(cnt results/expand_$m/checklist_glm.jsonl)" -lt "$NE" ] && {
        log "[expand] judge $m"; python scripts/checklist_judge.py --traces results/expand_$m/traces.jsonl \
          --rubrics $RUBE --judge openai/glm-5.1 --base http://$GN:8000/v1 --out results/expand_$m/checklist_glm.jsonl --workers 8 >>"$LOG" 2>&1; }
    done
    expand_done && { python scripts/compute_buckets.py --glm results/expand_glm/checklist_glm.jsonl \
      --qwen results/expand_qwen/checklist_glm.jsonl --dsv4 results/expand_dsv4/checklist_glm.jsonl \
      --out /tmp/buckets_expand.json --threshold 0.5 >>"$LOG" 2>&1; log "[expand] buckets done"; }
  fi

  # ---------- TRACK 2: v3.2/1969 question-granular relabel (GLM judge; traces exist) ----------
  if ! relabel_done; then
    [ "$(rubn $RUB1969)" -lt "$N1969" ] && { log "[relabel] gen_rubrics_1969_uid ($(rubn $RUB1969)/$N1969)"; \
      python scripts/gen_rubrics.py --data $GOLD --model glm-5.1 --base http://$GN:8000/v1 --out $RUB1969 --workers 8 >>"$LOG" 2>&1; }
    if [ "$(rubn $RUB1969)" -ge "$N1969" ]; then
      for m in glm qwen dsv4; do
        [ "$(cnt results/baseline_${m}_1969/checklist_uid.jsonl)" -lt "$N1969" ] && {
          log "[relabel] re-judge $m ($(cnt results/baseline_${m}_1969/checklist_uid.jsonl)/$N1969)"; \
          python scripts/checklist_judge.py --traces results/baseline_${m}_1969/traces.jsonl \
            --rubrics $RUB1969 --judge openai/glm-5.1 --base http://$GN:8000/v1 --out results/baseline_${m}_1969/checklist_uid.jsonl --workers 8 >>"$LOG" 2>&1; }
      done
      relabel_done && { python scripts/compute_buckets.py --glm results/baseline_glm_1969/checklist_uid.jsonl \
        --qwen results/baseline_qwen_1969/checklist_uid.jsonl --dsv4 results/baseline_dsv4_1969/checklist_uid.jsonl \
        --out /tmp/buckets_1969_uid.json --threshold 0.5 >>"$LOG" 2>&1; log "[relabel] buckets done"; }
    fi
  fi
done

log "build v3.4 (expand_done=$(expand_done && echo y || echo n) relabel_done=$(relabel_done && echo y || echo n))"
python scripts/build_v3_4.py >>"$LOG" 2>&1
for j in $GLM $QWEN $DSV4; do scancel "$j" 2>/dev/null; done
log "scancelled serving $GLM $QWEN $DSV4"
echo "NEXTBATCH_DONE" | tee -a "$LOG"
