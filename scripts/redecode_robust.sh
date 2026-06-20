#!/usr/bin/env bash
# Reproducibility re-decode (Reviewer C): a SECOND independent T=0 decode of all three roster
# models over the 423 robust-core questions, days after the first, to test whether robust-core
# membership is stable (the paper's robust core rests on a single T=0 sample). One 8xH200
# NORMAL node at a time: serve -> generate traces (T=0) -> scancel -> next; then serve GLM-5.1,
# judge all three re-decode trace sets, and run the seed1-vs-seed2 comparison.
set -u
cd /data/project/private/minstar/workspace/healthcare-research
source /data/project/private/minstar/miniconda3/etc/profile.d/conda.sh
conda activate kimi
export OPENAI_API_KEY=dummy EVAL_TEMPERATURE=0.0
DATA=data/eval_samples/robust_core_boundary.jsonl  # 131 boundary-proximal (max-of-3>=0.4); the only flip-capable members
RUB=data/eval_samples/rubrics_1969_uid.jsonl
N=$(wc -l < "$DATA")
LOG=results/redecode/run.log
mkdir -p results/redecode
log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG" >&2; }
jid_of(){ squeue -u minstar -h -n "$1" -o '%i' 2>/dev/null | head -1; }
traces(){ wc -l < "results/$1/traces.jsonl" 2>/dev/null || echo 0; }
DSV4=/data/project/private/minstar/settings/serving/deepseek_v4_flash_vllm.sh

# tag | jobname | served | serve-script
GEN=(
  "redecode_glm51|serve-glm51|glm-5.1|serving/serve_glm51.slurm"
  "redecode_qwen36|serve-qwen36|qwen-3.6|serving/serve_qwen36.slurm"
  "redecode_dsv4|serve-dsv4flash|deepseek-v4-flash|$DSV4"
)

# ---- phase 1: generate second-seed traces, one model/node at a time ----
for entry in "${GEN[@]}"; do
  IFS='|' read -r tag jn served script <<< "$entry"
  if [ "$(traces $tag)" -ge "$N" ]; then log "[$tag] traces complete -> skip"; continue; fi
  log "=== generate $tag (served=$served) ==="
  node=""
  while [ "$(traces $tag)" -lt "$N" ]; do
    jid=$(jid_of "$jn")
    [ -z "$jid" ] && { jid=$(sbatch --parsable "$script"); log "[$tag] submitted $jn -> $jid"; sleep 20; continue; }
    st=$(squeue -j "$jid" -h -o '%T' 2>/dev/null)
    if [ "$st" = "RUNNING" ]; then
      node=$(squeue -j "$jid" -h -o '%N' 2>/dev/null)
      if [ -n "$node" ] && curl -s --max-time 4 "http://$node:8000/v1/models" 2>/dev/null | grep -q "$served"; then
        log "[$tag] endpoint up @ $node; generating (T=0, 423 Q)"
        OPENAI_API_BASE=http://$node:8000/v1 python harness/run.py --data "$DATA" \
          --model "openai::http://$node:8000/v1::$served" --no-judge --workers 12 \
          --output results/$tag >> "$LOG" 2>&1
        continue   # re-check count (handles mid-run interruption -> resume)
      fi
    fi
    sleep 45
  done
  jid=$(jid_of "$jn"); [ -n "$jid" ] && scancel "$jid" && log "[$tag] scancel $jn -> node freed ($(traces $tag) traces)"
done

# ---- phase 2: serve GLM-5.1, judge all three re-decode trace sets ----
log "=== judging re-decode traces (GLM-5.1 judge) ==="
GN=""
while [ -z "$GN" ]; do
  jid=$(jid_of serve-glm51)
  [ -z "$jid" ] && { jid=$(sbatch --parsable serving/serve_glm51.slurm); log "submitted GLM judge $jid"; sleep 20; continue; }
  node=$(squeue -j "$jid" -h -o '%N' 2>/dev/null)
  [ "$(squeue -j "$jid" -h -o '%T')" = RUNNING ] && [ -n "$node" ] && \
    curl -s --max-time 4 "http://$node:8000/v1/models" 2>/dev/null | grep -q glm-5.1 && GN=$node || sleep 45
done
log "GLM-5.1 judge @ $GN"
cnt(){ python - "$1" <<'PY'
import json,sys
try: print(sum(1 for l in open(sys.argv[1]) if json.loads(l).get("checklist_score") is not None))
except FileNotFoundError: print(0)
PY
}
for tag in redecode_glm51 redecode_qwen36 redecode_dsv4; do
  [ "$(cnt results/$tag/checklist_glm.jsonl)" -ge "$N" ] && { log "[judge] $tag already done"; continue; }
  log "[judge] $tag"
  python scripts/checklist_judge.py --traces results/$tag/traces.jsonl \
    --rubrics "$RUB" --judge openai/glm-5.1 --base "http://$GN:8000/v1" \
    --out results/$tag/checklist_glm.jsonl --workers 8 2>&1 | grep -ivE httpx >> "$LOG"
done
jid=$(jid_of serve-glm51); [ -n "$jid" ] && scancel "$jid" && log "scancel GLM-5.1 judge (re-decode judging done)"

# ---- phase 3: compare seed1 vs seed2 ----
conda activate base
python scripts/redecode_compare.py 2>&1 | tee -a "$LOG"
echo "REDECODE_DONE" | tee -a "$LOG"
