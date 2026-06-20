#!/usr/bin/env bash
# MedQA-orthogonality (Reviewer E): score each open-weight OpenBioRQ model on closed-form
# MedQA / PubMedQA / MedMCQA, one 8xH200 NORMAL-partition node at a time (serve -> wait ->
# eval all 3 sets -> scancel -> next). Resumable: skips a model whose summary already exists.
#
# Frontier models (Gemini/Opus/GPT-5.5) are scored separately via OpenRouter (run_frontier_mc.sh).
set -u
cd /data/project/private/minstar/workspace/healthcare-research
PY=/data/project/private/minstar/miniconda3/envs/kimi/bin/python
LOG=results/medqa_ortho/run.log
mkdir -p results/medqa_ortho
log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG" >&2; }

# tag | jobname | served-name | serve-script
MODELS=(
  "glm51|serve-glm51|glm-5.1|serving/serve_glm51.slurm"
  "qwen36|serve-qwen36|qwen-3.6|serving/serve_qwen36.slurm"
  "dsv4|serve-dsv4flash|deepseek-v4-flash|/data/project/private/minstar/settings/serving/deepseek_v4_flash_vllm.sh"
  "glm5|serve-glm5|glm-5|serving/serve_glm5.slurm"
  "qwen35_397b|serve-qwen35-397b|qwen3.5-397b|serving/serve_qwen35_397b.slurm"
  "qwen3_235b|serve-qwen3-235b|qwen3-235b|serving/serve_qwen3_235b.slurm"
)
jid_of(){ squeue -u minstar -h -n "$1" -o '%i' 2>/dev/null | head -1; }

for entry in "${MODELS[@]}"; do
  IFS='|' read -r tag jn served script <<< "$entry"
  if [ -f "results/medqa_ortho/${tag}_summary.json" ]; then
    log "[$tag] summary exists -> skip"; continue
  fi
  log "=== $tag (served=$served) ==="
  # (re)submit serve on normal if not present
  jid=$(jid_of "$jn")
  [ -z "$jid" ] && { jid=$(sbatch --parsable "$script"); log "[$tag] submitted $jn -> $jid"; }
  # wait for endpoint up
  node=""
  while :; do
    jid=$(jid_of "$jn")
    [ -z "$jid" ] && { jid=$(sbatch --parsable "$script"); log "[$tag] re-submitted -> $jid"; sleep 20; continue; }
    st=$(squeue -j "$jid" -h -o '%T' 2>/dev/null)
    if [ "$st" = "RUNNING" ]; then
      node=$(squeue -j "$jid" -h -o '%N' 2>/dev/null)
      if [ -n "$node" ] && curl -s --max-time 4 "http://$node:8000/v1/models" 2>/dev/null | grep -q "$served"; then
        log "[$tag] endpoint up @ $node"; break
      fi
    fi
    sleep 30
  done
  # eval (all three datasets, full)
  log "[$tag] eval start @ $node"
  OPENAI_API_KEY=dummy $PY scripts/medqa_ortho/mc_eval.py \
    --base "http://$node:8000/v1" --model "$served" --tag "$tag" --workers 24 \
    >> "$LOG" 2>&1
  if [ -f "results/medqa_ortho/${tag}_summary.json" ]; then
    log "[$tag] eval done"
    jid=$(jid_of "$jn"); [ -n "$jid" ] && scancel "$jid" && log "[$tag] scancel $jn ($jid) -> node freed"
  else
    log "[$tag] eval FAILED (no summary) — leaving serve up for inspection"; exit 1
  fi
done
log "=== ALL OPEN-WEIGHT MODELS DONE ==="
ls -1 results/medqa_ortho/*_summary.json | tee -a "$LOG"
