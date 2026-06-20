#!/usr/bin/env bash
# ===========================================================================
# MONEY SCRIPT (paid API). Reviewed 3x before launch (user mandate 2026-06-05).
# Track-C #1: frontier agentic leaderboard on the 423 ROBUST CORE, with tools.
#   3 out-of-lineage families in parallel: GPT-5.5, Opus-4.7, Gemini-3-Pro.
# Track-C #4: GPT-5.5 --no-tools (zero-tool ablation) on the same 423.
# Trace generation ONLY (the paid part). Judging by local GLM-5.1 is FREE and
# runs separately (judge_frontier_robust.sh) once a GPU judge is up.
# Resumable: harness writes traces.jsonl incrementally; reruns skip done tasks.
# ===========================================================================
set -u
cd /data/project/private/minstar/workspace/healthcare-research
source /data/project/private/minstar/miniconda3/etc/profile.d/conda.sh
conda activate kimi
# real provider keys; DROP the Anthropic proxy so Opus goes through OpenRouter
set -a; source /data/project/private/minstar/.env 2>/dev/null; set +a
unset ANTHROPIC_BASE_URL ANTHROPIC_AUTH_TOKEN ANTHROPIC_MODEL

DATA=data/eval_samples/robust_core_eval.jsonl
N=$(wc -l < "$DATA")
LOG=results/frontier_robust.log
: > "$LOG"
log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG" >&2; }
[ "$N" -eq 423 ] || { log "ABORT: expected 423 robust-core rows, got $N"; exit 1; }
log "robust core n=$N | launching 3 with-tools + 1 no-tools in parallel"

# $1=tag $2=litellm-model $3=extra-flags
gen(){ local tag=$1 model=$2 extra=${3:-}
  local out=results/${tag}
  local have; have=$(wc -l < "$out/traces.jsonl" 2>/dev/null || echo 0)
  if [ "$have" -ge "$N" ]; then log "[skip] $tag already $have/$N"; return; fi
  log "[gen] $tag ($model) $extra  (have $have/$N)"
  python harness/run.py --data "$DATA" --model "litellm::$model" --no-judge \
    --workers 5 $extra --output "$out" >> "$LOG" 2>&1
  log "[done-gen] $tag -> $(wc -l < "$out/traces.jsonl" 2>/dev/null || echo 0)/$N"
}

gen api_gpt55_robust        "gpt-5.5"                                "" &
gen api_opus47_robust       "openrouter/anthropic/claude-opus-4.7"   "" &
gen api_gemini3pro_robust   "gemini/gemini-3.1-pro-preview"          "" &
gen api_gpt55_notool_robust "gpt-5.5"                                "--no-tools" &
wait

log "=== trace generation complete ==="
for t in api_gpt55_robust api_opus47_robust api_gemini3pro_robust api_gpt55_notool_robust; do
  log "  $t: $(wc -l < results/$t/traces.jsonl 2>/dev/null || echo 0)/$N"
done
echo "FRONTIER_ROBUST_GEN_DONE" | tee -a "$LOG"
