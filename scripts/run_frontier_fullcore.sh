#!/usr/bin/env bash
# ===========================================================================
# MONEY SCRIPT (paid API). Reviewed 3x before launch (user mandate 2026-06-05).
# Table-3 fill: frontier agentic leaderboard on the FULL 657 core, to populate
# the empty Full-core(657) cells (Gemini-3-Pro, Opus-4.7, GPT-5.5, GPT-5.5 no-tools).
#
# Cost-saving design: the 4 *_fullcore dirs are pre-seeded with the paid frozen-423
# traces. The harness resumes by task_id (only non-[ERROR], non-empty count as done),
# so each config runs ONLY the 234 complement (657-423) -> ~$350 total, not a re-run.
#
# ALL THREE families routed through OpenRouter (single funded key, ~$925 balance):
#   GPT-5.5  = openrouter/openai/gpt-5.5
#   Opus-4.7 = openrouter/anthropic/claude-opus-4.7
#   Gemini   = openrouter/google/gemini-3.1-pro-preview
# Same model weights as the frozen-423 (direct-provider) traces, so the 423+234
# combine into one 657 leaderboard is valid; provider routing is an impl detail.
#
# Trace generation ONLY (the paid part). Judging is local GLM-5.1 (FREE), separate.
# Resumable: traces.jsonl is incremental; a re-run resumes and retries [ERROR]/empty.
# PIDs written to results/frontier_fullcore.pids for the cost watchdog.
# ===========================================================================
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-kimi}"
# real provider keys; DROP any Anthropic proxy so everything routes via OpenRouter
set -a; source "${ENV_FILE:-$HOME/.env}" 2>/dev/null; set +a
unset ANTHROPIC_BASE_URL ANTHROPIC_AUTH_TOKEN ANTHROPIC_MODEL
[ -n "${OPENROUTER_API_KEY:-}" ] || { echo "ABORT: OPENROUTER_API_KEY not set"; exit 1; }

DATA=data/eval_samples/core_eval.jsonl
N=$(wc -l < "$DATA")
LOG=results/frontier_fullcore.log
PIDFILE=results/frontier_fullcore.pids
: > "$LOG"; : > "$PIDFILE"
log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG" >&2; }
[ "$N" -eq 657 ] || { log "ABORT: expected 657 full-core rows, got $N"; exit 1; }
log "full core n=$N | launching 3 with-tools + 1 no-tools via OpenRouter (seeded with frozen-423; ~234 new each)"

# $1=tag $2=litellm-model $3=extra-flags
gen(){ local tag=$1 model=$2 extra=${3:-}
  local out=results/${tag}
  local have; have=$(wc -l < "$out/traces.jsonl" 2>/dev/null || echo 0)
  if [ "$have" -ge "$N" ]; then log "[skip] $tag already $have/$N"; return; fi
  log "[gen] $tag ($model) $extra  (seeded $have/$N -> $((N-have)) new)"
  python harness/run.py --data "$DATA" --model "litellm::$model" --no-judge \
    --workers 5 $extra --output "$out" >> "$LOG" 2>&1 &
  local pid=$!
  echo "$pid $tag" >> "$PIDFILE"
  wait "$pid"
  log "[done-gen] $tag -> $(wc -l < "$out/traces.jsonl" 2>/dev/null || echo 0)/$N"
}

gen api_gpt55_fullcore        "openrouter/openai/gpt-5.5"               "" &
gen api_opus47_fullcore       "openrouter/anthropic/claude-opus-4.7"   "" &
gen api_gemini3pro_fullcore   "openrouter/google/gemini-3.1-pro-preview" "" &
gen api_gpt55_notool_fullcore "openrouter/openai/gpt-5.5"              "--no-tools" &
wait

log "=== trace generation complete ==="
for t in api_gpt55_fullcore api_opus47_fullcore api_gemini3pro_fullcore api_gpt55_notool_fullcore; do
  log "  $t: $(wc -l < results/$t/traces.jsonl 2>/dev/null || echo 0)/$N"
done
echo "FRONTIER_FULLCORE_GEN_DONE" | tee -a "$LOG"
