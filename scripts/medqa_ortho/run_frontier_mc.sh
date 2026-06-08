#!/usr/bin/env bash
# Frontier MedQA orthogonality via OpenRouter (PAID — review 3x before running).
# Scores Gemini-3-Pro / Opus-4.7 / GPT-5.5 on the same closed-form MC sets, so the scatter
# can show frontier agents also cluster high on MedQA while spreading on OpenBioRQ robust core.
#
# Cost guard: MedQA(1273)+PubMedQA(1000)+MedMCQA(4183)=6456 Q x 3 models = ~19k calls, each
# ~300 tok in / ~30 tok out (think-off not supported -> capped via max_tokens). Subsample with
# --limit to bound spend. DRY-RUN first with --limit 3.
set -u
cd /data/project/private/minstar/workspace/healthcare-research
PY=/data/project/private/minstar/miniconda3/envs/kimi/bin/python
LIMIT="${1:-0}"      # pass an integer to subsample per dataset (0 = full); default full
DS="${2:-MedQA,PubMedQA,MedMCQA}"

# OpenRouter unified OpenAI-compatible endpoint + key (do NOT echo the key)
set -a; source /data/project/private/minstar/settings/.env 2>/dev/null; set +a
: "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY not set}"
BASE="https://openrouter.ai/api/v1"

# OpenRouter slugs (match the OpenBioRQ frontier roster)
declare -A M=(
  [gemini3pro]="google/gemini-3.1-pro-preview"
  [opus47]="anthropic/claude-opus-4.7"
  [gpt55]="openai/gpt-5.5"
)
for tag in gemini3pro opus47 gpt55; do
  [ -f "results/medqa_ortho/${tag}_summary.json" ] && { echo "[$tag] exists, skip"; continue; }
  echo "=== frontier $tag (${M[$tag]}) limit=$LIMIT ==="
  # frontier reasoning is internal; keep-thinking (no enable_thinking kwarg) + modest workers to respect rate limits
  OPENAI_API_KEY="$OPENROUTER_API_KEY" $PY scripts/medqa_ortho/mc_eval.py \
    --base "$BASE" --model "${M[$tag]}" --tag "$tag" --datasets "$DS" \
    --limit "$LIMIT" --workers 8 --keep-thinking
done
echo "=== frontier MC done ==="
