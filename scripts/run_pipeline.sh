#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONDA_BASE="/data/project/private/minstar/miniconda3"

export XDG_CACHE_HOME="/data/project/private/minstar/.cache"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate base

# LLM configuration — override via env vars
MODEL="${MODEL:-gpt-4.1}"
BASE_URL="${BASE_URL:-}"           # set for local vLLM, e.g. http://127.0.0.1:8000/v1
API_KEY="${API_KEY:-}"
SOURCES="${SOURCES:-pubmed,arxiv,medrxiv,cochrane,web}"
DEDUP_THRESHOLD="${DEDUP_THRESHOLD:-0.90}"
EXPORT_FORMATS="${EXPORT_FORMATS:-mcp,hf,csv}"

cd "$PROJECT_DIR"

echo "==========================================="
echo "  ResearchMed — Full Pipeline"
echo "==========================================="
echo "Model:      ${MODEL}"
echo "Base URL:   ${BASE_URL:-OpenAI default}"
echo "Sources:    ${SOURCES}"
echo "Dedup:      threshold=${DEDUP_THRESHOLD}"
echo "Export:     ${EXPORT_FORMATS}"
echo "==========================================="
echo ""

MODEL_ARGS="--model ${MODEL}"
[[ -n "$BASE_URL" ]] && MODEL_ARGS="${MODEL_ARGS} --base-url ${BASE_URL}"
[[ -n "$API_KEY" ]] && MODEL_ARGS="${MODEL_ARGS} --api-key ${API_KEY}"

echo ">>> Step 1/5: Crawling sources..."
python run.py crawl --sources "$SOURCES"

echo ""
echo ">>> Step 2/5: Extracting open questions..."
python run.py extract ${MODEL_ARGS}

echo ""
echo ">>> Step 3/5: Refining questions (taxonomy + status + MCP tools)..."
python run.py refine ${MODEL_ARGS}

echo ""
echo ">>> Step 4/5: Deduplicating..."
python run.py dedup --threshold "$DEDUP_THRESHOLD"

echo ""
echo ">>> Step 5/5: Exporting..."
python run.py export --format "$EXPORT_FORMATS"

echo ""
echo ">>> Statistics:"
python run.py stats

echo ""
echo "==========================================="
echo "  Pipeline complete!"
echo "  Output: ${PROJECT_DIR}/data/final/"
echo "==========================================="
