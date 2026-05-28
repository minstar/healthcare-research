#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONDA_BASE="/data/project/private/minstar/miniconda3"

export XDG_CACHE_HOME="/data/project/private/minstar/.cache"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate base

SOURCES="${1:-pubmed,arxiv,medrxiv,cochrane,web}"

echo ">>> Starting medical open-question crawl"
echo ">>> Sources: ${SOURCES}"
echo ">>> Output: ${PROJECT_DIR}/data/raw/"
echo ""

cd "$PROJECT_DIR"
python run.py crawl --sources "$SOURCES"

echo ""
echo ">>> Crawl complete. Documents saved to data/raw/"
echo ">>> Next: run_extract.sh or run_pipeline.sh"
