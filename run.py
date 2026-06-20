"""ResearchMed — Unsolved Medical Question Collection Pipeline.

Main CLI entry point for running the crawl → extract → refine → dedup → export pipeline.

Usage:
    python run.py crawl [--sources pubmed,arxiv,medrxiv,cochrane,web]
    python run.py extract [--model gpt-4.1] [--base-url http://...]
    python run.py refine [--model gpt-4.1]
    python run.py dedup [--threshold 0.90]
    python run.py export [--format mcp,hf,csv]
    python run.py stats
    python run.py all   # full pipeline
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("researchmed")

BASE_DIR = Path(__file__).parent
DEFAULT_CONFIG = BASE_DIR / "config.yaml"


def load_config(config_path: Path = DEFAULT_CONFIG) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


@click.group()
@click.option("--config", default=str(DEFAULT_CONFIG), help="Path to config.yaml")
@click.pass_context
def cli(ctx, config):
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(Path(config))


@cli.command()
@click.option("--sources", default="pubmed,arxiv,medrxiv,cochrane,web,nature,biomedical_api", help="Comma-separated crawler names")
@click.option("--reset-checkpoint", is_flag=True, help="Clear checkpoints and start fresh")
@click.pass_context
def crawl(ctx, sources, reset_checkpoint):
    """Run crawlers to collect raw documents. Automatically resumes from last checkpoint on failure."""
    cfg = ctx.obj["config"]
    raw_dir = Path(cfg["paths"]["raw_dir"])
    source_list = [s.strip() for s in sources.split(",")]

    from crawlers import (
        PubMedCrawler, ArxivCrawler, MedRxivCrawler, CochraneCrawler,
        WebCrawler, NatureCrawler, BiomedicalAPICrawler,
    )

    crawler_map = {
        "pubmed": lambda: PubMedCrawler(
            raw_dir,
            email=cfg["crawlers"]["pubmed"].get("email", ""),
            rate_limit_sec=cfg["crawlers"]["pubmed"].get("rate_limit_sec", 0.4),
        ),
        "arxiv": lambda: ArxivCrawler(raw_dir),
        "medrxiv": lambda: MedRxivCrawler(raw_dir),
        "cochrane": lambda: CochraneCrawler(raw_dir),
        "web": lambda: WebCrawler(raw_dir),
        "nature": lambda: NatureCrawler(
            raw_dir,
            email=cfg["crawlers"]["pubmed"].get("email", ""),
        ),
        "biomedical_api": lambda: BiomedicalAPICrawler(raw_dir),
    }

    total_docs = 0
    for name in source_list:
        if name not in crawler_map:
            logger.warning(f"Unknown source: {name}, skipping")
            continue

        crawler_cfg = cfg["crawlers"].get(name, {})
        if not crawler_cfg.get("enabled", True):
            logger.info(f"[{name}] Disabled in config, skipping")
            continue

        logger.info(f"=== Crawling: {name} ===")
        crawler = crawler_map[name]()

        if reset_checkpoint:
            crawler.checkpoint.clear()
            logger.info(f"[{name}] Checkpoint cleared, starting fresh")

        queries = crawler_cfg.get("queries", [])
        max_results = crawler_cfg.get("max_results", 1000)

        docs = list(crawler.crawl(queries, max_results))
        crawler.save_documents(docs)
        total_docs += len(docs)
        logger.info(f"[{name}] Collected {len(docs)} documents")

    logger.info(f"=== Crawling complete: {total_docs} total documents ===")


@cli.command()
@click.pass_context
def checkpoint_status(ctx):
    """Show checkpoint status for all crawlers."""
    cfg = ctx.obj["config"]
    raw_dir = Path(cfg["paths"]["raw_dir"])
    checkpoint_dir = raw_dir / ".checkpoints"

    if not checkpoint_dir.exists():
        print("No checkpoints found. Run 'crawl' first.")
        return

    for cp_file in sorted(checkpoint_dir.glob("*_checkpoint.json")):
        with open(cp_file) as f:
            state = json.load(f)
        source = state.get("source", cp_file.stem)
        print(f"\n=== {source} ===")
        print(f"  Completed queries: {len(state.get('completed_queries', []))}")
        print(f"  Documents collected: {state.get('total_documents', 0)}")
        if state.get("current_query"):
            print(f"  In-progress query: {state['current_query'][:80]}...")
            print(f"  Cursor: {state.get('current_cursor', 0)}")
        if state.get("last_error"):
            err = state["last_error"]
            print(f"  Last error: {err['message'][:150]}")
            print(f"  Error time: {err.get('timestamp', 'N/A')}")


@cli.command()
@click.option("--model", default=None, help="LLM model name")
@click.option("--base-url", default=None, help="LLM API base URL (for local vLLM)")
@click.option("--api-key", default=None, help="API key")
@click.pass_context
def extract(ctx, model, base_url, api_key):
    """Extract open questions from crawled documents."""
    cfg = ctx.obj["config"]
    raw_dir = Path(cfg["paths"]["raw_dir"])
    extracted_dir = Path(cfg["paths"]["extracted_dir"])

    from pipeline.extractor import QuestionExtractor

    ext_cfg = cfg["pipeline"]["extractor"]
    extractor = QuestionExtractor(
        model=model or ext_cfg["model"],
        base_url=base_url,
        api_key=api_key,
        temperature=ext_cfg.get("temperature", 0.3),
        max_tokens=ext_cfg.get("max_tokens", 8192),
    )

    all_docs = []
    for jsonl_path in sorted(raw_dir.rglob("documents.jsonl")):
        logger.info(f"Loading {jsonl_path}")
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    all_docs.append(json.loads(line))

    if not all_docs:
        logger.error("No documents found. Run 'crawl' first.")
        sys.exit(1)

    logger.info(f"Extracting questions from {len(all_docs)} documents...")
    output_path = extracted_dir / "extracted_questions.jsonl"
    extractor.extract_batch(all_docs, output_path)


@cli.command()
@click.option("--model", default=None, help="LLM model name")
@click.option("--base-url", default=None, help="LLM API base URL")
@click.option("--api-key", default=None, help="API key")
@click.pass_context
def refine(ctx, model, base_url, api_key):
    """Refine extracted questions with taxonomy, status, and MCP tool mapping."""
    cfg = ctx.obj["config"]
    extracted_dir = Path(cfg["paths"]["extracted_dir"])
    refined_dir = Path(cfg["paths"]["refined_dir"])

    from pipeline.refiner import QuestionRefiner

    ref_cfg = cfg["pipeline"]["refiner"]
    refiner = QuestionRefiner(
        model=model or ref_cfg["model"],
        base_url=base_url,
        api_key=api_key,
        temperature=ref_cfg.get("temperature", 0.2),
        max_tokens=ref_cfg.get("max_tokens", 4096),
    )

    input_path = extracted_dir / "extracted_questions.jsonl"
    if not input_path.exists():
        logger.error(f"No extracted questions found at {input_path}. Run 'extract' first.")
        sys.exit(1)

    questions = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    logger.info(f"Refining {len(questions)} questions...")
    output_path = refined_dir / "refined_questions.jsonl"
    refiner.refine_batch(questions, output_path)


@cli.command()
@click.option("--threshold", default=None, type=float, help="Similarity threshold")
@click.pass_context
def dedup(ctx, threshold):
    """Deduplicate refined questions using embeddings."""
    cfg = ctx.obj["config"]
    refined_dir = Path(cfg["paths"]["refined_dir"])
    final_dir = Path(cfg["paths"]["final_dir"])

    from pipeline.dedup import EmbeddingDeduplicator

    dedup_cfg = cfg["pipeline"]["dedup"]
    deduplicator = EmbeddingDeduplicator(
        model_name=dedup_cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
        similarity_threshold=threshold or dedup_cfg.get("similarity_threshold", 0.90),
    )

    input_path = refined_dir / "refined_questions.jsonl"
    if not input_path.exists():
        logger.error(f"No refined questions at {input_path}. Run 'refine' first.")
        sys.exit(1)

    output_path = final_dir / "deduplicated_questions.jsonl"
    deduplicator.deduplicate_file(input_path, output_path)


@cli.command()
@click.option("--format", "fmt", default="mcp,hf,csv", help="Export formats")
@click.pass_context
def export(ctx, fmt):
    """Export final dataset to various formats."""
    cfg = ctx.obj["config"]
    final_dir = Path(cfg["paths"]["final_dir"])

    from evaluation.mcp_format import to_mcp_benchmark, to_hf_dataset, to_analysis_csv

    input_path = final_dir / "deduplicated_questions.jsonl"
    if not input_path.exists():
        logger.error(f"No deduplicated questions at {input_path}. Run 'dedup' first.")
        sys.exit(1)

    questions = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    formats = [f.strip() for f in fmt.split(",")]
    export_dir = final_dir / "exports"

    if "mcp" in formats:
        to_mcp_benchmark(questions, export_dir)
    if "hf" in formats:
        to_hf_dataset(questions, export_dir)
    if "csv" in formats:
        to_analysis_csv(questions, export_dir)


@cli.command()
@click.pass_context
def stats(ctx):
    """Show dataset statistics."""
    cfg = ctx.obj["config"]
    final_dir = Path(cfg["paths"]["final_dir"])

    from evaluation.mcp_format import generate_stats

    input_path = final_dir / "deduplicated_questions.jsonl"
    if not input_path.exists():
        # Try refined
        input_path = Path(cfg["paths"]["refined_dir"]) / "refined_questions.jsonl"
    if not input_path.exists():
        input_path = Path(cfg["paths"]["extracted_dir"]) / "extracted_questions.jsonl"
    if not input_path.exists():
        logger.error("No question data found. Run the pipeline first.")
        sys.exit(1)

    questions = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    s = generate_stats(questions)
    print(json.dumps(s, indent=2, default=str))


@cli.command(name="all")
@click.option("--sources", default="pubmed,arxiv,medrxiv,cochrane,web,nature,biomedical_api")
@click.option("--model", default=None)
@click.option("--base-url", default=None)
@click.option("--api-key", default=None)
@click.option("--reset-checkpoint", is_flag=True)
@click.pass_context
def run_all(ctx, sources, model, base_url, api_key, reset_checkpoint):
    """Run the full pipeline: crawl → extract → refine → dedup → export."""
    ctx.invoke(crawl, sources=sources, reset_checkpoint=reset_checkpoint)
    ctx.invoke(extract, model=model, base_url=base_url, api_key=api_key)
    ctx.invoke(refine, model=model, base_url=base_url, api_key=api_key)
    ctx.invoke(dedup)
    ctx.invoke(export, fmt="mcp,hf,csv")
    ctx.invoke(stats)


if __name__ == "__main__":
    cli()
