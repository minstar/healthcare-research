#!/usr/bin/env python3
"""CLI entry point for the healthcare open-questions benchmark harness.

Usage:
    python harness/run.py \
        --data data/export/mcp_benchmark_with_gold.jsonl \
        --model openai::http://localhost:8000/v1::solar-open-2-10b \
        --judge gemini/gemini-2.5-pro \
        --taxonomy-filter "Oncology" \
        --difficulty-min 3 \
        --limit 50 \
        --workers 4 \
        --output results/run_001/
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

# Ensure the project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from harness.task_loader import load_tasks
from harness.mcp_tools import MCPToolRegistry
from harness.completion_runner import run_tasks
from harness.judge import judge_single, EnsembleJudge
from harness.metrics import compute_metrics, save_results

logger = logging.getLogger("harness")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Healthcare Open-Questions Benchmark Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to benchmark JSONL file (e.g. data/export/mcp_benchmark_with_gold.jsonl)",
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=(
            "Model specification. Formats:\n"
            "  openai::http://localhost:8000/v1::model-name\n"
            "  litellm::provider/model\n"
            "  claude::model-name"
        ),
    )

    # Judge
    parser.add_argument(
        "--judge",
        type=str,
        default="gemini/gemini-2.5-pro",
        help=(
            "LiteLLM model string(s) for the judge. Use comma-separated list for "
            "multi-judge ensemble, e.g. 'gemini/gemini-2.5-pro,gpt-4o,claude-opus-4-20250514'. "
            "Single model uses the standard judge; multiple models use EnsembleJudge. "
            "(default: gemini/gemini-2.5-pro)"
        ),
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip judging (useful when no gold answers are available)",
    )

    # Filters
    parser.add_argument("--taxonomy-filter", type=str, default=None, help="Filter by taxonomy L1 (substring match)")
    parser.add_argument("--difficulty-min", type=float, default=None, help="Minimum average difficulty (1-5)")
    parser.add_argument("--difficulty-max", type=float, default=None, help="Maximum average difficulty (1-5)")
    parser.add_argument("--open-status", type=str, default=None, help="Filter by open_status (e.g. 'open')")
    parser.add_argument(
        "--required-tools",
        type=str,
        nargs="+",
        default=None,
        help="Only include tasks that require ALL of these tools",
    )

    # Sampling
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of tasks to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling (default: 42)")

    # Execution
    parser.add_argument("--workers", type=int, default=4, help="Max concurrent tasks (default: 4)")
    parser.add_argument("--judge-workers", type=int, default=4, help="Max concurrent judge calls (default: 4)")
    parser.add_argument("--pass-threshold", type=float, default=0.60, help="Pass/fail score threshold (default: 0.60)")

    # Output
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for results (e.g. results/run_001/)",
    )

    return parser.parse_args()


async def main_async(args: argparse.Namespace) -> None:
    t_start = time.monotonic()

    # ---- 1. Load tasks ----
    logger.info("Loading tasks from %s", args.data)
    tasks = load_tasks(
        args.data,
        taxonomy_l1=args.taxonomy_filter,
        difficulty_min=args.difficulty_min,
        difficulty_max=args.difficulty_max,
        open_status=args.open_status,
        required_tools=args.required_tools,
        limit=args.limit,
        seed=args.seed,
    )
    if not tasks:
        logger.error("No tasks matched the filters. Exiting.")
        sys.exit(1)

    logger.info("Loaded %d tasks", len(tasks))

    # Count tasks with gold answers
    n_with_gold = sum(1 for t in tasks if t.gold_answer is not None)
    logger.info("Tasks with gold answers: %d/%d", n_with_gold, len(tasks))

    # ---- 2. Run completions ----
    logger.info("Running completions with model: %s (workers=%d)", args.model, args.workers)
    registry = MCPToolRegistry()
    # incremental checkpoint = the same traces.jsonl metrics will consolidate at the end;
    # lets a preempted run resume instead of redoing completions.
    ckpt = str(Path(args.output) / "traces.jsonl")
    completion_results = await run_tasks(tasks, args.model, workers=args.workers,
                                         registry=registry, checkpoint_path=ckpt)
    logger.info("Completed %d/%d tasks", len(completion_results), len(tasks))

    # ---- 3. Judge (if gold answers available and not skipped) ----
    judge_results = None
    ensemble_results = None
    batch_kappa = None
    should_judge = not args.no_judge and n_with_gold > 0

    # Parse judge models (comma-separated for ensemble)
    judge_models = [m.strip() for m in args.judge.split(",") if m.strip()]
    use_ensemble = len(judge_models) > 1

    if should_judge:
        if use_ensemble:
            logger.info(
                "Ensemble judging %d tasks with %d judges: %s",
                n_with_gold, len(judge_models), ", ".join(judge_models),
            )
        else:
            logger.info("Judging %d tasks with %s", n_with_gold, judge_models[0])

        # Build judge inputs (only for tasks that have gold answers)
        task_map = {t.task_id: t for t in tasks}
        cr_map = {cr.task_id: cr for cr in completion_results}

        judge_inputs = []
        for task in tasks:
            if task.gold_answer is None:
                continue
            cr = cr_map.get(task.task_id)
            if cr is None:
                continue
            judge_inputs.append(
                {
                    "task_id": task.task_id,
                    "question": task.question,
                    "gold_answer": task.gold_answer,
                    "model_answer": cr.model_answer,
                    "tool_calls": cr.tool_calls,
                }
            )

        if judge_inputs:
            if use_ensemble:
                # Multi-judge ensemble
                ensemble_judge = EnsembleJudge(
                    judge_models=judge_models,
                    pass_threshold=args.pass_threshold,
                )
                ensemble_results, batch_kappa = await ensemble_judge.score_batch(
                    items=judge_inputs,
                    workers=args.judge_workers,
                )
                # Also build judge_results from ensemble medians for
                # backward-compatible metrics (JudgeResult per task).
                from harness.judge import JudgeResult, _compute_weighted_total

                judge_results = []
                for er in ensemble_results:
                    # Use the first individual result's task_id
                    task_id = er.individual_results[0].task_id
                    jr = JudgeResult(
                        task_id=task_id,
                        scores=dict(er.median_scores),
                        weighted_total=er.main_score,
                        reasoning=(
                            f"Ensemble ({len(judge_models)} judges). "
                            f"Kappa={er.cohens_kappa:.3f}. "
                            + er.individual_results[0].reasoning[:300]
                        ),
                    )
                    judge_results.append(jr)
                logger.info(
                    "Ensemble judged %d tasks (batch kappa=%.3f)",
                    len(ensemble_results), batch_kappa,
                )
            else:
                # Single judge (original path)
                semaphore = asyncio.Semaphore(args.judge_workers)
                judge_results_list: list = [None] * len(judge_inputs)

                async def _judge_worker(idx: int, item: dict) -> None:
                    async with semaphore:
                        jr = await judge_single(
                            task_id=item["task_id"],
                            question=item["question"],
                            gold_answer=item["gold_answer"],
                            model_answer=item["model_answer"],
                            tool_calls=item["tool_calls"],
                            judge_model=judge_models[0],
                        )
                        judge_results_list[idx] = jr

                await asyncio.gather(*[_judge_worker(i, inp) for i, inp in enumerate(judge_inputs)])
                judge_results = [jr for jr in judge_results_list if jr is not None]
                logger.info("Judged %d tasks", len(judge_results))
    else:
        if args.no_judge:
            logger.info("Judging skipped (--no-judge flag)")
        else:
            logger.info("No gold answers available; skipping judging")

    # ---- 4. Compute metrics ----
    logger.info("Computing metrics")
    summary = compute_metrics(
        tasks=tasks,
        completion_results=completion_results,
        judge_results=judge_results,
        pass_threshold=args.pass_threshold,
    )

    # Add run metadata
    summary["run_metadata"] = {
        "data_path": str(args.data),
        "model": args.model,
        "judge_model": args.judge if should_judge else None,
        "filters": {
            "taxonomy_filter": args.taxonomy_filter,
            "difficulty_min": args.difficulty_min,
            "difficulty_max": args.difficulty_max,
            "open_status": args.open_status,
            "required_tools": args.required_tools,
        },
        "limit": args.limit,
        "seed": args.seed,
        "workers": args.workers,
        "pass_threshold": args.pass_threshold,
        "total_wall_time": round(time.monotonic() - t_start, 2),
    }

    # ---- 5. Save results ----
    logger.info("Saving results to %s", args.output)
    save_results(
        output_dir=args.output,
        tasks=tasks,
        completion_results=completion_results,
        judge_results=judge_results,
        summary=summary,
    )

    # ---- 6. Print summary ----
    _print_summary(summary)


def _print_summary(summary: dict) -> None:
    """Print a human-readable summary to stdout."""
    print("\n" + "=" * 70)
    print("HEALTHCARE OPEN-QUESTIONS BENCHMARK RESULTS")
    print("=" * 70)

    print(f"\nTasks: {summary['n_tasks']} total, {summary['n_completed']} completed, {summary['n_judged']} judged")

    if summary.get("overall_weighted_score") is not None:
        print(f"\nOverall Weighted Score: {summary['overall_weighted_score']:.4f}")
        print(f"Pass Rate (@{summary['pass_threshold']:.2f}): {summary['pass_rate']:.2%} ({summary['passed']}/{summary['n_judged']})")

        print("\nPer-Dimension Averages:")
        for dim, avg in summary["per_dimension_averages"].items():
            print(f"  {dim:25s}: {avg:.4f}")

    if summary.get("taxonomy_breakdown"):
        print("\nTaxonomy Breakdown:")
        print(f"  {'Category':40s} {'Count':>6s} {'Judged':>7s} {'AvgScore':>9s} {'PassRate':>9s}")
        print(f"  {'-'*40} {'-'*6} {'-'*7} {'-'*9} {'-'*9}")
        for l1, info in sorted(summary["taxonomy_breakdown"].items()):
            avg_s = f"{info['avg_weighted']:.4f}" if info["avg_weighted"] is not None else "N/A"
            pr_s = f"{info['pass_rate']:.2%}" if info["pass_rate"] is not None else "N/A"
            print(f"  {l1:40s} {info['count']:>6d} {info['judged']:>7d} {avg_s:>9s} {pr_s:>9s}")

    if summary.get("difficulty_breakdown"):
        print("\nDifficulty Breakdown:")
        for bucket in ["easy", "medium", "hard"]:
            info = summary["difficulty_breakdown"].get(bucket, {})
            avg_s = f"{info.get('avg_weighted', 0):.4f}" if info.get("avg_weighted") is not None else "N/A"
            pr_s = f"{info.get('pass_rate', 0):.2%}" if info.get("pass_rate") is not None else "N/A"
            print(f"  {bucket:10s}: {info.get('count', 0):>4d} tasks, avg={avg_s}, pass_rate={pr_s}")

    tool_stats = summary.get("tool_stats", {})
    if tool_stats:
        print(f"\nTool Usage: {tool_stats['total_tool_calls']} total calls, {tool_stats['avg_calls_per_task']:.1f} avg/task")
        if tool_stats.get("avg_suggested_tool_overlap") is not None:
            print(f"  Suggested tool overlap: {tool_stats['avg_suggested_tool_overlap']:.2%}")
        if tool_stats.get("tool_frequency"):
            print("  Tool frequency:")
            for tool, count in sorted(tool_stats["tool_frequency"].items(), key=lambda x: -x[1]):
                print(f"    {tool:25s}: {count}")

    timing = summary.get("timing_stats", {})
    if timing:
        print(f"\nTiming: {timing.get('total_wall_time', 0):.1f}s total, {timing.get('avg_wall_time', 0):.1f}s avg/task")
        print(f"  Tokens: {timing.get('total_prompt_tokens', 0)} prompt + {timing.get('total_completion_tokens', 0)} completion")

    print("\n" + "=" * 70)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
