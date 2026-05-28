"""Aggregate evaluation results into benchmark metrics."""
from __future__ import annotations

import csv
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Difficulty buckets
_DIFFICULTY_BUCKETS = {
    "easy": (1.0, 2.5),
    "medium": (2.5, 3.5),
    "hard": (3.5, 5.1),
}


def _avg_difficulty(difficulty: dict) -> float:
    vals = [v for v in difficulty.values() if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else 3.0


def _bucket_label(avg_diff: float) -> str:
    for label, (lo, hi) in _DIFFICULTY_BUCKETS.items():
        if lo <= avg_diff < hi:
            return label
    return "hard"


def compute_metrics(
    tasks: list[Any],
    completion_results: list[Any],
    judge_results: list[Any] | None = None,
    pass_threshold: float = 0.60,
) -> dict:
    """Compute aggregate metrics from evaluation results.

    Args:
        tasks: List of Task objects (from task_loader).
        completion_results: List of CompletionResult objects.
        judge_results: List of JudgeResult objects (optional, may be None if no gold answers).
        pass_threshold: Weighted score threshold for pass/fail classification.

    Returns:
        Dict containing all summary metrics.
    """
    # Build lookup dicts
    task_map = {t.task_id: t for t in tasks}
    completion_map = {cr.task_id: cr for cr in completion_results}
    judge_map = {}
    if judge_results:
        judge_map = {jr.task_id: jr for jr in judge_results}

    n_tasks = len(tasks)
    n_completed = len(completion_results)
    n_judged = len(judge_map)

    # ------------------------------------------------------------------
    # 1. Overall weighted score (only if judging happened)
    # ------------------------------------------------------------------
    overall_weighted = 0.0
    per_dim_sums: dict[str, float] = defaultdict(float)
    per_dim_counts: dict[str, int] = defaultdict(int)

    if judge_map:
        weighted_scores = [jr.weighted_total for jr in judge_map.values()]
        overall_weighted = sum(weighted_scores) / len(weighted_scores) if weighted_scores else 0.0

        for jr in judge_map.values():
            for dim, score in jr.scores.items():
                per_dim_sums[dim] += score
                per_dim_counts[dim] += 1

    per_dim_averages = {
        dim: round(per_dim_sums[dim] / per_dim_counts[dim], 4) if per_dim_counts[dim] else 0.0
        for dim in ["coverage", "evidence_quality", "tool_usage", "reasoning_quality", "completeness"]
    }

    # ------------------------------------------------------------------
    # 2. Breakdown by taxonomy_l1
    # ------------------------------------------------------------------
    taxonomy_scores: dict[str, list[float]] = defaultdict(list)
    taxonomy_counts: dict[str, int] = defaultdict(int)

    for task in tasks:
        l1 = task.taxonomy.get("l1", "Unknown")
        taxonomy_counts[l1] += 1
        if task.task_id in judge_map:
            taxonomy_scores[l1].append(judge_map[task.task_id].weighted_total)

    taxonomy_breakdown = {}
    for l1 in sorted(taxonomy_counts.keys()):
        scores = taxonomy_scores.get(l1, [])
        taxonomy_breakdown[l1] = {
            "count": taxonomy_counts[l1],
            "judged": len(scores),
            "avg_weighted": round(sum(scores) / len(scores), 4) if scores else None,
            "pass_rate": round(sum(1 for s in scores if s >= pass_threshold) / len(scores), 4) if scores else None,
        }

    # ------------------------------------------------------------------
    # 3. Breakdown by difficulty bucket
    # ------------------------------------------------------------------
    difficulty_scores: dict[str, list[float]] = defaultdict(list)
    difficulty_counts: dict[str, int] = defaultdict(int)

    for task in tasks:
        bucket = _bucket_label(_avg_difficulty(task.difficulty))
        difficulty_counts[bucket] += 1
        if task.task_id in judge_map:
            difficulty_scores[bucket].append(judge_map[task.task_id].weighted_total)

    difficulty_breakdown = {}
    for bucket in ["easy", "medium", "hard"]:
        scores = difficulty_scores.get(bucket, [])
        difficulty_breakdown[bucket] = {
            "count": difficulty_counts.get(bucket, 0),
            "judged": len(scores),
            "avg_weighted": round(sum(scores) / len(scores), 4) if scores else None,
            "pass_rate": round(sum(1 for s in scores if s >= pass_threshold) / len(scores), 4) if scores else None,
        }

    # ------------------------------------------------------------------
    # 4. Tool usage statistics
    # ------------------------------------------------------------------
    tool_call_counter: Counter = Counter()
    tool_overlap_scores: list[float] = []
    total_tool_calls = 0

    for cr in completion_results:
        task = task_map.get(cr.task_id)
        called_tools = set()
        for tc in cr.tool_calls:
            tool_name = tc.get("tool", "").replace("-", "_")
            tool_call_counter[tool_name] += 1
            total_tool_calls += 1
            called_tools.add(tool_name)

        # Overlap with suggested tools
        if task and task.suggested_tools:
            suggested = set(t.replace("-", "_") for t in task.suggested_tools)
            if suggested:
                overlap = len(called_tools & suggested) / len(suggested)
                tool_overlap_scores.append(overlap)

    tool_stats = {
        "total_tool_calls": total_tool_calls,
        "avg_calls_per_task": round(total_tool_calls / n_completed, 2) if n_completed else 0,
        "tool_frequency": dict(tool_call_counter.most_common()),
        "avg_suggested_tool_overlap": (
            round(sum(tool_overlap_scores) / len(tool_overlap_scores), 4)
            if tool_overlap_scores
            else None
        ),
    }

    # ------------------------------------------------------------------
    # 5. Pass rate
    # ------------------------------------------------------------------
    if judge_map:
        passed = sum(1 for jr in judge_map.values() if jr.weighted_total >= pass_threshold)
        pass_rate = round(passed / len(judge_map), 4) if judge_map else 0.0
    else:
        passed = 0
        pass_rate = None

    # ------------------------------------------------------------------
    # 6. Timing and token stats
    # ------------------------------------------------------------------
    wall_times = [cr.wall_time for cr in completion_results]
    prompt_tokens = [cr.tokens.get("prompt", 0) for cr in completion_results]
    completion_tokens = [cr.tokens.get("completion", 0) for cr in completion_results]

    timing_stats = {
        "total_wall_time": round(sum(wall_times), 2),
        "avg_wall_time": round(sum(wall_times) / len(wall_times), 2) if wall_times else 0,
        "max_wall_time": round(max(wall_times), 2) if wall_times else 0,
        "total_prompt_tokens": sum(prompt_tokens),
        "total_completion_tokens": sum(completion_tokens),
    }

    # ------------------------------------------------------------------
    # Assemble summary
    # ------------------------------------------------------------------
    summary = {
        "n_tasks": n_tasks,
        "n_completed": n_completed,
        "n_judged": n_judged,
        "pass_threshold": pass_threshold,
        "overall_weighted_score": round(overall_weighted, 4) if judge_map else None,
        "pass_rate": pass_rate,
        "passed": passed,
        "per_dimension_averages": per_dim_averages,
        "taxonomy_breakdown": taxonomy_breakdown,
        "difficulty_breakdown": difficulty_breakdown,
        "tool_stats": tool_stats,
        "timing_stats": timing_stats,
    }

    return summary


def save_results(
    output_dir: str | Path,
    tasks: list[Any],
    completion_results: list[Any],
    judge_results: list[Any] | None,
    summary: dict,
) -> None:
    """Save all results to the output directory.

    Creates:
        - summary.json: Aggregate metrics
        - per_task.csv: Per-task scores and details
        - aggregated.csv: Taxonomy x difficulty pivot
        - traces/: Full conversation traces per task (JSONL)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Summary JSON
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Saved summary to %s", summary_path)

    # 2. Per-task CSV
    task_map = {t.task_id: t for t in tasks}
    completion_map = {cr.task_id: cr for cr in completion_results}
    judge_map = {}
    if judge_results:
        judge_map = {jr.task_id: jr for jr in judge_results}

    per_task_path = output_dir / "per_task.csv"
    fieldnames = [
        "task_id",
        "taxonomy_l1",
        "taxonomy_l2",
        "avg_difficulty",
        "difficulty_bucket",
        "n_tool_calls",
        "tools_used",
        "suggested_tools",
        "tool_overlap",
        "wall_time",
        "prompt_tokens",
        "completion_tokens",
        "coverage",
        "evidence_quality",
        "tool_usage",
        "reasoning_quality",
        "completeness",
        "weighted_total",
        "pass",
        "judge_reasoning",
    ]

    with open(per_task_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for task in tasks:
            cr = completion_map.get(task.task_id)
            jr = judge_map.get(task.task_id)

            called_tools = set()
            if cr:
                for tc in cr.tool_calls:
                    called_tools.add(tc.get("tool", "").replace("-", "_"))

            suggested = set(t.replace("-", "_") for t in task.suggested_tools)
            overlap = (
                len(called_tools & suggested) / len(suggested)
                if suggested
                else None
            )

            avg_diff = _avg_difficulty(task.difficulty)

            row = {
                "task_id": task.task_id,
                "taxonomy_l1": task.taxonomy.get("l1", ""),
                "taxonomy_l2": task.taxonomy.get("l2", ""),
                "avg_difficulty": round(avg_diff, 2),
                "difficulty_bucket": _bucket_label(avg_diff),
                "n_tool_calls": len(cr.tool_calls) if cr else 0,
                "tools_used": ";".join(sorted(called_tools)),
                "suggested_tools": ";".join(sorted(suggested)),
                "tool_overlap": round(overlap, 3) if overlap is not None else "",
                "wall_time": round(cr.wall_time, 2) if cr else "",
                "prompt_tokens": cr.tokens.get("prompt", 0) if cr else "",
                "completion_tokens": cr.tokens.get("completion", 0) if cr else "",
            }

            if jr:
                row.update(
                    {
                        "coverage": jr.scores.get("coverage", ""),
                        "evidence_quality": jr.scores.get("evidence_quality", ""),
                        "tool_usage": jr.scores.get("tool_usage", ""),
                        "reasoning_quality": jr.scores.get("reasoning_quality", ""),
                        "completeness": jr.scores.get("completeness", ""),
                        "weighted_total": jr.weighted_total,
                        "pass": 1 if jr.weighted_total >= summary.get("pass_threshold", 0.6) else 0,
                        "judge_reasoning": jr.reasoning[:500],
                    }
                )
            else:
                row.update({k: "" for k in fieldnames if k not in row})

            writer.writerow(row)

    logger.info("Saved per-task results to %s", per_task_path)

    # 3. Aggregated CSV (taxonomy x difficulty pivot)
    agg_path = output_dir / "aggregated.csv"
    agg_data: dict[tuple[str, str], list[float]] = defaultdict(list)
    agg_counts: dict[tuple[str, str], int] = defaultdict(int)

    for task in tasks:
        l1 = task.taxonomy.get("l1", "Unknown")
        bucket = _bucket_label(_avg_difficulty(task.difficulty))
        key = (l1, bucket)
        agg_counts[key] += 1
        if task.task_id in judge_map:
            agg_data[key].append(judge_map[task.task_id].weighted_total)

    with open(agg_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["taxonomy_l1", "difficulty", "n_tasks", "n_judged", "avg_weighted", "pass_rate"])
        for (l1, bucket) in sorted(agg_counts.keys()):
            scores = agg_data.get((l1, bucket), [])
            writer.writerow(
                [
                    l1,
                    bucket,
                    agg_counts[(l1, bucket)],
                    len(scores),
                    round(sum(scores) / len(scores), 4) if scores else "",
                    round(sum(1 for s in scores if s >= summary.get("pass_threshold", 0.6)) / len(scores), 4) if scores else "",
                ]
            )

    logger.info("Saved aggregated results to %s", agg_path)

    # 4. Traces (one JSONL per run, all tasks)
    traces_path = output_dir / "traces.jsonl"
    with open(traces_path, "w") as f:
        for cr in completion_results:
            record = {
                "task_id": cr.task_id,
                "model_answer": cr.model_answer[:2000],
                "tool_calls": cr.tool_calls,
                "trace": cr.trace,
                "tokens": cr.tokens,
                "wall_time": cr.wall_time,
            }
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    logger.info("Saved traces to %s", traces_path)
