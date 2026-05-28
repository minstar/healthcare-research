"""Load and filter benchmark tasks from JSONL files."""
from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Task:
    task_id: str
    question: str
    gold_answer: dict | None
    suggested_tools: list[str]
    difficulty: dict
    taxonomy: dict
    source: dict
    metadata: dict


def _parse_task(raw: dict) -> Task:
    """Parse a single JSONL record into a Task dataclass.

    Handles both the original mcp_benchmark.jsonl schema and the
    gold-augmented mcp_benchmark_with_gold.jsonl schema.
    """
    # Task ID
    task_id = raw.get("id") or raw.get("task_id", "")

    # Question text
    question = raw.get("question") or raw.get("task_description", "")

    # Gold answer (may be absent)
    gold_answer = raw.get("gold_answer", None)

    # Suggested tools — normalise hyphens to underscores for consistency
    raw_tools = raw.get("relevant_mcp_tools") or raw.get("suggested_tools", [])
    suggested_tools = [t.replace("-", "_") for t in raw_tools]

    # Difficulty axes — support both flat and nested formats
    if "difficulty_axes" in raw:
        difficulty = raw["difficulty_axes"]
    elif "difficulty" in raw and isinstance(raw["difficulty"], dict):
        difficulty = raw["difficulty"]
    else:
        difficulty = {
            "clinical_knowledge": raw.get("difficulty_clinical_knowledge", 3),
            "research_depth": raw.get("difficulty_research_depth", 3),
            "multi_step_reasoning": raw.get("difficulty_multi_step_reasoning", 3),
        }

    # Taxonomy
    if "taxonomy" in raw and isinstance(raw["taxonomy"], dict):
        taxonomy = raw["taxonomy"]
    else:
        taxonomy = {
            "l1": raw.get("category") or raw.get("taxonomy_level_1", ""),
            "l2": raw.get("subcategory") or raw.get("taxonomy_level_2", ""),
            "l3": raw.get("topic") or raw.get("taxonomy_level_3", ""),
        }

    # Source
    source = raw.get("source", {})

    # Metadata — collect remaining informational fields
    metadata = raw.get("metadata", {})
    for extra_key in ("open_status", "status_reasoning", "verification_venues"):
        if extra_key in raw:
            metadata[extra_key] = raw[extra_key]

    return Task(
        task_id=task_id,
        question=question,
        gold_answer=gold_answer,
        suggested_tools=suggested_tools,
        difficulty=difficulty,
        taxonomy=taxonomy,
        source=source,
        metadata=metadata,
    )


def _avg_difficulty(difficulty: dict) -> float:
    """Return the mean of all numeric difficulty values."""
    vals = [v for v in difficulty.values() if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else 3.0


def load_tasks(
    data_path: str | Path,
    *,
    taxonomy_l1: str | None = None,
    difficulty_min: float | None = None,
    difficulty_max: float | None = None,
    open_status: str | None = None,
    required_tools: list[str] | None = None,
    limit: int | None = None,
    seed: int | None = None,
) -> list[Task]:
    """Load benchmark tasks from a JSONL file with optional filtering.

    Args:
        data_path: Path to the JSONL file.
        taxonomy_l1: Keep only tasks whose taxonomy l1 matches (case-insensitive substring).
        difficulty_min: Keep tasks with avg difficulty >= this value.
        difficulty_max: Keep tasks with avg difficulty <= this value.
        open_status: Keep tasks whose open_status matches exactly.
        required_tools: Keep tasks that include ALL of these tools in suggested_tools.
        limit: Maximum number of tasks to return (applied after filtering).
        seed: Random seed for reproducible sampling when limit < available tasks.

    Returns:
        List of Task objects.
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Benchmark data not found: {data_path}")

    # ---- Load all records ----
    raw_records: list[dict] = []
    with open(data_path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw_records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON at line %d", lineno)

    logger.info("Loaded %d raw records from %s", len(raw_records), data_path)

    # ---- Parse ----
    tasks = [_parse_task(r) for r in raw_records]

    # ---- Filter ----
    if taxonomy_l1:
        key = taxonomy_l1.lower()
        tasks = [t for t in tasks if key in t.taxonomy.get("l1", "").lower()]
        logger.info("After taxonomy_l1 filter (%s): %d tasks", taxonomy_l1, len(tasks))

    if difficulty_min is not None:
        tasks = [t for t in tasks if _avg_difficulty(t.difficulty) >= difficulty_min]
        logger.info("After difficulty_min filter (>= %s): %d tasks", difficulty_min, len(tasks))

    if difficulty_max is not None:
        tasks = [t for t in tasks if _avg_difficulty(t.difficulty) <= difficulty_max]
        logger.info("After difficulty_max filter (<= %s): %d tasks", difficulty_max, len(tasks))

    if open_status:
        tasks = [t for t in tasks if t.metadata.get("open_status") == open_status]
        logger.info("After open_status filter (%s): %d tasks", open_status, len(tasks))

    if required_tools:
        norm_req = {t.replace("-", "_") for t in required_tools}
        tasks = [t for t in tasks if norm_req.issubset(set(t.suggested_tools))]
        logger.info("After required_tools filter (%s): %d tasks", required_tools, len(tasks))

    # ---- Sample / limit ----
    if limit is not None and limit < len(tasks):
        rng = random.Random(seed)
        tasks = rng.sample(tasks, limit)
        logger.info("Sampled %d tasks (seed=%s)", limit, seed)

    logger.info("Returning %d tasks", len(tasks))
    return tasks
