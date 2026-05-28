"""Export refined questions to MCP benchmark format.

Converts the final deduplicated dataset into formats compatible with:
1. MCP-Atlas style evaluation (tool-augmented QA)
2. Standard QA benchmark (HuggingFace datasets)
3. Analysis-ready CSV/Parquet
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_jsonl(path: Path) -> list[dict]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def to_mcp_benchmark(questions: list[dict], output_dir: Path) -> Path:
    """Convert to MCP-Atlas compatible benchmark format.

    Each question becomes a task with:
    - task description (the self-contained question)
    - expected tool usage hints
    - difficulty metadata
    - gold standard info (if available)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "medical_open_questions_mcp.jsonl"

    tasks = []
    for i, q in enumerate(questions):
        task = {
            "task_id": f"med_open_{i:05d}",
            "task_description": q.get("self_contained_problem", ""),
            "category": q.get("taxonomy_level_1", ""),
            "subcategory": q.get("taxonomy_level_2", ""),
            "topic": q.get("taxonomy_level_3", ""),
            "open_status": q.get("open_status", "unknown"),
            "difficulty": {
                "clinical_knowledge": q.get("difficulty_clinical_knowledge", 3),
                "research_depth": q.get("difficulty_research_depth", 3),
                "multi_step_reasoning": q.get("difficulty_multi_step_reasoning", 3),
            },
            "suggested_tools": q.get("relevant_mcp_tools", []),
            "tool_rationale": q.get("mcp_tool_rationale", ""),
            "source": {
                "origin": q.get("source", ""),
                "id": q.get("source_id", ""),
                "url": q.get("source_url", ""),
                "title": q.get("source_title", ""),
            },
            "verification": {
                "venues": q.get("verification_venues", []),
                "notes": q.get("verification_notes", ""),
            },
            "metadata": {
                "question_type": q.get("question_type", ""),
                "why_open": q.get("why_open", ""),
                "status_reasoning": q.get("status_reasoning", ""),
                "status_evidence": q.get("status_evidence", ""),
                "original_question": q.get("original_question", ""),
            },
        }
        tasks.append(task)

    with open(out_path, "w") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")

    logger.info(f"Exported {len(tasks)} MCP benchmark tasks to {out_path}")
    return out_path


def to_hf_dataset(questions: list[dict], output_dir: Path) -> Path:
    """Convert to HuggingFace datasets format (Parquet)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "medical_open_questions.parquet"

    rows = []
    for q in questions:
        rows.append({
            "source_id": q.get("source_id", ""),
            "source_url": q.get("source_url", ""),
            "original_question": q.get("original_question", ""),
            "self_contained_problem": q.get("self_contained_problem", ""),
            "taxonomy_level_1": q.get("taxonomy_level_1", ""),
            "taxonomy_level_2": q.get("taxonomy_level_2", ""),
            "taxonomy_level_3": q.get("taxonomy_level_3", ""),
            "open_status": q.get("open_status", "unknown"),
            "status_reasoning": q.get("status_reasoning", ""),
            "status_evidence": q.get("status_evidence", ""),
            "relevant_mcp_tools": json.dumps(q.get("relevant_mcp_tools", [])),
            "question_type": q.get("question_type", ""),
            "difficulty_clinical_knowledge": q.get("difficulty_clinical_knowledge", 3),
            "difficulty_research_depth": q.get("difficulty_research_depth", 3),
            "difficulty_multi_step_reasoning": q.get("difficulty_multi_step_reasoning", 3),
        })

    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    logger.info(f"Exported {len(df)} questions to {out_path}")
    return out_path


def to_analysis_csv(questions: list[dict], output_dir: Path) -> Path:
    """Export CSV for quick analysis and visualization."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "medical_open_questions_analysis.csv"

    rows = []
    for q in questions:
        rows.append({
            "source": q.get("source", ""),
            "source_id": q.get("source_id", ""),
            "taxonomy_l1": q.get("taxonomy_level_1", ""),
            "taxonomy_l2": q.get("taxonomy_level_2", ""),
            "taxonomy_l3": q.get("taxonomy_level_3", ""),
            "open_status": q.get("open_status", ""),
            "question_type": q.get("question_type", ""),
            "difficulty_knowledge": q.get("difficulty_clinical_knowledge", 0),
            "difficulty_research": q.get("difficulty_research_depth", 0),
            "difficulty_reasoning": q.get("difficulty_multi_step_reasoning", 0),
            "n_mcp_tools": len(q.get("relevant_mcp_tools", [])),
            "question_length": len(q.get("self_contained_problem", "")),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    logger.info(f"Exported analysis CSV to {out_path}")
    return out_path


def generate_stats(questions: list[dict]) -> dict:
    """Generate dataset statistics."""
    df = pd.DataFrame(questions)

    stats = {
        "total_questions": len(questions),
        "sources": df["source"].value_counts().to_dict() if "source" in df else {},
        "open_status": df["open_status"].value_counts().to_dict() if "open_status" in df else {},
        "taxonomy_level_1": df["taxonomy_level_1"].value_counts().to_dict() if "taxonomy_level_1" in df else {},
        "taxonomy_level_2": df["taxonomy_level_2"].value_counts().head(20).to_dict() if "taxonomy_level_2" in df else {},
        "question_type": df["question_type"].value_counts().to_dict() if "question_type" in df else {},
        "avg_difficulty": {
            "clinical_knowledge": df.get("difficulty_clinical_knowledge", pd.Series([0])).mean(),
            "research_depth": df.get("difficulty_research_depth", pd.Series([0])).mean(),
            "multi_step_reasoning": df.get("difficulty_multi_step_reasoning", pd.Series([0])).mean(),
        },
        "avg_question_length": df["self_contained_problem"].str.len().mean() if "self_contained_problem" in df else 0,
    }
    return stats
