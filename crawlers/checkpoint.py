"""Checkpoint manager for resumable crawling.

Tracks which queries/pages have been completed so that if a crawl
fails due to API errors, it can resume from the exact failure point.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class CrawlCheckpoint:
    """Tracks crawl progress per source with query-level granularity.

    State file format (JSON):
    {
      "source": "pubmed",
      "started_at": "2026-05-28T10:00:00",
      "completed_queries": ["query1", "query2"],
      "current_query": "query3",
      "current_cursor": 200,
      "collected_ids": ["id1", "id2", ...],
      "total_documents": 150,
      "last_error": null
    }
    """

    def __init__(self, source_name: str, checkpoint_dir: Path):
        self.source_name = source_name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.checkpoint_dir / f"{source_name}_checkpoint.json"
        self.state = self._load()

    def _load(self) -> dict:
        if self.state_path.exists():
            with open(self.state_path) as f:
                state = json.load(f)
            logger.info(
                f"[{self.source_name}] Resuming from checkpoint: "
                f"{len(state.get('completed_queries', []))} queries done, "
                f"{state.get('total_documents', 0)} docs collected"
            )
            return state
        return {
            "source": self.source_name,
            "started_at": datetime.now().isoformat(),
            "completed_queries": [],
            "current_query": None,
            "current_cursor": 0,
            "collected_ids": [],
            "total_documents": 0,
            "last_error": None,
        }

    def save(self):
        with open(self.state_path, "w") as f:
            json.dump(self.state, f, indent=2, ensure_ascii=False)

    def is_query_done(self, query: str) -> bool:
        return query in self.state.get("completed_queries", [])

    def mark_query_started(self, query: str, cursor: int = 0):
        self.state["current_query"] = query
        self.state["current_cursor"] = cursor
        self.state["last_error"] = None
        self.save()

    def update_cursor(self, cursor: int):
        self.state["current_cursor"] = cursor
        self.save()

    def mark_query_done(self, query: str):
        if query not in self.state["completed_queries"]:
            self.state["completed_queries"].append(query)
        self.state["current_query"] = None
        self.state["current_cursor"] = 0
        self.save()

    def add_collected_id(self, doc_id: str):
        if doc_id not in self.state["collected_ids"]:
            self.state["collected_ids"].append(doc_id)
            self.state["total_documents"] = len(self.state["collected_ids"])

    def is_collected(self, doc_id: str) -> bool:
        return doc_id in self.state["collected_ids"]

    def record_error(self, error: str):
        self.state["last_error"] = {
            "message": error,
            "timestamp": datetime.now().isoformat(),
            "query": self.state.get("current_query"),
            "cursor": self.state.get("current_cursor"),
        }
        self.save()
        logger.error(
            f"[{self.source_name}] Error at query='{self.state.get('current_query')}' "
            f"cursor={self.state.get('current_cursor')}: {error}"
        )

    def get_resume_point(self) -> tuple[str | None, int]:
        return self.state.get("current_query"), self.state.get("current_cursor", 0)

    def clear(self):
        if self.state_path.exists():
            self.state_path.unlink()
        self.state = {
            "source": self.source_name,
            "started_at": datetime.now().isoformat(),
            "completed_queries": [],
            "current_query": None,
            "current_cursor": 0,
            "collected_ids": [],
            "total_documents": 0,
            "last_error": None,
        }

    @property
    def total_collected(self) -> int:
        return self.state.get("total_documents", 0)

    def summary(self) -> str:
        s = self.state
        lines = [
            f"Source: {s['source']}",
            f"Started: {s.get('started_at', 'N/A')}",
            f"Completed queries: {len(s.get('completed_queries', []))}",
            f"Current query: {s.get('current_query', 'None')}",
            f"Current cursor: {s.get('current_cursor', 0)}",
            f"Total documents: {s.get('total_documents', 0)}",
        ]
        if s.get("last_error"):
            lines.append(f"Last error: {s['last_error']['message'][:200]}")
        return "\n".join(lines)
