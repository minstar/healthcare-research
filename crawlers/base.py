"""Base crawler interface and shared utilities."""
from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator

from .checkpoint import CrawlCheckpoint

logger = logging.getLogger(__name__)


@dataclass
class RawDocument:
    source: str
    source_id: str
    url: str
    title: str
    abstract: str
    full_text: str = ""
    authors: list[str] = field(default_factory=list)
    publication_date: str = ""
    document_type: str = ""
    keywords: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class BaseCrawler(ABC):
    def __init__(self, name: str, output_dir: Path, rate_limit_sec: float = 0.5):
        self.name = name
        self.output_dir = output_dir / name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit_sec = rate_limit_sec
        self.checkpoint = CrawlCheckpoint(name, output_dir / ".checkpoints")

    @abstractmethod
    def crawl(self, queries: list[str], max_results: int) -> Iterator[RawDocument]:
        ...

    def save_documents(self, docs: list[RawDocument]) -> Path:
        out_path = self.output_dir / "documents.jsonl"
        with open(out_path, "w") as f:
            for doc in docs:
                f.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")
        logger.info(f"[{self.name}] Saved {len(docs)} documents to {out_path}")
        return out_path

    def save_documents_append(self, doc: RawDocument) -> None:
        """Append a single document to the output file (for incremental saving)."""
        out_path = self.output_dir / "documents.jsonl"
        with open(out_path, "a") as f:
            f.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")

    def rate_limit(self):
        time.sleep(self.rate_limit_sec)
