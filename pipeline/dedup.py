"""Embedding-based deduplication for medical questions.

Uses sentence-transformers to compute embeddings and cosine similarity,
removing near-duplicates above a configurable threshold.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingDeduplicator:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold: float = 0.90,
        batch_size: int = 256,
    ):
        self.model_name = model_name
        self.threshold = similarity_threshold
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def deduplicate(
        self, questions: list[dict], text_field: str = "self_contained_problem"
    ) -> list[dict]:
        if not questions:
            return []

        texts = [q.get(text_field, "") for q in questions]
        logger.info(f"Computing embeddings for {len(texts)} questions...")
        embeddings = self.model.encode(
            texts, batch_size=self.batch_size, show_progress_bar=True, normalize_embeddings=True
        )

        keep_mask = np.ones(len(questions), dtype=bool)
        removed_count = 0

        logger.info("Computing pairwise similarities...")
        for i in range(len(questions)):
            if not keep_mask[i]:
                continue
            for j in range(i + 1, len(questions)):
                if not keep_mask[j]:
                    continue
                sim = float(np.dot(embeddings[i], embeddings[j]))
                if sim >= self.threshold:
                    # Keep the one from a more authoritative source
                    priority = {"pubmed": 3, "cochrane": 3, "arxiv": 2, "medrxiv": 1, "biorxiv": 1}
                    pi = priority.get(questions[i].get("source", ""), 0)
                    pj = priority.get(questions[j].get("source", ""), 0)
                    if pj > pi:
                        keep_mask[i] = False
                        removed_count += 1
                        break
                    else:
                        keep_mask[j] = False
                        removed_count += 1

        deduplicated = [q for q, keep in zip(questions, keep_mask) if keep]
        logger.info(
            f"Deduplication: {len(questions)} → {len(deduplicated)} "
            f"(removed {removed_count} duplicates at threshold {self.threshold})"
        )
        return deduplicated

    def deduplicate_file(self, input_path: Path, output_path: Path, text_field: str = "self_contained_problem"):
        questions = []
        with open(input_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    questions.append(json.loads(line))

        deduplicated = self.deduplicate(questions, text_field)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for q in deduplicated:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")

        return deduplicated
