"""Runs a full evaluation pass: many queries, aggregated Precision@K,
Recall@K, and retrieval latency for one stage's index+model.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd
import tensorflow as tf

from src.evaluation.metrics import precision_at_k, recall_at_k
from src.preprocessing.transforms import load_and_preprocess_single_image
from src.retrieval.base_index import BaseSimilarityIndex


@dataclass
class EvaluationResult:
    """Aggregate metrics for one stage, across all evaluated queries."""

    stage: str
    num_queries: int
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    mean_latency_ms: float = 0.0
    median_latency_ms: float = 0.0


class RetrievalEvaluator:
    """Evaluates one stage's retrieval quality and speed over a query set."""

    def __init__(
        self,
        model: tf.keras.Model,
        index: BaseSimilarityIndex,
        metadata_df: pd.DataFrame,
        images_dir: str,
        image_size: tuple[int, int],
        backbone: str,
        category_column: str,
        logger: logging.Logger,
    ):
        self.model = model
        self.index = index
        self.images_dir = images_dir
        self.image_size = image_size
        self.backbone = backbone
        self.category_column = category_column
        self.logger = logger

        self.metadata_df = metadata_df.set_index(metadata_df["id"].astype(str))
        # Precompute total relevant count per category (catalog-wide,
        # not yet adjusted for excluding the query itself).
        self._category_counts = metadata_df[category_column].value_counts().to_dict()

    def _query_one(self, query_row: pd.Series, max_k: int) -> tuple[List[str], float]:
        """Run one query, excluding the query's own id from results.

        Returns:
            (retrieved_categories, latency_ms)
        """
        query_id = str(query_row["id"])
        image_path = f"{self.images_dir.rstrip('/')}/{query_row['image_filename']}"

        start = time.perf_counter()
        image_batch = load_and_preprocess_single_image(image_path, self.image_size, self.backbone)
        query_embedding = self.model(image_batch, training=False).numpy()[0]
        # Query for one extra result in case the query's own id appears
        # (it will, since queries are drawn from the catalog) - this
        # guarantees `max_k` genuine (non-self) results remain after filtering.
        matches = self.index.query(query_embedding, top_k=max_k + 1)
        latency_ms = (time.perf_counter() - start) * 1000

        retrieved_ids = [pid for pid, _ in matches if pid != query_id][:max_k]
        retrieved_categories = [
            self.metadata_df.loc[pid, self.category_column]
            for pid in retrieved_ids
            if pid in self.metadata_df.index
        ]
        return retrieved_categories, latency_ms

    def evaluate(self, query_df: pd.DataFrame, k_values: List[int]) -> EvaluationResult:
        """Run evaluation over every row in `query_df`.

        Args:
            query_df: Query images (typically the validation set).
            k_values: Which K values to compute Precision@K/Recall@K for.

        Returns:
            Aggregated `EvaluationResult` across all queries.
        """
        max_k = max(k_values)
        precision_sums = {k: 0.0 for k in k_values}
        recall_sums = {k: 0.0 for k in k_values}
        latencies = []

        for i, (_, query_row) in enumerate(query_df.iterrows()):
            retrieved_categories, latency_ms = self._query_one(query_row, max_k)
            latencies.append(latency_ms)

            query_category = query_row[self.category_column]
            total_relevant = self._category_counts.get(query_category, 0) - 1  # exclude self

            for k in k_values:
                precision_sums[k] += precision_at_k(retrieved_categories, query_category, k)
                recall_sums[k] += recall_at_k(retrieved_categories, query_category, total_relevant, k)

            if (i + 1) % 50 == 0:
                self.logger.info("Evaluated %d/%d queries...", i + 1, len(query_df))

        n = len(query_df)
        latencies_sorted = sorted(latencies)
        mid = len(latencies_sorted) // 2
        median_latency = (
            latencies_sorted[mid]
            if len(latencies_sorted) % 2 == 1
            else (latencies_sorted[mid - 1] + latencies_sorted[mid]) / 2
        )

        return EvaluationResult(
            stage="",  # filled in by the caller, which knows the stage name
            num_queries=n,
            precision_at_k={k: precision_sums[k] / n for k in k_values},
            recall_at_k={k: recall_sums[k] / n for k in k_values},
            mean_latency_ms=sum(latencies) / n,
            median_latency_ms=median_latency,
        )
