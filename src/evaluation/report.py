"""Builds the final cross-stage comparison table and chart."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")  # no display available in this environment - write straight to file
import matplotlib.pyplot as plt
import pandas as pd

from src.evaluation.evaluator import EvaluationResult


def build_comparison_table(
    results: List[EvaluationResult],
    embedding_generation_times: Dict[str, float],
    k_values: List[int],
) -> pd.DataFrame:
    """Assemble one row per stage with all metrics side by side.

    Args:
        results: One `EvaluationResult` per stage.
        embedding_generation_times: stage -> generation_time_seconds,
            pulled from each stage's saved embedding metadata JSON
            (Phase 3/6/7's output) rather than recomputed here.
        k_values: Which K values were evaluated.

    Returns:
        A DataFrame with one row per stage, ready to print or save.
    """
    rows = []
    for result in results:
        row = {
            "stage": result.stage,
            "num_queries": result.num_queries,
            "mean_latency_ms": round(result.mean_latency_ms, 2),
            "median_latency_ms": round(result.median_latency_ms, 2),
            "embedding_generation_time_s": embedding_generation_times.get(result.stage),
        }
        for k in k_values:
            row[f"precision@{k}"] = round(result.precision_at_k[k], 4)
            row[f"recall@{k}"] = round(result.recall_at_k[k], 4)
        rows.append(row)

    return pd.DataFrame(rows)


def save_comparison_table(df: pd.DataFrame, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_comparison_chart(
    results: List[EvaluationResult],
    k_values: List[int],
    output_path: str,
) -> None:
    """Grouped bar chart: Precision@K and Recall@K, one group per stage."""
    stages = [r.stage for r in results]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    bar_width = 0.8 / len(k_values)
    x = range(len(stages))

    for ax, metric_name, metric_key in [
        (axes[0], "Precision@K", "precision_at_k"),
        (axes[1], "Recall@K", "recall_at_k"),
    ]:
        for i, k in enumerate(k_values):
            values = [getattr(r, metric_key)[k] for r in results]
            offsets = [xi + i * bar_width for xi in x]
            ax.bar(offsets, values, width=bar_width, label=f"K={k}")

        ax.set_title(metric_name)
        ax.set_xticks([xi + bar_width * (len(k_values) - 1) / 2 for xi in x])
        ax.set_xticklabels(stages, rotation=15)
        ax.set_ylim(0, max(1.0, ax.get_ylim()[1]))
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Retrieval Quality Comparison (relevance = same category)")
    fig.tight_layout()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
