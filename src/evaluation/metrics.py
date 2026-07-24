"""Precision@K and Recall@K for retrieval evaluation.

Relevance criterion: an item is relevant to a query if it shares the
query's category (articleType). This is an approximation - no
manually-labeled visual-similarity ground truth exists for this
dataset - stated explicitly since it shapes how every number here
should be interpreted: a "relevant" result is same-category, not
necessarily visually similar, and vice versa.
"""

from __future__ import annotations

from typing import List


def precision_at_k(retrieved_categories: List[str], query_category: str, k: int) -> float:
    """Fraction of the top-k retrieved items that share the query's category.

    Args:
        retrieved_categories: Categories of retrieved items, ranked
            (already excludes the query itself), truncated or not.
        query_category: The query image's own category.
        k: How many top results to consider.

    Returns:
        Precision@k in [0, 1]. Returns 0.0 if `retrieved_categories`
        has fewer than k items available (can't fill k slots).
    """
    top_k = retrieved_categories[:k]
    if len(top_k) == 0:
        return 0.0
    relevant_count = sum(1 for cat in top_k if cat == query_category)
    return relevant_count / k


def recall_at_k(
    retrieved_categories: List[str], query_category: str, total_relevant: int, k: int
) -> float:
    """Fraction of all relevant items (in the whole catalog) found in the top-k.

    Args:
        retrieved_categories: Categories of retrieved items, ranked
            (already excludes the query itself).
        query_category: The query image's own category.
        total_relevant: Total number of relevant items available in
            the catalog for this query (same category, excluding the
            query itself).
        k: How many top results to consider.

    Returns:
        Recall@k in [0, 1]. Returns 0.0 if `total_relevant` is 0
        (shouldn't happen given Phase 1's min-samples-per-category
        guard, but handled defensively).
    """
    if total_relevant == 0:
        return 0.0
    top_k = retrieved_categories[:k]
    relevant_count = sum(1 for cat in top_k if cat == query_category)
    return relevant_count / total_relevant
