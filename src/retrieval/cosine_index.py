"""Brute-force cosine similarity search.

Vectors are L2-normalized once at build time, so cosine similarity
between any query and the catalog reduces to a single matrix multiply
(normalized_query @ normalized_matrix.T). This is also exactly the
convention FAISS's IndexFlatIP expects pre-normalized vectors to
follow, so migrating to FAISS later is a backend swap, not a rewrite.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from src.retrieval.base_index import BaseSimilarityIndex


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalize, guarding against zero-norm rows."""
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1e-12, norms)
    return matrix / norms


class CosineSimilarityIndex(BaseSimilarityIndex):
    """Exact (brute-force) cosine similarity search over all embeddings."""

    def __init__(self):
        self._ids: np.ndarray | None = None
        self._normalized_embeddings: np.ndarray | None = None

    def build(self, ids: np.ndarray, embeddings: np.ndarray) -> None:
        if len(ids) != len(embeddings):
            raise ValueError(
                f"ids ({len(ids)}) and embeddings ({len(embeddings)}) length mismatch"
            )
        self._ids = np.asarray(ids)
        self._normalized_embeddings = _l2_normalize(embeddings.astype(np.float32))

    def query(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        if self._normalized_embeddings is None:
            raise RuntimeError("Index has not been built yet - call build() first.")

        query = _l2_normalize(query_embedding.reshape(1, -1).astype(np.float32))
        similarities = (query @ self._normalized_embeddings.T).flatten()

        top_k = min(top_k, len(similarities))
        # argpartition for O(N) top-k selection, then sort just those k.
        top_indices = np.argpartition(-similarities, top_k - 1)[:top_k]
        top_indices = top_indices[np.argsort(-similarities[top_indices])]

        return [
            (str(self._ids[i]), float(similarities[i])) for i in top_indices
        ]
