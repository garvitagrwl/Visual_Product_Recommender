"""Abstract interface for similarity search backends.

Any concrete index (cosine/NumPy now, FAISS later) implements just
`build` and `query`. Callers (recommend.py, the Phase 5 UI) depend
only on this interface, so swapping the backend later means adding a
new class here, not touching calling code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np


class BaseSimilarityIndex(ABC):
    """Interface every similarity search backend must implement."""

    @abstractmethod
    def build(self, ids: np.ndarray, embeddings: np.ndarray) -> None:
        """Build the index from a set of (id, embedding) pairs.

        Args:
            ids: Shape (N,) array of product ids.
            embeddings: Shape (N, D) array of embeddings, row-aligned
                with `ids`.
        """
        raise NotImplementedError

    @abstractmethod
    def query(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        """Find the top_k most similar items to a query embedding.

        Args:
            query_embedding: Shape (D,) embedding of the query image.
            top_k: Number of results to return.

        Returns:
            List of (id, similarity_score) tuples, sorted by
            similarity_score descending.
        """
        raise NotImplementedError
