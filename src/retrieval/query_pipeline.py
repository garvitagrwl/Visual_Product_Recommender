"""End-to-end recommendation pipeline for a single uploaded image.

Pipeline: uploaded image -> preprocessing -> embedding extraction ->
cosine similarity -> ranking -> top-K results joined with product
metadata (name, brand, category, colour) for display.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd
import tensorflow as tf

from src.preprocessing.transforms import load_and_preprocess_single_image
from src.retrieval.base_index import BaseSimilarityIndex


@dataclass
class RecommendationResult:
    """One ranked recommendation, with display-ready metadata."""

    product_id: str
    similarity_score: float
    product_display_name: str
    article_type: str
    base_colour: str
    image_filename: str


class RecommendationPipeline:
    """Ties together preprocessing, embedding, and similarity search."""

    def __init__(
        self,
        model: tf.keras.Model,
        index: BaseSimilarityIndex,
        metadata_df: pd.DataFrame,
        image_size: tuple[int, int],
        backbone: str,
    ):
        self.model = model
        self.index = index
        self.metadata_df = metadata_df.set_index(metadata_df["id"].astype(str))
        self.image_size = image_size
        self.backbone = backbone

    def recommend(self, query_image_path: str, top_k: int) -> List[RecommendationResult]:
        """Return the top_k most visually similar products to an image.

        Args:
            query_image_path: Path to the uploaded query image.
            top_k: Number of recommendations to return.

        Returns:
            List of `RecommendationResult`, ranked by similarity
            descending.
        """
        image_batch = load_and_preprocess_single_image(
            query_image_path, self.image_size, self.backbone
        )
        query_embedding = self.model(image_batch, training=False).numpy()[0]

        matches = self.index.query(query_embedding, top_k)

        results = []
        for product_id, score in matches:
            if product_id not in self.metadata_df.index:
                continue  # Defensive: skip if metadata is somehow missing.
            row = self.metadata_df.loc[product_id]
            results.append(
                RecommendationResult(
                    product_id=product_id,
                    similarity_score=score,
                    product_display_name=row["productDisplayName"],
                    article_type=row["articleType"],
                    base_colour=row["baseColour"],
                    image_filename=row["image_filename"],
                )
            )
        return results
