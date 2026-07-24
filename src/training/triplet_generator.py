"""Generates Anchor/Positive/Negative triplets for Siamese training.

Positive: a different image from the SAME category as the anchor.
Negative: an image from a DIFFERENT category than the anchor.

Built once and saved to CSV (like Phase 1's subset/split), rather
than generated fresh in-memory each epoch - keeps this step
reproducible and auditable via the config seed.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd


class TripletGenerator:
    """Builds one (anchor, positive, negative) triplet per anchor image."""

    def __init__(self, category_column: str, seed: int, logger: logging.Logger):
        self.category_column = category_column
        self.seed = seed
        self.logger = logger

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build one triplet per row in `df`, treated as the anchor.

        Args:
            df: Metadata DataFrame with `id`, `image_filename`, and
                the category column (as produced by Phase 1).

        Returns:
            DataFrame with columns: anchor_id, anchor_filename,
            positive_id, positive_filename, negative_id,
            negative_filename, anchor_category, negative_category.

        Raises:
            ValueError: If any category in `df` has fewer than 2
                images (no valid positive can be formed) or if only
                one category exists (no valid negative can be formed).
        """
        rng = np.random.default_rng(self.seed)

        category_counts = df[self.category_column].value_counts()
        too_small = category_counts[category_counts < 2]
        if len(too_small) > 0:
            raise ValueError(
                f"Categories with fewer than 2 images can't form a positive pair: "
                f"{too_small.to_dict()}"
            )
        if df[self.category_column].nunique() < 2:
            raise ValueError("Need at least 2 categories to form negative pairs.")

        by_category = {cat: group for cat, group in df.groupby(self.category_column)}
        categories = list(by_category.keys())

        triplets = []
        for _, anchor_row in df.iterrows():
            anchor_category = anchor_row[self.category_column]

            # Positive: a different image from the same category.
            same_cat_df = by_category[anchor_category]
            candidates = same_cat_df[same_cat_df["id"] != anchor_row["id"]]
            positive_row = candidates.sample(n=1, random_state=rng.integers(0, 2**31)).iloc[0]

            # Negative: any image from a different category.
            other_categories = [c for c in categories if c != anchor_category]
            negative_category = other_categories[rng.integers(0, len(other_categories))]
            negative_row = by_category[negative_category].sample(
                n=1, random_state=rng.integers(0, 2**31)
            ).iloc[0]

            triplets.append(
                {
                    "anchor_id": anchor_row["id"],
                    "anchor_filename": anchor_row["image_filename"],
                    "positive_id": positive_row["id"],
                    "positive_filename": positive_row["image_filename"],
                    "negative_id": negative_row["id"],
                    "negative_filename": negative_row["image_filename"],
                    "anchor_category": anchor_category,
                    "negative_category": negative_category,
                }
            )

        triplets_df = pd.DataFrame(triplets)
        self.logger.info(
            "Generated %d triplets across %d categories", len(triplets_df), len(categories)
        )
        return triplets_df
