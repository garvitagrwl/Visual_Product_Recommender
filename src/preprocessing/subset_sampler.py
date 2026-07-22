"""Mandatory dataset subsetting.

Per the project's problem-statement PDF, the pipeline must operate on
an explicit subset (5-8 categories, ~200-300 images per category)
rather than the full ~44k-image dataset. This keeps class balance
controlled and training/embedding time tractable, and is required for
a fair comparison across the baseline/transfer-learning/Siamese
approaches later in the project.
"""

from __future__ import annotations

import logging
from typing import List

import pandas as pd


class SubsetSampler:
    """Filters to configured categories and samples a balanced subset."""

    def __init__(
        self,
        category_column: str,
        categories: List[str],
        min_samples_per_category: int,
        max_samples_per_category: int,
        seed: int,
        logger: logging.Logger,
    ):
        self.category_column = category_column
        self.categories = categories
        self.min_samples_per_category = min_samples_per_category
        self.max_samples_per_category = max_samples_per_category
        self.seed = seed
        self.logger = logger

    def report_available_categories(self, df: pd.DataFrame, top_n: int = 30) -> pd.Series:
        """Return value counts for the category column (for inspection).

        Useful the first time this is run against the real dataset, to
        confirm the configured `categories` list actually exists and
        has enough images before committing to it.
        """
        counts = df[self.category_column].value_counts()
        self.logger.info(
            "Top %d available values in '%s':\n%s",
            top_n,
            self.category_column,
            counts.head(top_n).to_string(),
        )
        return counts

    def sample(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to configured categories and sample a balanced subset.

        Args:
            df: Cleaned metadata DataFrame (post-validation).

        Returns:
            Subset DataFrame containing only the configured categories,
            with up to `max_samples_per_category` rows per category.

        Raises:
            ValueError: If a configured category has fewer records than
                `min_samples_per_category`.
        """
        filtered = df[df[self.category_column].isin(self.categories)].copy()

        sampled_frames = []
        for category in self.categories:
            cat_df = filtered[filtered[self.category_column] == category]
            available = len(cat_df)

            if available < self.min_samples_per_category:
                raise ValueError(
                    f"Category '{category}' has only {available} valid images, "
                    f"below the minimum of {self.min_samples_per_category}. "
                    "Choose a different category or lower the minimum in "
                    "configs/config.yaml."
                )

            n = min(available, self.max_samples_per_category)
            sampled = cat_df.sample(n=n, random_state=self.seed)
            sampled_frames.append(sampled)

            self.logger.info(
                "Category '%s': %d available -> %d sampled",
                category,
                available,
                n,
            )

        subset = pd.concat(sampled_frames, ignore_index=True)
        self.logger.info(
            "Final subset: %d images across %d categories",
            len(subset),
            len(self.categories),
        )
        return subset
