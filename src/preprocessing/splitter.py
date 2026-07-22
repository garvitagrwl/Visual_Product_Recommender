"""Stratified train/validation split, keeping category balance intact."""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.model_selection import train_test_split


class DatasetSplitter:
    """Creates a reproducible, category-stratified train/val split."""

    def __init__(self, train_ratio: float, stratify_column: str, seed: int, logger: logging.Logger):
        self.train_ratio = train_ratio
        self.stratify_column = stratify_column
        self.seed = seed
        self.logger = logger

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split a DataFrame into train and validation sets.

        Args:
            df: Subset DataFrame to split.

        Returns:
            (train_df, val_df)
        """
        train_df, val_df = train_test_split(
            df,
            train_size=self.train_ratio,
            random_state=self.seed,
            stratify=df[self.stratify_column],
        )
        self.logger.info(
            "Split: %d train / %d val (ratio=%.2f, stratified by '%s')",
            len(train_df),
            len(val_df),
            self.train_ratio,
            self.stratify_column,
        )
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True)
