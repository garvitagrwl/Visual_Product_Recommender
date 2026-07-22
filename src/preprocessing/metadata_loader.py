"""Loads and merges the Fashion Product Images dataset's tabular metadata.

The Kaggle dataset ships two relevant tabular files:
- styles.csv: one row per product id with rich metadata (category,
  article type, colour, etc). Known to have a handful of malformed
  rows (embedded commas in free-text fields), so parsing is defensive.
- images.csv: maps product id -> image filename.

This module produces a single merged DataFrame used by every
downstream preprocessing step.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

REQUIRED_STYLE_COLUMNS: List[str] = [
    "id",
    "gender",
    "masterCategory",
    "subCategory",
    "articleType",
    "baseColour",
    "season",
    "usage",
    "productDisplayName",
]


class MetadataLoadError(Exception):
    """Raised when required metadata files are missing or unreadable."""


class MetadataLoader:
    """Loads and merges styles/images metadata into one DataFrame."""

    def __init__(self, styles_csv: str | Path, images_csv: str | Path | None = None):
        """
        Args:
            styles_csv: Path to styles.csv.
            images_csv: Path to images.csv. Optional - if absent, image
                filenames are derived as `{id}.jpg`, which matches the
                dataset's actual convention.
        """
        self.styles_csv = Path(styles_csv)
        self.images_csv = Path(images_csv) if images_csv else None

    def load(self) -> pd.DataFrame:
        """Load and merge metadata.

        Returns:
            DataFrame with one row per product, including a
            `image_filename` column pointing to the expected image file.

        Raises:
            MetadataLoadError: If styles.csv is missing or missing
                required columns.
        """
        if not self.styles_csv.exists():
            raise MetadataLoadError(f"styles.csv not found at {self.styles_csv}")

        # engine="python" + on_bad_lines="skip": a known handful of rows
        # in this dataset have unescaped commas in productDisplayName
        # that break the default C parser. Skipping keeps the pipeline
        # robust; the validator reports how many rows were dropped here.
        styles_df = pd.read_csv(
            self.styles_csv,
            engine="python",
            on_bad_lines="skip",
        )

        missing_cols = [c for c in REQUIRED_STYLE_COLUMNS if c not in styles_df.columns]
        if missing_cols:
            raise MetadataLoadError(
                f"styles.csv is missing required columns: {missing_cols}"
            )

        styles_df["id"] = styles_df["id"].astype(str)

        if self.images_csv and self.images_csv.exists():
            images_df = pd.read_csv(self.images_csv)
            # images.csv typically has columns: filename, link
            images_df["id"] = images_df["filename"].str.replace(
                r"\.[a-zA-Z]+$", "", regex=True
            )
            merged = styles_df.merge(
                images_df[["id", "filename"]], on="id", how="left"
            )
            merged["image_filename"] = merged["filename"].fillna(
                merged["id"] + ".jpg"
            )
            merged = merged.drop(columns=["filename"])
        else:
            merged = styles_df.copy()
            merged["image_filename"] = merged["id"] + ".jpg"

        return merged
