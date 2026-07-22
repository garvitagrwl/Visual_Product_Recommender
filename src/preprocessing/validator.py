"""Dataset validation: structure checks, missing-image detection, and
invalid-record removal.

Produces a cleaned DataFrame plus a JSON-serializable report so every
run is auditable (how many records were dropped, and why).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List

import pandas as pd

REQUIRED_NON_NULL_COLUMNS: List[str] = [
    "id",
    "articleType",
    "image_filename",
]


@dataclass
class ValidationReport:
    """Summary of what the validator found and removed."""

    total_records_loaded: int = 0
    missing_images_count: int = 0
    null_required_field_count: int = 0
    duplicate_id_count: int = 0
    valid_records_remaining: int = 0
    missing_image_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


class DatasetValidator:
    """Validates dataset structure and cleans the metadata DataFrame."""

    def __init__(self, images_dir: str | Path, logger: logging.Logger):
        self.images_dir = Path(images_dir)
        self.logger = logger

    def validate_structure(self) -> None:
        """Raise if the expected dataset directory layout is absent."""
        if not self.images_dir.exists():
            raise FileNotFoundError(
                f"Images directory not found: {self.images_dir}. "
                "Download and extract the Kaggle Fashion Product Images "
                "dataset so it matches the path in configs/config.yaml."
            )

    def _detect_missing_images(self, df: pd.DataFrame) -> pd.Series:
        """Return a boolean mask of rows whose image file does not exist."""
        exists = df["image_filename"].apply(
            lambda fname: (self.images_dir / str(fname)).exists()
        )
        return ~exists

    def validate(self, df: pd.DataFrame) -> tuple[pd.DataFrame, ValidationReport]:
        """Run all validation checks and return a cleaned DataFrame.

        Args:
            df: Merged metadata DataFrame from MetadataLoader.

        Returns:
            (cleaned_df, report) - cleaned_df has invalid rows removed;
            report summarizes what was removed and why.
        """
        report = ValidationReport(total_records_loaded=len(df))
        working = df.copy()

        # 1. Drop rows missing required fields.
        null_mask = working[REQUIRED_NON_NULL_COLUMNS].isnull().any(axis=1)
        report.null_required_field_count = int(null_mask.sum())
        working = working[~null_mask]

        # 2. Drop duplicate product ids.
        dup_mask = working["id"].duplicated(keep="first")
        report.duplicate_id_count = int(dup_mask.sum())
        working = working[~dup_mask]

        # 3. Drop rows whose image file is missing on disk.
        missing_mask = self._detect_missing_images(working)
        report.missing_images_count = int(missing_mask.sum())
        report.missing_image_ids = working.loc[missing_mask, "id"].tolist()[:50]
        working = working[~missing_mask]

        report.valid_records_remaining = len(working)

        self.logger.info(
            "Validation complete: %d loaded -> %d valid "
            "(dropped %d null-field, %d duplicate, %d missing-image)",
            report.total_records_loaded,
            report.valid_records_remaining,
            report.null_required_field_count,
            report.duplicate_id_count,
            report.missing_images_count,
        )

        return working.reset_index(drop=True), report
