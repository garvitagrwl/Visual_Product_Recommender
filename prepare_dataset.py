"""Phase 1 entry point: Dataset Preparation.

Pipeline:
    load metadata -> validate -> mandatory subset sample -> train/val split
    -> write data/subset_full.csv, data/train.csv, data/val.csv,
       data/validation_report.json

Usage:
    python prepare_dataset.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.preprocessing.metadata_loader import MetadataLoader
from src.preprocessing.splitter import DatasetSplitter
from src.preprocessing.subset_sampler import SubsetSampler
from src.preprocessing.validator import DatasetValidator
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.seed import set_seed


def main(config_path: str) -> None:
    config = load_config(config_path)
    set_seed(config["seed"])

    logger = get_logger(
        "prepare_dataset",
        level=config["logging"]["level"],
        log_file=config["logging"]["log_file"],
    )

    logger.info("=== Phase 1: Dataset Preparation ===")

    # 1. Load metadata
    loader = MetadataLoader(
        styles_csv=config["dataset"]["styles_csv"],
        images_csv=config["dataset"]["images_csv"],
    )
    df = loader.load()
    logger.info("Loaded %d raw metadata records", len(df))

    # 2. Validate: structure, missing images, invalid records
    validator = DatasetValidator(images_dir=config["dataset"]["images_dir"], logger=logger)
    validator.validate_structure()
    clean_df, report = validator.validate(df)

    # 3. Mandatory subset sampling
    sampler = SubsetSampler(
        category_column=config["subset"]["category_column"],
        categories=config["subset"]["categories"],
        min_samples_per_category=config["subset"]["min_samples_per_category"],
        max_samples_per_category=config["subset"]["max_samples_per_category"],
        seed=config["seed"],
        logger=logger,
    )
    sampler.report_available_categories(clean_df)
    subset_df = sampler.sample(clean_df)

    # 4. Stratified train/val split
    splitter = DatasetSplitter(
        train_ratio=config["split"]["train_ratio"],
        stratify_column=config["split"]["stratify_column"],
        seed=config["seed"],
        logger=logger,
    )
    train_df, val_df = splitter.split(subset_df)

    # 5. Persist outputs
    processed_dir = Path(config["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    subset_df.to_csv(config["data"]["full_subset_csv"], index=False)
    train_df.to_csv(config["data"]["train_csv"], index=False)
    val_df.to_csv(config["data"]["val_csv"], index=False)

    report_path = Path(config["data"]["validation_report_json"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)

    logger.info("Wrote %s (%d rows)", config["data"]["full_subset_csv"], len(subset_df))
    logger.info("Wrote %s (%d rows)", config["data"]["train_csv"], len(train_df))
    logger.info("Wrote %s (%d rows)", config["data"]["val_csv"], len(val_df))
    logger.info("Wrote %s", report_path)
    logger.info("=== Phase 1 complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: Dataset Preparation")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()
    main(args.config)
