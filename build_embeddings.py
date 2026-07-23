"""Embedding generation entry point - Phase 3 (baseline) and Phase 6
(transfer learning) both use this script, distinguished by --stage.

Loads the configured backbone - fresh ImageNet weights for
"baseline", or the fine-tuned weights saved by train.py for
"transfer_learning" - runs it over the full dataset subset, and saves
embeddings to a stage-specific path so baseline embeddings are never
overwritten by a later stage. Skips regeneration if the target file
already exists, unless --force is passed.

Usage:
    python build_embeddings.py --stage baseline
    python build_embeddings.py --stage transfer_learning
    python build_embeddings.py --stage baseline --force   # regenerate even if cached
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.feature_extraction.backbone import build_backbone
from src.feature_extraction.embedder import EmbeddingGenerator, save_embeddings
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.seed import set_seed

STAGE_CONFIG_KEYS = {
    "baseline": {
        "output_path": "baseline_path",
        "metadata_path": "baseline_metadata_path",
    },
    "transfer_learning": {
        "output_path": "transfer_learning_path",
        "metadata_path": "transfer_learning_metadata_path",
    },
}


def main(config_path: str, stage: str, force: bool) -> None:
    config = load_config(config_path)
    set_seed(config["seed"])

    logger = get_logger(
        "build_embeddings",
        level=config["logging"]["level"],
        log_file=config["logging"]["log_file"],
    )

    logger.info("=== Embedding Generation: stage=%s ===", stage)

    keys = STAGE_CONFIG_KEYS[stage]
    output_path = config["embeddings"][keys["output_path"]]
    metadata_path = config["embeddings"][keys["metadata_path"]]

    if Path(output_path).exists() and not force:
        logger.info(
            "Embeddings already exist at %s - skipping generation. "
            "Use --force to regenerate.",
            output_path,
        )
        return

    df = pd.read_csv(config["data"]["full_subset_csv"])
    df["id"] = df["id"].astype(str)
    logger.info("Generating embeddings for %d images", len(df))

    backbone_name = config["model"]["backbone"]
    image_size = tuple(config["preprocessing"]["image_size"])

    logger.info("Loading backbone: %s", backbone_name)
    model = build_backbone(backbone_name, image_size)

    if stage == "transfer_learning":
        checkpoint_path = config["training"]["checkpoint_path"]
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(
                f"No fine-tuned weights found at {checkpoint_path}. "
                "Run `python train.py` (Phase 6) first."
            )
        model.load_weights(checkpoint_path)
        logger.info("Loaded fine-tuned weights from %s", checkpoint_path)

    generator = EmbeddingGenerator(
        model=model,
        images_dir=config["dataset"]["images_dir"],
        image_size=image_size,
        backbone=backbone_name,
        batch_size=config["preprocessing"]["batch_size"],
        logger=logger,
    )
    ids, embeddings, metadata = generator.generate(df)

    save_embeddings(ids, embeddings, metadata, output_path, metadata_path)
    logger.info("Saved embeddings to %s", output_path)
    logger.info("Saved metadata to %s", metadata_path)
    logger.info("=== Embedding generation complete (stage=%s) ===", stage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embedding Generation (baseline / transfer_learning)")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument(
        "--stage",
        type=str,
        default="baseline",
        choices=list(STAGE_CONFIG_KEYS.keys()),
        help="Which embeddings to generate: 'baseline' (fresh ImageNet) or "
        "'transfer_learning' (requires train.py to have run first)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate embeddings even if a cached file already exists",
    )
    args = parser.parse_args()
    main(args.config, args.stage, args.force)