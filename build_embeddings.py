"""Embedding generation entry point - shared by Phase 3 (baseline),
Phase 6 (transfer learning), and Phase 7 (siamese), distinguished by
--stage.

Loads the configured backbone - fresh ImageNet weights for
"baseline", or the trained weights saved by train.py for the other
stages - runs it over the full dataset subset, and saves embeddings
to a stage-specific path so no stage overwrites another. Skips
regeneration if the target file already exists, unless --force is
passed.

STAGE_CONFIG maps each stage to (a) where its embeddings/metadata are
saved and (b) where its trained weights live, if any. Adding a future
stage means adding one entry here, not new branching logic - this
same map is reused by recommend.py so both scripts always agree on
where each stage's weights live.

Usage:
    python build_embeddings.py --stage baseline
    python build_embeddings.py --stage transfer_learning
    python build_embeddings.py --stage siamese
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

# checkpoint_config is None for "baseline" (uses fresh ImageNet weights,
# nothing to load) and (config_section, config_key) for trained stages.
STAGE_CONFIG = {
    "baseline": {
        "output_path": "baseline_path",
        "metadata_path": "baseline_metadata_path",
        "checkpoint_config": None,
    },
    "transfer_learning": {
        "output_path": "transfer_learning_path",
        "metadata_path": "transfer_learning_metadata_path",
        "checkpoint_config": ("training", "checkpoint_path"),
    },
    "siamese": {
        "output_path": "siamese_path",
        "metadata_path": "siamese_metadata_path",
        "checkpoint_config": ("siamese", "checkpoint_path"),
    },
}

# Kept as an alias for backward compatibility with earlier phases'
# imports (recommend.py's Phase 6 version imported this exact name).
STAGE_CONFIG_KEYS = STAGE_CONFIG


def load_stage_model(config: dict, stage: str, backbone_name: str, image_size: tuple, logger) -> "tf.keras.Model":
    """Build the backbone and, for trained stages, load its checkpoint.

    Shared by build_embeddings.py and recommend.py so both scripts are
    guaranteed to load a stage's weights identically.
    """
    model = build_backbone(backbone_name, image_size)

    checkpoint_config = STAGE_CONFIG[stage]["checkpoint_config"]
    if checkpoint_config is not None:
        section, key = checkpoint_config
        checkpoint_path = config[section][key]
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(
                f"No trained weights found at {checkpoint_path} for stage '{stage}'. "
                f"Run `python train.py --stage {stage}` first, or use --stage baseline."
            )
        model.load_weights(checkpoint_path)
        logger.info("Loaded '%s' weights from %s", stage, checkpoint_path)

    return model


def main(config_path: str, stage: str, force: bool) -> None:
    config = load_config(config_path)
    set_seed(config["seed"])

    logger = get_logger(
        "build_embeddings",
        level=config["logging"]["level"],
        log_file=config["logging"]["log_file"],
    )

    logger.info("=== Embedding Generation: stage=%s ===", stage)

    keys = STAGE_CONFIG[stage]
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
    model = load_stage_model(config, stage, backbone_name, image_size, logger)

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
    parser = argparse.ArgumentParser(description="Embedding Generation (baseline / transfer_learning / siamese)")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument(
        "--stage",
        type=str,
        default="baseline",
        choices=list(STAGE_CONFIG.keys()),
        help="Which embeddings to generate",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate embeddings even if a cached file already exists",
    )
    args = parser.parse_args()
    main(args.config, args.stage, args.force)