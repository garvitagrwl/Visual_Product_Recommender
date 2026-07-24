"""Training entry point - Phase 6 (transfer learning, classification-
based fine-tuning) and Phase 7 (Siamese network, triplet loss), both
selected via --stage.

Usage:
    python train.py --stage transfer_learning
    python train.py --stage siamese
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.preprocessing.transforms import build_labeled_dataset, build_triplet_dataset
from src.training.model_builder import build_finetune_model
from src.training.siamese_model import build_siamese_model
from src.training.siamese_trainer import SiameseTrainer
from src.training.trainer import Trainer
from src.training.triplet_generator import TripletGenerator
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.seed import set_seed


def train_transfer_learning(config: dict, logger) -> None:
    """Phase 6: fine-tune via a temporary classification head."""
    backbone_name = config["model"]["backbone"]
    image_size = tuple(config["preprocessing"]["image_size"])
    categories = config["subset"]["categories"]
    category_column = config["subset"]["category_column"]

    train_df = pd.read_csv(config["data"]["train_csv"])
    val_df = pd.read_csv(config["data"]["val_csv"])
    logger.info("Loaded %d train / %d val images across %d categories", len(train_df), len(val_df), len(categories))

    train_ds = build_labeled_dataset(
        df=train_df,
        images_dir=config["dataset"]["images_dir"],
        image_size=image_size,
        backbone=backbone_name,
        batch_size=config["preprocessing"]["batch_size"],
        category_column=category_column,
        categories=categories,
        training=True,
        aug_config=config["preprocessing"]["augmentation"],
    )
    val_ds = build_labeled_dataset(
        df=val_df,
        images_dir=config["dataset"]["images_dir"],
        image_size=image_size,
        backbone=backbone_name,
        batch_size=config["preprocessing"]["batch_size"],
        category_column=category_column,
        categories=categories,
        training=False,
    )

    logger.info(
        "Building fine-tune model: %s, unfreezing last %d layers",
        backbone_name,
        config["training"]["unfreeze_last_n_layers"],
    )
    training_model, embedding_backbone = build_finetune_model(
        backbone=backbone_name,
        image_size=image_size,
        num_classes=len(categories),
        unfreeze_last_n_layers=config["training"]["unfreeze_last_n_layers"],
    )

    trainer = Trainer(
        training_model=training_model,
        embedding_backbone=embedding_backbone,
        learning_rate=config["training"]["learning_rate"],
        logger=logger,
    )
    history = trainer.fit(
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=config["training"]["epochs"],
        early_stopping_patience=config["training"]["early_stopping_patience"],
    )
    trainer.save(
        checkpoint_path=config["training"]["checkpoint_path"],
        history=history,
        history_path=config["training"]["history_path"],
    )


def train_siamese(config: dict, logger) -> None:
    """Phase 7: train a Siamese network with triplet loss."""
    backbone_name = config["model"]["backbone"]
    image_size = tuple(config["preprocessing"]["image_size"])
    category_column = config["subset"]["category_column"]

    train_triplets_path = config["data"]["train_triplets_csv"]
    val_triplets_path = config["data"]["val_triplets_csv"]

    if Path(train_triplets_path).exists() and Path(val_triplets_path).exists():
        logger.info("Loading existing triplets from %s / %s", train_triplets_path, val_triplets_path)
        train_triplets = pd.read_csv(train_triplets_path)
        val_triplets = pd.read_csv(val_triplets_path)
    else:
        logger.info("No cached triplets found - generating them now")
        train_df = pd.read_csv(config["data"]["train_csv"])
        val_df = pd.read_csv(config["data"]["val_csv"])

        generator = TripletGenerator(category_column=category_column, seed=config["seed"], logger=logger)
        train_triplets = generator.build(train_df)
        val_triplets = generator.build(val_df)

        Path(train_triplets_path).parent.mkdir(parents=True, exist_ok=True)
        train_triplets.to_csv(train_triplets_path, index=False)
        val_triplets.to_csv(val_triplets_path, index=False)
        logger.info("Saved triplets to %s / %s", train_triplets_path, val_triplets_path)

    train_ds = build_triplet_dataset(
        triplets_df=train_triplets,
        images_dir=config["dataset"]["images_dir"],
        image_size=image_size,
        backbone=backbone_name,
        batch_size=config["preprocessing"]["batch_size"],
        training=True,
    )
    val_ds = build_triplet_dataset(
        triplets_df=val_triplets,
        images_dir=config["dataset"]["images_dir"],
        image_size=image_size,
        backbone=backbone_name,
        batch_size=config["preprocessing"]["batch_size"],
        training=False,
    )

    logger.info(
        "Building Siamese model: %s, unfreezing last %d layers",
        backbone_name,
        config["siamese"]["unfreeze_last_n_layers"],
    )
    siamese_model, embedding_backbone = build_siamese_model(
        backbone=backbone_name,
        image_size=image_size,
        unfreeze_last_n_layers=config["siamese"]["unfreeze_last_n_layers"],
    )

    trainer = SiameseTrainer(
        siamese_model=siamese_model,
        embedding_backbone=embedding_backbone,
        learning_rate=config["siamese"]["learning_rate"],
        margin=config["siamese"]["margin"],
        logger=logger,
    )
    history = trainer.fit(
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=config["siamese"]["epochs"],
        early_stopping_patience=config["siamese"]["early_stopping_patience"],
    )
    trainer.save(
        checkpoint_path=config["siamese"]["checkpoint_path"],
        history=history,
        history_path=config["siamese"]["history_path"],
    )


def main(config_path: str, stage: str) -> None:
    config = load_config(config_path)
    set_seed(config["seed"])

    logger = get_logger(
        "train",
        level=config["logging"]["level"],
        log_file=config["logging"]["log_file"],
    )

    logger.info("=== Training: stage=%s ===", stage)

    if stage == "transfer_learning":
        train_transfer_learning(config, logger)
    elif stage == "siamese":
        train_siamese(config, logger)
    else:
        raise ValueError(f"Unknown stage: {stage}")

    logger.info("=== Training complete (stage=%s) ===", stage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training (transfer_learning / siamese)")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument(
        "--stage",
        type=str,
        default="transfer_learning",
        choices=["transfer_learning", "siamese"],
        help="Which training procedure to run",
    )
    args = parser.parse_args()
    main(args.config, args.stage)