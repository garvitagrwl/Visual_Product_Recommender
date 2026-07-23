"""Phase 6 entry point: Transfer Learning.

Fine-tunes the backbone's later layers on a temporary classification
task (predicting the subset's categories), then saves the fine-tuned
backbone's weights for reuse by build_embeddings.py --stage transfer_learning.

Usage:
    python train.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.preprocessing.transforms import build_labeled_dataset
from src.training.model_builder import build_finetune_model
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.seed import set_seed


def main(config_path: str) -> None:
    config = load_config(config_path)
    set_seed(config["seed"])

    logger = get_logger(
        "train",
        level=config["logging"]["level"],
        log_file=config["logging"]["log_file"],
    )

    logger.info("=== Phase 6: Transfer Learning ===")

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

    logger.info("=== Phase 6 complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 6: Transfer Learning")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
