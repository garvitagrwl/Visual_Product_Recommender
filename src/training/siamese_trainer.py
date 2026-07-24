"""Training loop orchestration for Phase 7: Siamese Network."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import tensorflow as tf

from src.training.losses import make_triplet_accuracy, make_triplet_loss


class SiameseTrainer:
    """Compiles and fits the Siamese model, then saves results."""

    def __init__(
        self,
        siamese_model: tf.keras.Model,
        embedding_backbone: tf.keras.Model,
        learning_rate: float,
        margin: float,
        logger: logging.Logger,
    ):
        self.siamese_model = siamese_model
        self.embedding_backbone = embedding_backbone
        self.logger = logger

        self.siamese_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=make_triplet_loss(margin),
            metrics=[make_triplet_accuracy()],
        )

    def fit(
        self,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        epochs: int,
        early_stopping_patience: int,
    ) -> dict:
        """Train with early stopping on validation triplet accuracy.

        Returns:
            The Keras training history as a plain dict (JSON-serializable).
        """
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_triplet_accuracy",
            patience=early_stopping_patience,
            restore_best_weights=True,
            mode="max",
        )

        history = self.siamese_model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=[early_stopping],
        )

        best_val_acc = max(history.history.get("val_triplet_accuracy", [0.0]))
        self.logger.info(
            "Training complete. Best val_triplet_accuracy: %.4f "
            "(fraction of triplets where positive was correctly closer than negative)",
            best_val_acc,
        )

        return {k: [float(v) for v in vals] for k, vals in history.history.items()}

    def save(self, checkpoint_path: str, history: dict, history_path: str) -> None:
        """Save the trained embedding backbone's weights and training history."""
        ckpt_path = Path(checkpoint_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding_backbone.save_weights(str(ckpt_path))
        self.logger.info("Saved Siamese-trained backbone weights to %s", ckpt_path)

        hist_path = Path(history_path)
        hist_path.parent.mkdir(parents=True, exist_ok=True)
        with hist_path.open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        self.logger.info("Saved training history to %s", hist_path)
