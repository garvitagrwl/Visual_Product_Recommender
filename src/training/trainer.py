"""Training loop orchestration for Phase 6: Transfer Learning."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import tensorflow as tf


class Trainer:
    """Compiles and fits the fine-tuning model, then saves results."""

    def __init__(
        self,
        training_model: tf.keras.Model,
        embedding_backbone: tf.keras.Model,
        learning_rate: float,
        logger: logging.Logger,
    ):
        self.training_model = training_model
        self.embedding_backbone = embedding_backbone
        self.logger = logger

        self.training_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    def fit(
        self,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        epochs: int,
        early_stopping_patience: int,
    ) -> dict:
        """Train the model with early stopping on validation accuracy.

        Returns:
            The Keras training history as a plain dict (JSON-serializable).
        """
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=early_stopping_patience,
            restore_best_weights=True,
        )

        history = self.training_model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=[early_stopping],
        )

        best_val_acc = max(history.history.get("val_accuracy", [0.0]))
        self.logger.info("Training complete. Best val_accuracy: %.4f", best_val_acc)

        return {k: [float(v) for v in vals] for k, vals in history.history.items()}

    def save(self, checkpoint_path: str, history: dict, history_path: str) -> None:
        """Save the fine-tuned backbone's weights and training history.

        Note: only `embedding_backbone`'s weights are saved (no
        classification head) - that's the model Phase 4/5's retrieval
        pipeline actually needs.
        """
        ckpt_path = Path(checkpoint_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding_backbone.save_weights(str(ckpt_path))
        self.logger.info("Saved fine-tuned backbone weights to %s", ckpt_path)

        hist_path = Path(history_path)
        hist_path.parent.mkdir(parents=True, exist_ok=True)
        with hist_path.open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        self.logger.info("Saved training history to %s", hist_path)
