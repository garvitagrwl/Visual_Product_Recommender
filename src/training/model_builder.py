"""Builds the fine-tuning model for Phase 6: Transfer Learning.

Attaches a temporary classification head on top of the embedding
backbone from Phase 3. Since Keras's functional API shares layer
objects rather than copying them, training this combined model
updates the underlying embedding backbone's weights in place - so
after training, the backbone alone (no head) is already fine-tuned,
with no separate weight-transfer step needed.
"""

from __future__ import annotations

import tensorflow as tf

from src.feature_extraction.backbone import build_backbone


def freeze_all_but_last_n(model: tf.keras.Model, n: int) -> None:
    """Freeze all of a model's layers except the last `n`.

    Args:
        model: The backbone model (e.g. from build_backbone).
        n: Number of trailing layers to leave trainable. Earlier
            layers retain their pretrained ImageNet weights unchanged;
            these last layers adapt to the fashion-specific dataset.
    """
    if n >= len(model.layers):
        n = len(model.layers)

    for layer in model.layers[:-n]:
        layer.trainable = False
    for layer in model.layers[-n:]:
        layer.trainable = True


def build_finetune_model(
    backbone: str,
    image_size: tuple[int, int],
    num_classes: int,
    unfreeze_last_n_layers: int,
) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Build the fine-tuning model and return it alongside the backbone.

    Args:
        backbone: Backbone name ("resnet50" or "efficientnetb0").
        image_size: (height, width) input size.
        num_classes: Number of categories for the temporary
            classification head.
        unfreeze_last_n_layers: How many trailing backbone layers to
            leave trainable.

    Returns:
        (training_model, embedding_backbone) - `training_model` is
        used for `.fit()`; `embedding_backbone` is the same underlying
        model with no head, which will already reflect the fine-tuned
        weights once training completes (they share layer objects).
    """
    embedding_backbone = build_backbone(backbone, image_size)
    embedding_backbone.trainable = True  # Phase 3 froze it entirely; unfreeze for fine-tuning.
    freeze_all_but_last_n(embedding_backbone, unfreeze_last_n_layers)

    classification_head = tf.keras.layers.Dense(
        num_classes, activation="softmax", name="classification_head"
    )
    outputs = classification_head(embedding_backbone.output)
    training_model = tf.keras.Model(
        inputs=embedding_backbone.input, outputs=outputs, name="finetune_model"
    )

    return training_model, embedding_backbone
