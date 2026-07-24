"""Builds the Siamese network for Phase 7.

Three inputs (anchor, positive, negative) all pass through the SAME
embedding backbone (shared weights - a true Siamese network, not
three separate models). Their embeddings are stacked into one output
tensor for the triplet loss function to consume.

As in Phase 6, `embedding_backbone` shares layer objects with the
larger training model, so after training it already holds the
fine-tuned weights - no separate weight transfer needed.
"""

from __future__ import annotations

import tensorflow as tf

from src.feature_extraction.backbone import build_backbone
from src.training.model_builder import freeze_all_but_last_n


def build_siamese_model(
    backbone: str,
    image_size: tuple[int, int],
    unfreeze_last_n_layers: int,
) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Build the Siamese training model and return it with the embedding backbone.

    Args:
        backbone: Backbone name ("resnet50" or "efficientnetb0").
        image_size: (height, width) input size.
        unfreeze_last_n_layers: How many trailing backbone layers to
            leave trainable.

    Returns:
        (siamese_model, embedding_backbone) - `siamese_model` takes
        [anchor, positive, negative] and outputs stacked embeddings of
        shape (batch, 3, embedding_dim); `embedding_backbone` is the
        shared single-image embedding model, usable standalone once
        trained (same object build_embeddings.py's build_backbone()
        would produce).
    """
    embedding_backbone = build_backbone(backbone, image_size)
    embedding_backbone.trainable = True
    freeze_all_but_last_n(embedding_backbone, unfreeze_last_n_layers)

    input_shape = (*image_size, 3)
    anchor_input = tf.keras.Input(shape=input_shape, name="anchor")
    positive_input = tf.keras.Input(shape=input_shape, name="positive")
    negative_input = tf.keras.Input(shape=input_shape, name="negative")

    # Same `embedding_backbone` object called three times => shared
    # weights, i.e. an actual Siamese architecture.
    anchor_embedding = embedding_backbone(anchor_input)
    positive_embedding = embedding_backbone(positive_input)
    negative_embedding = embedding_backbone(negative_input)

    stacked = tf.keras.layers.Lambda(
        lambda embeddings: tf.stack(embeddings, axis=1),
        output_shape=lambda input_shapes: (input_shapes[0][0], 3, input_shapes[0][1]),
        name="stack_embeddings",
    )([anchor_embedding, positive_embedding, negative_embedding])

    siamese_model = tf.keras.Model(
        inputs=[anchor_input, positive_input, negative_input],
        outputs=stacked,
        name="siamese_model",
    )

    return siamese_model, embedding_backbone
