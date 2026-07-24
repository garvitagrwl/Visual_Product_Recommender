"""Triplet loss for Siamese network training.

Embeddings are L2-normalized before computing distance. This isn't
arbitrary: for unit vectors, squared Euclidean distance and cosine
similarity are monotonically related
(squared_dist = 2 - 2*cosine_similarity). Since Phase 4's retrieval
index (CosineSimilarityIndex) also L2-normalizes and compares via
cosine similarity, training on normalized-embedding distance directly
optimizes the same space retrieval actually searches over.
"""

from __future__ import annotations

import tensorflow as tf


def l2_normalize(x: tf.Tensor) -> tf.Tensor:
    return tf.math.l2_normalize(x, axis=-1)


def make_triplet_loss(margin: float):
    """Return a Keras-compatible triplet loss function.

    Args:
        margin: How much further the negative should be than the
            positive, in squared-distance terms, before loss is zero.

    Returns:
        A function `loss(y_true, y_pred)` where `y_pred` has shape
        (batch, 3, embedding_dim) - stacked (anchor, positive,
        negative) embeddings. `y_true` is ignored (a dummy placeholder
        required by Keras's fit() signature).
    """

    def triplet_loss(y_true, y_pred):
        del y_true  # unused - loss is computed purely from the triplet embeddings

        anchor = l2_normalize(y_pred[:, 0, :])
        positive = l2_normalize(y_pred[:, 1, :])
        negative = l2_normalize(y_pred[:, 2, :])

        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)

        loss = tf.maximum(pos_dist - neg_dist + margin, 0.0)
        return tf.reduce_mean(loss)

    return triplet_loss


def make_triplet_accuracy():
    """Return a metric: fraction of triplets where positive is closer than negative.

    Not the loss itself - a human-interpretable "is this working"
    signal, since raw triplet loss values aren't intuitive on their own.
    """

    def triplet_accuracy(y_true, y_pred):
        del y_true

        anchor = l2_normalize(y_pred[:, 0, :])
        positive = l2_normalize(y_pred[:, 1, :])
        negative = l2_normalize(y_pred[:, 2, :])

        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)

        correct = tf.cast(pos_dist < neg_dist, tf.float32)
        return tf.reduce_mean(correct)

    return triplet_accuracy
