"""Pretrained backbone loading, with the classification head removed.

`include_top=False, pooling="avg"` does two things in one step:
- drops the original 1000-class ImageNet classification head
- applies global average pooling, so the model's output IS the
  embedding vector (2048-D for ResNet50, 1280-D for EfficientNetB0)
  rather than a spatial feature map that would need further reduction.
"""

from __future__ import annotations

import tensorflow as tf

from src.preprocessing.transforms import UnsupportedBackboneError

_EMBEDDING_DIMS = {
    "resnet50": 2048,
    "efficientnetb0": 1280,
}


def get_embedding_dim(backbone: str) -> int:
    """Return the known output embedding dimension for a backbone."""
    key = backbone.lower()
    if key not in _EMBEDDING_DIMS:
        raise UnsupportedBackboneError(
            f"Unknown backbone '{backbone}'. Supported: {list(_EMBEDDING_DIMS)}"
        )
    return _EMBEDDING_DIMS[key]


def build_backbone(backbone: str, image_size: tuple[int, int]) -> tf.keras.Model:
    """Load a pretrained backbone with its classification head removed.

    Args:
        backbone: One of "resnet50", "efficientnetb0".
        image_size: (height, width) the model should expect.

    Returns:
        A `tf.keras.Model` mapping a preprocessed image batch directly
        to an embedding batch (no classification head, no extra
        pooling layer needed - `pooling="avg"` already does that).

    Raises:
        UnsupportedBackboneError: If `backbone` isn't recognized.
    """
    key = backbone.lower()
    input_shape = (*image_size, 3)

    if key == "resnet50":
        model = tf.keras.applications.ResNet50(
            include_top=False,
            weights="imagenet",
            pooling="avg",
            input_shape=input_shape,
        )
    elif key == "efficientnetb0":
        model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            pooling="avg",
            input_shape=input_shape,
        )
    else:
        raise UnsupportedBackboneError(
            f"Unknown backbone '{backbone}'. Supported: {list(_EMBEDDING_DIMS)}"
        )

    model.trainable = False  # Phase 3 is feature extraction only, no training.
    return model
