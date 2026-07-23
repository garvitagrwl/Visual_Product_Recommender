"""Image preprocessing: resize, backbone-correct normalization, and
optional training-time augmentation, assembled as a tf.data pipeline.

This module is shared by every later phase:
- Phase 3 (baseline) and Phase 4 (similarity search) use it with
  training=False to get deterministic embeddings.
- Phase 6 (transfer learning) and Phase 7 (Siamese) use it with
  training=True to get augmented batches during training.
"""

from __future__ import annotations

from typing import Callable, Tuple

import pandas as pd
import tensorflow as tf

# Each backbone was trained with a different input convention -
# using the wrong one silently produces bad embeddings with no error,
# so this mapping is the single source of truth for that choice.
_PREPROCESS_FUNCTIONS = {
    "resnet50": tf.keras.applications.resnet50.preprocess_input,
    "efficientnetb0": tf.keras.applications.efficientnet.preprocess_input,
}


class UnsupportedBackboneError(Exception):
    """Raised when configs/config.yaml names a backbone we don't handle."""


def get_preprocess_fn(backbone: str) -> Callable[[tf.Tensor], tf.Tensor]:
    """Return the correct `preprocess_input` function for a backbone.

    Args:
        backbone: One of "resnet50", "efficientnetb0" (matches
            configs/config.yaml `model.backbone`).

    Returns:
        A function mapping a float32 [0, 255]-range image tensor to
        the input convention that backbone's pretrained weights expect.

    Raises:
        UnsupportedBackboneError: If `backbone` isn't recognized.
    """
    key = backbone.lower()
    if key not in _PREPROCESS_FUNCTIONS:
        raise UnsupportedBackboneError(
            f"Unknown backbone '{backbone}'. Supported: {list(_PREPROCESS_FUNCTIONS)}"
        )
    return _PREPROCESS_FUNCTIONS[key]


def build_augmentation_layer(aug_config: dict) -> tf.keras.Sequential:
    """Build a Sequential of augmentation layers from config.

    Args:
        aug_config: The `preprocessing.augmentation` block from config.

    Returns:
        A `tf.keras.Sequential` applying flip/rotation/zoom/contrast.
        Intended for training pipelines only.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip(aug_config["random_flip"]),
            tf.keras.layers.RandomRotation(aug_config["random_rotation"]),
            tf.keras.layers.RandomZoom(aug_config["random_zoom"]),
            tf.keras.layers.RandomContrast(aug_config["random_contrast"]),
        ],
        name="augmentation",
    )


def _load_image(path: tf.Tensor, image_size: Tuple[int, int]) -> tf.Tensor:
    """Read a JPEG from disk and resize it to `image_size`."""
    raw = tf.io.read_file(path)
    image = tf.image.decode_jpeg(raw, channels=3)
    image = tf.image.resize(image, image_size)
    return tf.cast(image, tf.float32)


def build_dataset(
    df: pd.DataFrame,
    images_dir: str,
    image_size: Tuple[int, int],
    backbone: str,
    batch_size: int,
    training: bool = False,
    aug_config: dict | None = None,
) -> tf.data.Dataset:
    """Build a tf.data pipeline: read -> resize -> normalize -> (augment) -> batch.

    Args:
        df: Metadata DataFrame with an `id` and `image_filename` column
            (as produced by Phase 1).
        images_dir: Directory containing the image files.
        image_size: Target (height, width), e.g. (224, 224).
        backbone: Backbone name, used to select the correct
            preprocess_input normalization.
        batch_size: Batch size for the returned dataset.
        training: If True, applies augmentation (requires `aug_config`)
            and shuffles. If False, returns a deterministic, ordered
            pipeline suitable for embedding generation.
        aug_config: The `preprocessing.augmentation` config block.
            Required if `training=True` and `augmentation.enabled`.

    Returns:
        A batched, prefetched `tf.data.Dataset` yielding
        (preprocessed_image_batch, id_batch).
    """
    paths = (images_dir.rstrip("/") + "/" + df["image_filename"]).tolist()
    ids = df["id"].astype(str).tolist()

    ds = tf.data.Dataset.from_tensor_slices((paths, ids))

    if training:
        ds = ds.shuffle(buffer_size=len(paths), seed=42, reshuffle_each_iteration=True)

    preprocess_fn = get_preprocess_fn(backbone)

    def _map_fn(path, image_id):
        image = _load_image(path, image_size)
        image = preprocess_fn(image)
        return image, image_id

    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)

    if training and aug_config is not None and aug_config.get("enabled", False):
        aug_layer = build_augmentation_layer(aug_config)
        ds = ds.map(
            lambda image, image_id: (aug_layer(image, training=True), image_id),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
