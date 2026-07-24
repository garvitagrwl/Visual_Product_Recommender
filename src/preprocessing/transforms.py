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


def load_and_preprocess_single_image(
    path: str,
    image_size: Tuple[int, int],
    backbone: str,
) -> tf.Tensor:
    """Preprocess one image file the same way the catalog was preprocessed.

    Used by the Phase 4 query pipeline for an uploaded image - reuses
    the exact same resize + backbone-correct normalization as
    `build_dataset(..., training=False)`, which is required for query
    and catalog embeddings to be comparable.

    Args:
        path: Path to the image file.
        image_size: Target (height, width), must match what the
            catalog embeddings were generated with.
        backbone: Backbone name, selects the matching preprocess_input.

    Returns:
        A (1, H, W, 3) float32 tensor ready to feed into the backbone.
    """
    image = _load_image(tf.constant(path), image_size)
    preprocess_fn = get_preprocess_fn(backbone)
    image = preprocess_fn(image)
    return tf.expand_dims(image, axis=0)


def build_labeled_dataset(
    df: pd.DataFrame,
    images_dir: str,
    image_size: Tuple[int, int],
    backbone: str,
    batch_size: int,
    category_column: str,
    categories: list,
    training: bool = False,
    aug_config: dict | None = None,
) -> tf.data.Dataset:
    """Build a tf.data pipeline yielding (image, integer_label) pairs.

    Used for Phase 6 fine-tuning, where a temporary classification
    head needs a label per image. `categories` fixes the label index
    order so it's identical between training and any later reuse of
    the same encoding.

    Args:
        df: Metadata DataFrame with `image_filename` and the category
            column (as produced by Phase 1).
        images_dir: Directory containing the image files.
        image_size: Target (height, width).
        backbone: Backbone name, for correct normalization.
        batch_size: Batch size.
        category_column: Column name holding the category label
            (e.g. "articleType").
        categories: Fixed, ordered list of category strings - defines
            the integer label each maps to (index in this list).
        training: If True, shuffles and applies augmentation.
        aug_config: Required if `training=True` and augmentation is enabled.

    Returns:
        A batched, prefetched `tf.data.Dataset` yielding
        (image_batch, label_batch) where labels are integer class indices.
    """
    category_to_index = {cat: i for i, cat in enumerate(categories)}
    paths = (images_dir.rstrip("/") + "/" + df["image_filename"]).tolist()
    labels = df[category_column].map(category_to_index).tolist()

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(buffer_size=len(paths), seed=42, reshuffle_each_iteration=True)

    preprocess_fn = get_preprocess_fn(backbone)

    def _map_fn(path, label):
        image = _load_image(path, image_size)
        image = preprocess_fn(image)
        return image, label

    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)

    if training and aug_config is not None and aug_config.get("enabled", False):
        aug_layer = build_augmentation_layer(aug_config)
        ds = ds.map(
            lambda image, label: (aug_layer(image, training=True), label),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_triplet_dataset(
    triplets_df: pd.DataFrame,
    images_dir: str,
    image_size: Tuple[int, int],
    backbone: str,
    batch_size: int,
    training: bool = False,
) -> tf.data.Dataset:
    """Build a tf.data pipeline yielding ((anchor, positive, negative), dummy_label).

    The dummy label is a zero placeholder - Keras's `.fit()` expects a
    (x, y) signature, but the triplet loss function (src/training/losses.py)
    computes loss purely from the three embeddings and ignores y_true
    entirely.

    Args:
        triplets_df: Output of TripletGenerator.build().
        images_dir: Directory containing the image files.
        image_size: Target (height, width).
        backbone: Backbone name, for correct normalization.
        batch_size: Batch size.
        training: If True, shuffles triplets each epoch.

    Returns:
        A batched, prefetched `tf.data.Dataset`.
    """
    anchor_paths = (images_dir.rstrip("/") + "/" + triplets_df["anchor_filename"]).tolist()
    positive_paths = (images_dir.rstrip("/") + "/" + triplets_df["positive_filename"]).tolist()
    negative_paths = (images_dir.rstrip("/") + "/" + triplets_df["negative_filename"]).tolist()

    ds = tf.data.Dataset.from_tensor_slices((anchor_paths, positive_paths, negative_paths))

    if training:
        ds = ds.shuffle(buffer_size=len(anchor_paths), seed=42, reshuffle_each_iteration=True)

    preprocess_fn = get_preprocess_fn(backbone)

    def _map_fn(anchor_path, positive_path, negative_path):
        anchor = preprocess_fn(_load_image(anchor_path, image_size))
        positive = preprocess_fn(_load_image(positive_path, image_size))
        negative = preprocess_fn(_load_image(negative_path, image_size))
        dummy_label = tf.constant(0.0)
        return (anchor, positive, negative), dummy_label

    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


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