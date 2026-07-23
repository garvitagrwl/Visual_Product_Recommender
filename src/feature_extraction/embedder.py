"""Embedding generation: run every image in a DataFrame through a
backbone and persist the resulting embeddings, id-aligned, to disk.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src.preprocessing.transforms import build_dataset


@dataclass
class EmbeddingRunMetadata:
    """Recorded alongside the embeddings so a saved .npz is self-describing."""

    backbone: str
    embedding_dim: int
    num_images: int
    image_size: list
    generated_at_unix: float
    generation_time_seconds: float


class EmbeddingGenerator:
    """Generates and persists embeddings for a set of images."""

    def __init__(
        self,
        model: tf.keras.Model,
        images_dir: str,
        image_size: tuple[int, int],
        backbone: str,
        batch_size: int,
        logger: logging.Logger,
    ):
        self.model = model
        self.images_dir = images_dir
        self.image_size = image_size
        self.backbone = backbone
        self.batch_size = batch_size
        self.logger = logger

    def generate(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, EmbeddingRunMetadata]:
        """Run the backbone over every image in `df`.

        Args:
            df: Metadata DataFrame with `id` and `image_filename` columns.

        Returns:
            (ids, embeddings, metadata) - `ids` is shape (N,), `embeddings`
            is shape (N, embedding_dim), row-aligned with `ids`.
        """
        ds = build_dataset(
            df=df,
            images_dir=self.images_dir,
            image_size=self.image_size,
            backbone=self.backbone,
            batch_size=self.batch_size,
            training=False,  # deterministic order, no augmentation
        )

        all_ids = []
        all_embeddings = []

        start = time.time()
        num_batches = 0
        for image_batch, id_batch in ds:
            embeddings = self.model(image_batch, training=False)
            all_embeddings.append(embeddings.numpy())
            all_ids.append(id_batch.numpy())
            num_batches += 1
            if num_batches % 10 == 0:
                self.logger.info("Processed %d batches...", num_batches)
        elapsed = time.time() - start

        ids = np.concatenate(all_ids).astype(str)
        embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)

        metadata = EmbeddingRunMetadata(
            backbone=self.backbone,
            embedding_dim=embeddings.shape[1],
            num_images=embeddings.shape[0],
            image_size=list(self.image_size),
            generated_at_unix=time.time(),
            generation_time_seconds=round(elapsed, 3),
        )

        self.logger.info(
            "Generated %d embeddings (dim=%d) in %.2fs (%.3fs/image)",
            metadata.num_images,
            metadata.embedding_dim,
            elapsed,
            elapsed / max(metadata.num_images, 1),
        )

        return ids, embeddings, metadata


def save_embeddings(
    ids: np.ndarray,
    embeddings: np.ndarray,
    metadata: EmbeddingRunMetadata,
    output_path: str,
    metadata_path: str,
) -> None:
    """Persist embeddings + ids to a .npz file, and metadata to JSON."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, ids=ids, embeddings=embeddings)

    meta_path = Path(metadata_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(metadata), f, indent=2)


def load_embeddings(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load ids and embeddings previously saved by `save_embeddings`."""
    data = np.load(path)
    return data["ids"], data["embeddings"]
