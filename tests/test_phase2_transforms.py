"""Verification script for Phase 2.

Checks:
1. Both supported backbones' preprocess_input produce the expected
   value ranges (sanity check that we're not accidentally using the
   wrong normalization).
2. The tf.data pipeline produces correctly-shaped batches for both
   training=False (deterministic, no augmentation) and training=True
   (augmented).
3. Two calls with training=False on the same data produce IDENTICAL
   output (critical: baseline embeddings must be reproducible).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow as tf
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.preprocessing.transforms import build_dataset, get_preprocess_fn
from src.utils.config import load_config


def check_preprocess_ranges():
    dummy = tf.random.uniform((224, 224, 3), 0, 255, dtype=tf.float32)

    resnet_fn = get_preprocess_fn("resnet50")
    resnet_out = resnet_fn(tf.identity(dummy)).numpy()
    print(
        f"[resnet50]        min={resnet_out.min():.2f} max={resnet_out.max():.2f} "
        f"(expect roughly centered around 0, e.g. [-124, 152])"
    )

    eff_fn = get_preprocess_fn("efficientnetb0")
    eff_out = eff_fn(tf.identity(dummy)).numpy()
    print(
        f"[efficientnetb0]  min={eff_out.min():.2f} max={eff_out.max():.2f} "
        f"(expect ~[0, 255], EfficientNet normalizes internally in the model)"
    )

    assert not np.allclose(resnet_out, eff_out), (
        "ResNet50 and EfficientNet preprocessing produced identical output - "
        "the backbone-specific normalization isn't actually being applied!"
    )
    print("PASS: the two backbones produce different (correctly distinct) preprocessing.\n")


def check_pipeline_shapes(config):
    df = pd.read_csv(config["data"]["train_csv"]).head(10)
    images_dir = config["dataset"]["images_dir"]
    image_size = tuple(config["preprocessing"]["image_size"])

    ds = build_dataset(
        df=df,
        images_dir=images_dir,
        image_size=image_size,
        backbone=config["model"]["backbone"],
        batch_size=4,
        training=False,
    )
    batch_images, batch_ids = next(iter(ds))
    print(f"Eval pipeline batch shape: {batch_images.shape}, dtype={batch_images.dtype}")
    assert batch_images.shape[1:] == (224, 224, 3)
    print("PASS: eval pipeline produces correctly shaped batches.\n")

    ds_train = build_dataset(
        df=df,
        images_dir=images_dir,
        image_size=image_size,
        backbone=config["model"]["backbone"],
        batch_size=4,
        training=True,
        aug_config=config["preprocessing"]["augmentation"],
    )
    batch_images_aug, _ = next(iter(ds_train))
    print(f"Train (augmented) pipeline batch shape: {batch_images_aug.shape}")
    assert batch_images_aug.shape[1:] == (224, 224, 3)
    print("PASS: training pipeline with augmentation produces correctly shaped batches.\n")


def check_determinism(config):
    df = pd.read_csv(config["data"]["val_csv"]).head(5)
    images_dir = config["dataset"]["images_dir"]
    image_size = tuple(config["preprocessing"]["image_size"])

    def get_first_batch():
        ds = build_dataset(
            df=df,
            images_dir=images_dir,
            image_size=image_size,
            backbone=config["model"]["backbone"],
            batch_size=5,
            training=False,
        )
        images, ids = next(iter(ds))
        return images.numpy(), ids.numpy()

    images_a, ids_a = get_first_batch()
    images_b, ids_b = get_first_batch()

    assert np.array_equal(ids_a, ids_b), "Order of ids changed between runs (should be fixed for eval)"
    assert np.allclose(images_a, images_b), "Pixel values changed between runs (eval pipeline must be deterministic)"
    print("PASS: eval pipeline (training=False) is fully deterministic across repeated runs.\n")


if __name__ == "__main__":
    config = load_config("configs/config.yaml")
    check_preprocess_ranges()
    check_pipeline_shapes(config)
    check_determinism(config)
    print("All Phase 2 checks passed.")
