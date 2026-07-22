"""Deterministic seeding utility.

Called at the start of any script that involves randomness (sampling,
splitting, training, augmentation) so results are reproducible.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Seed all relevant random number generators.

    Args:
        seed: Seed value to use across `random`, `numpy`, and (if
            installed) `tensorflow`.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import tensorflow as tf  # noqa: WPS433 (local import - optional dep at this phase)

        tf.random.set_seed(seed)
    except ImportError:
        # TensorFlow isn't required until Phase 3 (feature extraction),
        # so its absence during Phase 1 (pure pandas/numpy work) is fine.
        pass
