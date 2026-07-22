"""Configuration loading utilities.

Centralizes access to the project's YAML config so that no module
hardcodes paths or hyperparameters directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigError(Exception):
    """Raised when the configuration file is missing or malformed."""


def load_config(config_path: str | Path = "configs/config.yaml") -> Dict[str, Any]:
    """Load the project YAML configuration.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Parsed configuration as a nested dictionary.

    Raises:
        ConfigError: If the file does not exist or cannot be parsed.
    """
    path = Path(config_path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ConfigError(f"Failed to parse config file {path}: {exc}") from exc

    if not isinstance(config, dict):
        raise ConfigError(f"Config file {path} did not parse to a dictionary")

    return config
