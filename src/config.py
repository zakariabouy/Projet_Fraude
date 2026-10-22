"""Project configuration: one YAML file, paths resolved from the project root."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "config.yaml"


def load_config(path: str | Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    """Return `path` as an absolute path, relative paths being taken from the project root."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path
