"""YAML config loader with path templating.

Loads configs/default.yaml, optionally merges an override config, and
resolves {drive} / {local} placeholders using paths.drive_root and
paths.colab_local.

Called by: all scripts (01-08), src/training/trainer.py,
           src/training/continual_trainer.py, notebooks/colab_setup.ipynb
Reads: configs/default.yaml (or a custom config path)
Writes: none
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


REQUIRED_TOP_LEVEL_KEYS = (
    "teacher",
    "student",
    "data",
    "training",
    "evaluation",
    "edge",
    "paths",
)


def load_config(
    config_path: str | Path = "configs/default.yaml",
    override_path: str | Path | None = None,
    resolve_paths: bool = True,
) -> dict[str, Any]:
    """Load a YAML config file and return the parsed dict.

    Args:
        config_path: Path to the base YAML file.
        override_path: Optional path to a YAML file whose keys override
            the base config (deep-merged).
        resolve_paths: If True, replaces the substrings ``{drive}`` and
            ``{local}`` in every string leaf with ``paths.drive_root``
            and ``paths.colab_local`` respectively.

    Returns:
        A plain ``dict`` containing the merged config.

    Raises:
        FileNotFoundError: If ``config_path`` does not exist.
        ValueError: If the config is missing required top-level keys.
    """
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f) or {}

    if override_path is not None:
        override_path = Path(override_path)
        if not override_path.is_file():
            raise FileNotFoundError(f"Override config not found: {override_path}")
        with open(override_path, "r", encoding="utf-8") as f:
            override = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, override)

    _validate(cfg)

    if resolve_paths:
        drive = cfg["paths"].get("drive_root", "")
        local = cfg["paths"].get("colab_local", "")
        cfg = _resolve_placeholders(cfg, {"drive": drive, "local": local})

    return cfg


def _validate(cfg: dict[str, Any]) -> None:
    """Ensure required top-level keys exist. Fail fast with a clear message."""
    missing = [k for k in REQUIRED_TOP_LEVEL_KEYS if k not in cfg]
    if missing:
        raise ValueError(
            f"Config is missing required top-level keys: {missing}. "
            f"Present keys: {list(cfg.keys())}"
        )


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursive dict merge. Values in ``override`` win over ``base``."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _resolve_placeholders(obj: Any, substitutions: dict[str, str]) -> Any:
    """Walk the config tree and substitute {placeholder} tokens in strings."""
    if isinstance(obj, dict):
        return {k: _resolve_placeholders(v, substitutions) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_placeholders(item, substitutions) for item in obj]
    if isinstance(obj, str):
        out = obj
        for token, replacement in substitutions.items():
            out = out.replace("{" + token + "}", str(replacement))
        return out
    return obj
