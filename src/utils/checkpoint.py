"""Save/load student checkpoints with bundled metadata.

Checkpoint layout (a single ``.pth`` file, ``torch.save``-able dict):

.. code-block:: python

    {
        "epoch":                int,           # 0-indexed epoch at save time
        "model_state_dict":     dict,          # student parameters
        "optimizer_state_dict": dict | None,   # optional — present during training
        "scheduler_state_dict": dict | None,   # optional
        "best_val_auc":         float,         # best val AUC seen so far
        "config":               dict,          # snapshot of the YAML config block
        "generation":           str,           # "gen1" / "gen2" / "gen3"
        "metrics":              dict,          # arbitrary train/val metrics dict
        "timestamp":            str,           # ISO 8601 ``%Y-%m-%dT%H:%M:%S``
    }

Called by:
    src/training/trainer.py
    src/training/continual_trainer.py
    scripts/04_initial_distillation.py
    scripts/05_continual_distillation.py
    scripts/07_edge_evaluation.py (load-only)
Reads / Writes: ``.pth`` checkpoint files under ``{drive}/checkpoints/students/``.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S"

# Keys the loader tolerates when ``state_dict`` is wrapped inside a container.
_STATE_DICT_CONTAINER_KEYS = ("model_state_dict", "state_dict", "model", "net")


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    epoch: int,
    best_val_auc: float,
    config: dict[str, Any] | None = None,
    generation: str | None = None,
    metrics: dict[str, Any] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    timestamp: str | None = None,
) -> Path:
    """Serialize a student checkpoint to ``path`` and return the resolved path.

    Creates the parent directory on demand. ``timestamp`` defaults to
    ``datetime.now()`` formatted in ISO 8601.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt: dict[str, Any] = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_val_auc": float(best_val_auc),
        "config": config or {},
        "generation": generation,
        "metrics": metrics or {},
        "timestamp": timestamp or datetime.now().strftime(_TIMESTAMP_FORMAT),
    }
    torch.save(ckpt, out_path)
    return out_path


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint and optionally restore model/optimizer/scheduler state.

    Returns the loaded dict (minus the tensor state) plus a
    ``"missing_keys"`` / ``"unexpected_keys"`` report when ``model`` is given.
    """
    ckpt_path = Path(path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    raw: Any = torch.load(ckpt_path, map_location=map_location)
    if not isinstance(raw, dict):
        raise ValueError(
            f"Expected dict checkpoint at {ckpt_path}, got {type(raw).__name__}"
        )

    state_dict = _unwrap_state_dict(raw)
    missing: list[str] = []
    unexpected: list[str] = []

    if model is not None:
        m_missing, m_unexpected = model.load_state_dict(state_dict, strict=strict)
        missing = list(m_missing)
        unexpected = list(m_unexpected)

    if optimizer is not None and raw.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(raw["optimizer_state_dict"])
    if scheduler is not None and raw.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(raw["scheduler_state_dict"])

    # Return a shallow copy without the heavy tensor dicts.
    trimmed = {
        k: v
        for k, v in raw.items()
        if k not in {"model_state_dict", "optimizer_state_dict", "scheduler_state_dict"}
    }
    trimmed["missing_keys"] = missing
    trimmed["unexpected_keys"] = unexpected
    trimmed["checkpoint_path"] = str(ckpt_path)
    return trimmed


def _unwrap_state_dict(raw: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Find the actual ``state_dict`` inside a saved checkpoint container."""
    for key in _STATE_DICT_CONTAINER_KEYS:
        value = raw.get(key)
        if isinstance(value, dict) and value and all(
            isinstance(k, str) for k in value.keys()
        ):
            return value  # type: ignore[return-value]
    # Fall through: assume the whole dict *is* the state_dict (bare tensors).
    if all(isinstance(v, torch.Tensor) for v in raw.values()):
        return raw  # type: ignore[return-value]
    raise ValueError(
        "Could not locate state_dict inside checkpoint; "
        f"expected one of {_STATE_DICT_CONTAINER_KEYS} or a bare tensor mapping."
    )
