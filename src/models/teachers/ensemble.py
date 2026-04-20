"""Weighted teacher ensemble aggregation.

Combines the per-teacher ``.npy`` probability arrays produced during
soft-label generation into a single ensemble ``.npy`` file, weighted by
each teacher's validation AUC (softmax over AUCs). Weights are written
alongside the ensemble as a JSON manifest for reproducibility.

Called by:
    scripts/03_generate_soft_labels.py
Reads:
    per-teacher soft-label arrays (in-memory or .npy files on disk)
        shape: (N,), dtype: float32, values in [0, 1]  (P(fake))
    val split CSV (for weighting)
        columns: face_path, frame_idx, video_id, label, dataset, generation, technique
Writes:
    ensemble .npy at {drive}/soft_labels/{generation}/ensemble.npy
        shape: (N,), dtype: float32, values in [0, 1]
    weights JSON at {drive}/soft_labels/{generation}/ensemble_weights.json
        {
            "generation": "gen1",
            "generated_at": "2026-04-18T14:02:31",   # ISO 8601 %Y-%m-%dT%H:%M:%S
            "weights":  {"efficientnet_b4": 0.35, "recce": 0.30, "clip_vit_l14": 0.35},
            "val_auc":  {"efficientnet_b4": 0.92, "recce": 0.88, "clip_vit_l14": 0.91},
            "weighting_temperature": 1.0,
            "num_samples": 160000
        }
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_val_auc(
    soft_labels: np.ndarray,
    val_labels: np.ndarray,
) -> float:
    """ROC-AUC between per-sample soft labels and hard val labels."""
    if soft_labels.shape != val_labels.shape:
        raise ValueError(
            f"Shape mismatch: soft={soft_labels.shape}, val={val_labels.shape}"
        )
    # Guard against a degenerate val split (only one class).
    if np.unique(val_labels).size < 2:
        logger.warning(
            "Only one class present in val labels; returning 0.5 AUC."
        )
        return 0.5
    return float(roc_auc_score(val_labels, soft_labels))


def softmax_weights(
    per_teacher_auc: dict[str, float],
    *,
    temperature: float = 1.0,
) -> dict[str, float]:
    """Turn per-teacher val AUCs into a softmax weighting.

    Args:
        per_teacher_auc: Mapping ``teacher_name -> val_auc``.
        temperature: Lower values = sharper weighting toward the best
            teacher. Default 1.0 is a simple softmax over raw AUCs.
    """
    if not per_teacher_auc:
        raise ValueError("per_teacher_auc must not be empty")
    names = list(per_teacher_auc.keys())
    aucs = np.array([per_teacher_auc[n] for n in names], dtype=np.float64)
    scaled = aucs / max(temperature, 1e-6)
    # Numerical-stability shift.
    scaled = scaled - scaled.max()
    exp = np.exp(scaled)
    weights = exp / exp.sum()
    return {name: float(w) for name, w in zip(names, weights)}


def aggregate(
    per_teacher_soft: dict[str, np.ndarray],
    weights: dict[str, float],
) -> np.ndarray:
    """Weighted average across teachers -> single (N,) ensemble array."""
    if not per_teacher_soft:
        raise ValueError("per_teacher_soft must not be empty")

    missing = set(per_teacher_soft) - set(weights)
    if missing:
        raise ValueError(f"Missing weights for teachers: {sorted(missing)}")

    # Ensure consistent shape.
    first_name = next(iter(per_teacher_soft))
    n = per_teacher_soft[first_name].shape[0]
    for name, arr in per_teacher_soft.items():
        if arr.shape != (n,):
            raise ValueError(
                f"Teacher '{name}' has shape {arr.shape}; expected ({n},)"
            )

    ensemble = np.zeros(n, dtype=np.float32)
    for name, arr in per_teacher_soft.items():
        ensemble += weights[name] * arr.astype(np.float32, copy=False)
    # Clip just in case weights don't sum to 1.0 due to rounding.
    return np.clip(ensemble, 0.0, 1.0)


def build_and_save_ensemble(
    *,
    per_teacher_soft: dict[str, np.ndarray],
    val_labels: np.ndarray | None,
    val_mask: np.ndarray | None,
    output_dir: str | Path,
    generation: str,
    weighting_temperature: float = 1.0,
) -> tuple[Path, Path]:
    """End-to-end: compute weights, aggregate, write .npy + .json.

    Args:
        per_teacher_soft: ``teacher_name -> (N,) float32 array`` for the
            full generation CSV (train + val + test concatenated in CSV
            order).
        val_labels: Hard labels for the val rows only (used to score
            teachers). Can be ``None`` to skip weighting and fall back to
            uniform weights.
        val_mask: Boolean array of shape ``(N,)`` marking which rows in
            ``per_teacher_soft`` belong to the val split. Required when
            ``val_labels`` is not ``None``.
        output_dir: Target directory for ``ensemble.npy`` and
            ``ensemble_weights.json``.
        generation: ``"gen1"`` / ``"gen2"`` / ``"gen3"`` — recorded in the
            manifest.
        weighting_temperature: Temperature for the softmax over AUCs.
    """
    if val_labels is not None and val_mask is None:
        raise ValueError("val_mask is required when val_labels is provided")

    if val_labels is not None and val_mask is not None:
        per_teacher_auc = {
            name: compute_val_auc(arr[val_mask], val_labels)
            for name, arr in per_teacher_soft.items()
        }
        weights = softmax_weights(
            per_teacher_auc, temperature=weighting_temperature
        )
    else:
        logger.warning(
            "No val labels supplied; falling back to uniform ensemble weights."
        )
        n_teachers = len(per_teacher_soft)
        weights = {name: 1.0 / n_teachers for name in per_teacher_soft}
        per_teacher_auc = {name: float("nan") for name in per_teacher_soft}

    ensemble = aggregate(per_teacher_soft, weights)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npy_path = out_dir / "ensemble.npy"
    json_path = out_dir / "ensemble_weights.json"

    np.save(npy_path, ensemble)
    manifest = {
        "generation": generation,
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "weights": weights,
        "val_auc": per_teacher_auc,
        "weighting_temperature": weighting_temperature,
        "num_samples": int(ensemble.shape[0]),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(
        "Saved ensemble for %s: %s (N=%d) with weights=%s",
        generation,
        npy_path,
        ensemble.shape[0],
        {k: round(v, 4) for k, v in weights.items()},
    )
    return npy_path, json_path


def build_val_mask(
    full_csv: str | Path,
    val_csv: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(val_mask, val_labels)`` aligned to the ordering of ``full_csv``.

    ``full_csv`` is the concatenated (train + val + test) split used to drive
    teacher inference; ``val_csv`` is the val split only. Alignment is done
    on the ``face_path`` column (unique per row).
    """
    full_df = pd.read_csv(full_csv, usecols=["face_path", "label"])
    val_df = pd.read_csv(val_csv, usecols=["face_path", "label"])

    val_face_to_label = dict(zip(val_df["face_path"], val_df["label"]))
    mask = full_df["face_path"].isin(val_face_to_label).to_numpy()
    labels = (
        full_df.loc[mask, "face_path"]
        .map(val_face_to_label)
        .to_numpy()
        .astype(np.int64)
    )
    return mask, labels
