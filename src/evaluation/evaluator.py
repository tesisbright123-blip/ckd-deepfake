"""Evaluation pipeline for per-generation and cross-generation assessment.

Runs the student on one or more evaluation loaders, returns per-split
``BinaryMetrics`` (AUC, log-loss, accuracy), and — when invoked across
multiple generations — aggregates a ``CGRS`` summary for continual-learning
comparisons.

Called by:
    src/training/trainer.py (per-epoch val + per-generation test hooks)
    src/training/continual_trainer.py
    scripts/04_initial_distillation.py
    scripts/05_continual_distillation.py
    scripts/06_ablation_study.py
Reads:
    Student ``nn.Module`` in memory (or a checkpoint path loaded via
    :mod:`src.utils.checkpoint` before calling this module).
    Val/test DataLoaders yielding ``(image, hard_label, soft_label, meta)``
    tuples produced by :class:`~src.data.dataset.DeepfakeDataset`.
Writes:
    Optional JSON summary via :func:`write_metrics_json` — layout::

        {
          "generation": "gen1",
          "generated_at": "2026-04-18T14:02:31",     # ISO 8601 %Y-%m-%dT%H:%M:%S
          "splits": {
              "gen1_test": {"auc": 0.94, "log_loss": 0.18,
                            "accuracy": 0.89, "num_samples": 24000}
          }
        }
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.evaluation.metrics import (
    BinaryMetrics,
    compute_average_forgetting,
    compute_binary_metrics,
    compute_cgrs,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S"


@torch.inference_mode()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: str | torch.device = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    """Run ``model`` over ``loader`` and return ``(y_true, y_prob)`` numpy arrays.

    ``y_prob`` is the softmax fake-class probability (index 1). The loader is
    expected to be non-shuffled so the order matches the input CSV.
    """
    model.eval()
    was_on = next(model.parameters()).device
    model.to(device)

    probs_chunks: list[np.ndarray] = []
    labels_chunks: list[np.ndarray] = []
    for batch in loader:
        images, hard_labels, *_ = batch
        images = images.to(device, non_blocking=True)
        logits = model(images)
        if logits.ndim != 2 or logits.size(1) < 2:
            raise RuntimeError(
                f"Expected (B, >=2) logits from student, got {tuple(logits.shape)}"
            )
        p_fake = F.softmax(logits, dim=1)[:, 1]
        probs_chunks.append(p_fake.detach().float().cpu().numpy())
        labels_chunks.append(
            hard_labels.detach().cpu().numpy().astype(np.int64, copy=False)
        )

    # Restore whatever device the caller had the model on.
    model.to(was_on)

    if not probs_chunks:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return (
        np.concatenate(probs_chunks).astype(np.float32, copy=False),
        np.concatenate(labels_chunks).astype(np.int64, copy=False),
    )


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: str | torch.device = "cuda",
) -> BinaryMetrics:
    """Convenience wrapper: collect predictions then compute binary metrics."""
    y_prob, y_true = collect_predictions(model, loader, device=device)
    return compute_binary_metrics(y_true, y_prob)


def evaluate_loaders(
    model: nn.Module,
    loaders: dict[str, DataLoader],
    *,
    device: str | torch.device = "cuda",
) -> dict[str, BinaryMetrics]:
    """Evaluate ``model`` on every loader in ``loaders`` (``split_name -> loader``)."""
    results: dict[str, BinaryMetrics] = {}
    for split_name, loader in loaders.items():
        metrics = evaluate_loader(model, loader, device=device)
        logger.info(
            "Eval %s: auc=%.4f log_loss=%.4f acc=%.4f n=%d",
            split_name,
            metrics.auc,
            metrics.log_loss,
            metrics.accuracy,
            metrics.num_samples,
        )
        results[split_name] = metrics
    return results


def summarize_cgrs(
    auc_after: dict[str, float],
    auc_peak: dict[str, float],
    *,
    generations: Iterable[str] | None = None,
    current_generation: str | None = None,
) -> dict[str, float]:
    """Shrink ``(auc_after, auc_peak)`` dicts into a CGRS / forgetting summary.

    Reports both:

    * ``cgrs`` — our novel ratio metric averaged over all generations in
      ``generations`` (or all of ``auc_after`` if unspecified).
    * ``avg_forgetting_all`` — Chaudhry et al. 2018 mean ``peak − after`` over
      the same set of generations. Compatible with the broader continual
      learning literature.
    * ``avg_forgetting_prev`` — same metric but excluding the generation that
      was *just* trained (passed via ``current_generation``). This is the
      stricter "did we forget the past" reading.

    ``generations`` restricts the computation to a subset; if ``None`` every
    key of ``auc_after`` is used.
    """
    if generations is not None:
        auc_after = {g: auc_after[g] for g in generations if g in auc_after}
        auc_peak = {g: auc_peak[g] for g in generations if g in auc_peak}
    return {
        "cgrs": compute_cgrs(auc_after, auc_peak),
        "avg_forgetting_all": compute_average_forgetting(auc_after, auc_peak),
        "avg_forgetting_prev": compute_average_forgetting(
            auc_after, auc_peak, exclude_current=current_generation
        ),
        "num_generations": len(auc_after),
    }


def write_metrics_json(
    path: str | Path,
    *,
    generation: str,
    split_metrics: dict[str, BinaryMetrics],
    extra: dict[str, float | str] | None = None,
) -> Path:
    """Serialize evaluator output as a JSON file.

    The on-disk layout is described in the module docstring. ``extra`` is
    merged at the top level, so callers can attach auxiliary fields
    (``checkpoint_path``, ``cgrs``, ``cde``, ...).
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "generation": generation,
        "generated_at": datetime.now().strftime(_TIMESTAMP_FORMAT),
        "splits": {name: m.as_dict() for name, m in split_metrics.items()},
    }
    if extra:
        payload.update(extra)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out_path
