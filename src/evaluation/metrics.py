"""Evaluation metrics: AUC, log-loss, accuracy, CDE, CGRS.

Definitions (per the CKD thesis):

* ``AUC``         — sklearn's ``roc_auc_score`` on fake-class probability.
* ``log_loss``    — sklearn's binary ``log_loss`` (natural log, clipped).
* ``accuracy``    — argmax-based top-1 accuracy.
* ``CDE``         — Compute-Distillation Efficiency::

      CDE = delta_AUC / compute_cost_update

  where ``delta_AUC`` is the improvement on a target generation and
  ``compute_cost_update`` is the GPU-hour (or param-update) cost of the
  training round that produced it.

* ``CGRS``        — Cross-Generational Retention Score::

      CGRS = (1 / N) * sum_i(AUC_i_after / AUC_i_peak)

  averaged over all generations ``i`` seen so far. ``1.0`` means "no
  forgetting relative to each generation's peak performance".

Called by:
    src/evaluation/evaluator.py
    scripts/06_ablation_study.py
    scripts/08_generate_figures.py
Reads / Writes: none — pure functions over numpy arrays.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import log_loss as _sk_log_loss
from sklearn.metrics import roc_auc_score

_EPS = 1.0e-7


@dataclass(frozen=True)
class BinaryMetrics:
    """Bundle of per-split metrics for one student checkpoint."""

    auc: float
    log_loss: float
    accuracy: float
    num_samples: int

    def as_dict(self) -> dict[str, float]:
        return {
            "auc": float(self.auc),
            "log_loss": float(self.log_loss),
            "accuracy": float(self.accuracy),
            "num_samples": int(self.num_samples),
        }


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> BinaryMetrics:
    """Compute AUC / log-loss / accuracy from labels + fake-class probs.

    Args:
        y_true: ``(N,)`` int64 labels in ``{0, 1}``.
        y_prob: ``(N,)`` float32 ``P(fake)`` in ``[0, 1]``.
    """
    y_true = np.asarray(y_true).astype(np.int64, copy=False)
    y_prob = np.asarray(y_prob).astype(np.float64, copy=False)
    if y_true.shape != y_prob.shape:
        raise ValueError(
            f"Shape mismatch: y_true={y_true.shape}, y_prob={y_prob.shape}"
        )
    if y_true.size == 0:
        raise ValueError("Cannot compute metrics on empty arrays")

    if np.unique(y_true).size < 2:
        # AUC is undefined for single-class evaluation; surface a neutral 0.5.
        auc = 0.5
    else:
        auc = float(roc_auc_score(y_true, y_prob))

    y_prob_clipped = np.clip(y_prob, _EPS, 1.0 - _EPS)
    loss = float(_sk_log_loss(y_true, y_prob_clipped, labels=[0, 1]))
    acc = float(((y_prob >= 0.5).astype(np.int64) == y_true).mean())
    return BinaryMetrics(
        auc=auc,
        log_loss=loss,
        accuracy=acc,
        num_samples=int(y_true.size),
    )


def compute_cde(delta_auc: float, compute_cost_update: float) -> float:
    """Compute-Distillation Efficiency = delta AUC per unit update cost.

    ``compute_cost_update`` should be a positive scalar (e.g. GPU-hours or
    parameter updates). Zero / negative inputs return ``nan`` to signal an
    ill-defined ratio rather than a misleading number.
    """
    if compute_cost_update is None or compute_cost_update <= 0:
        return float("nan")
    return float(delta_auc) / float(compute_cost_update)


def compute_cgrs(
    auc_after: dict[str, float],
    auc_peak: dict[str, float],
) -> float:
    """Cross-Generational Retention Score averaged over generations in ``auc_after``.

    Args:
        auc_after: ``gen_id -> AUC``  measured *after* the final training round.
        auc_peak:  ``gen_id -> AUC``  measured when that generation was just
            finished training (the per-generation peak).

    Returns:
        The mean of ``auc_after[g] / auc_peak[g]`` across every generation
        present in ``auc_after``. A value of ``1.0`` means no forgetting.
    """
    if not auc_after:
        raise ValueError("auc_after must not be empty")

    ratios: list[float] = []
    for gen_id, auc_now in auc_after.items():
        peak = auc_peak.get(gen_id)
        if peak is None or peak <= 0:
            raise ValueError(
                f"Missing or non-positive peak AUC for generation '{gen_id}'"
            )
        ratios.append(float(auc_now) / float(peak))
    return float(np.mean(ratios))
