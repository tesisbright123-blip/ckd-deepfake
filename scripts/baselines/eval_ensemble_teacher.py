"""Baseline B1 — direct evaluation of the teacher ensemble on test sets.

CLI entrypoint. B1 establishes the "theoretical upper bound": how well the
full teacher ensemble (EfficientNet-B4 + RECCE + CLIP ViT-L/14, ~350M params)
performs on each generation's test set, *before* any distillation. The gap
between B1 and the distilled student quantifies the compression cost.

Two modes:

1. **Fast / default (no GPU).** Reuses the ensemble soft-labels already
   written to ``soft_labels/{gen}/test/ensemble.npy`` during Phase 3
   (``scripts/03_generate_soft_labels.py``). These are the weighted-ensemble
   fake-probabilities on each generation's OWN test split (the diagonal:
   gen1-ensemble × gen1-test, etc.). AUC / log-loss / accuracy are computed
   directly against the test-split labels. Zero GPU, ~seconds.

   This diagonal is the meaningful upper bound: each generation's ensemble is
   calibrated (excess-AUC weighted) for that generation, so the diagonal shows
   the best the ensemble can do per generation — exactly the ceiling the
   student is distilled toward.

2. **Per-teacher breakdown (optional).** If individual teacher npy files
   (``soft_labels/{gen}/test/{teacher}.npy``) exist, per-teacher AUC is also
   reported, which is useful context for BAB IV (shows CLIP dominance on Gen3,
   RECCE weakness on Gen1/Gen2, etc.).

Cross-generation B1 (e.g. gen3-calibrated ensemble on gen1-test) is NOT
computed here because it would require re-running teacher inference on
cross-gen test sets (heavy: loads all three teachers incl. CLIP ~1.7 GB).
The diagonal is sufficient for the upper-bound reference in the thesis; a
note is emitted if cross-gen numbers are ever needed.

Usage:
    python scripts/baselines/eval_ensemble_teacher.py --config configs/local.yaml
    python scripts/baselines/eval_ensemble_teacher.py \
        --config configs/macbook.yaml --generations gen1,gen2,gen3

Reads:
    {drive}/soft_labels/{gen}/test/ensemble.npy      (weighted ensemble fake-prob)
    {drive}/soft_labels/{gen}/test/{teacher}.npy     (optional, per-teacher)
    {drive}/datasets/splits/{gen}_test.csv           (labels)
    {drive}/soft_labels/{gen}/ensemble_weights.json  (optional, for weights in output)
Writes:
    {drive}/results/raw/baselines/B1_ensemble_metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.evaluation.metrics import compute_binary_metrics  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

_TIMESTAMP_FMT = "%Y-%m-%dT%H:%M:%S"
_TEACHER_NAMES = ("efficientnet_b4", "recce", "clip_vit_l14")


def _load_labels(test_csv: Path) -> np.ndarray:
    df = pd.read_csv(test_csv, usecols=["label"], low_memory=False)
    return df["label"].to_numpy().astype(np.int64)


def _eval_npy(
    npy_path: Path, labels: np.ndarray, *, logger
) -> dict[str, Any] | None:
    """Load a fake-prob .npy and compute metrics against ``labels``."""
    if not npy_path.is_file():
        return None
    probs = np.load(npy_path).astype(np.float64)
    if probs.shape[0] != labels.shape[0]:
        logger.error(
            "Length mismatch: %s has %d preds but labels have %d — skipping",
            npy_path.name,
            probs.shape[0],
            labels.shape[0],
        )
        return None
    metrics = compute_binary_metrics(labels, probs)
    return metrics.as_dict()


def _eval_generation(
    gen: str, *, drive_root: Path, logger
) -> dict[str, Any]:
    """Evaluate the ensemble (and optional per-teacher) on one generation's test."""
    sl_dir = drive_root / "soft_labels" / gen / "test"
    test_csv = drive_root / "datasets" / "splits" / f"{gen}_test.csv"

    if not test_csv.is_file():
        logger.error("Missing test CSV for %s: %s", gen, test_csv)
        return {"generation": gen, "status": "missing_csv"}

    labels = _load_labels(test_csv)
    result: dict[str, Any] = {
        "generation": gen,
        "num_samples": int(labels.shape[0]),
        "num_real": int((labels == 0).sum()),
        "num_fake": int((labels == 1).sum()),
    }

    # Ensemble (diagonal)
    ens = _eval_npy(sl_dir / "ensemble.npy", labels, logger=logger)
    if ens is None:
        logger.error(
            "Missing ensemble.npy for %s (%s) — run scripts/03 first",
            gen,
            sl_dir / "ensemble.npy",
        )
        result["status"] = "missing_ensemble"
        return result
    result["ensemble"] = ens
    logger.info(
        "B1 %s ensemble: AUC=%.4f log_loss=%.4f acc=%.4f (n=%d)",
        gen,
        ens["auc"],
        ens["log_loss"],
        ens["accuracy"],
        ens["num_samples"],
    )

    # Per-teacher (optional)
    per_teacher: dict[str, Any] = {}
    for tname in _TEACHER_NAMES:
        t = _eval_npy(sl_dir / f"{tname}.npy", labels, logger=logger)
        if t is not None:
            per_teacher[tname] = t
            logger.info("  teacher %s: AUC=%.4f", tname, t["auc"])
    if per_teacher:
        result["per_teacher"] = per_teacher

    # Ensemble weights (optional, for provenance)
    weights_path = drive_root / "soft_labels" / gen / "ensemble_weights.json"
    if weights_path.is_file():
        try:
            result["ensemble_weights"] = json.loads(weights_path.read_text())
        except (OSError, json.JSONDecodeError):
            pass

    result["status"] = "ok"
    return result


def _run(args: argparse.Namespace) -> int:
    logger = get_logger("baselines.B1", log_file="runs/baseline_B1_ensemble.log")
    cfg = load_config(args.config)
    drive_root = Path(cfg["paths"]["drive_root"])

    gens = (
        ["gen1", "gen2", "gen3"]
        if args.generations == "all"
        else [g.strip() for g in args.generations.split(",") if g.strip()]
    )
    logger.info("B1 ensemble-teacher eval on: %s (drive_root=%s)", gens, drive_root)

    per_gen = [_eval_generation(g, drive_root=drive_root, logger=logger) for g in gens]

    results_dir = (
        Path(args.results_dir)
        if args.results_dir
        else drive_root / "results" / "raw" / "baselines"
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "B1_ensemble_metrics.json"
    payload = {
        "baseline": "B1",
        "label": "Ensemble teacher direct evaluation (upper bound)",
        "generated_at": datetime.now(timezone.utc).strftime(_TIMESTAMP_FMT),
        "note": (
            "Diagonal evaluation: each generation's excess-AUC-weighted "
            "ensemble on its OWN test split. Cross-generation B1 would require "
            "re-running teacher inference and is out of scope."
        ),
        "generations": per_gen,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote B1 metrics: %s", out_path)

    ok = all(g.get("status") == "ok" for g in per_gen)
    if not ok:
        logger.warning(
            "Some generations did not evaluate cleanly — inspect %s", out_path
        )
    return 0 if ok else 1


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Baseline B1 — ensemble teacher direct evaluation (upper bound)"
    )
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument(
        "--generations",
        default="all",
        help="Comma-separated list (e.g. 'gen1,gen2,gen3') or 'all'",
    )
    p.add_argument("--results-dir", default=None)
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(_run(_parse_args()))
