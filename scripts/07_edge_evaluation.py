"""Step 7: TFLite export, latency benchmark, and on-device accuracy check.

Entrypoint script — invoked via CLI, not imported by other code.

Usage:
    python scripts/07_edge_evaluation.py --generation gen1
    python scripts/07_edge_evaluation.py --generation gen3 --method replay
    python scripts/07_edge_evaluation.py --checkpoint path/to/best.pth \
        --modes fp32,fp16,int8 --num-runs 200 --num-threads 4

Calls:
    src/evaluation/edge_eval.py (export_onnx, convert_to_tflite,
        benchmark_tflite, evaluate_tflite, run_edge_benchmark)
    src/models/students/mobilenetv3.py (build_student)
    src/data/dataloader.py (build_dataloader)
    src/utils/config.py (load_config)
    src/utils/checkpoint.py (load_checkpoint)
Reads:
    YAML config (--config).
    Student checkpoint ``{drive}/checkpoints/students/{generation}[_{method}]/best.pth``
        (override via --checkpoint).
    Test-split CSV + face JPEGs for accuracy evaluation.
    Train-split CSV for INT8 calibration (representative loader).
Writes:
    ONNX at {drive}/edge/{generation}[_{method}]/model.onnx
    TFLite at {drive}/edge/{generation}[_{method}]/tflite/{fp32,fp16,int8}.tflite
    Benchmark summary at
        {drive}/results/raw/{generation}[_{method}]_edge_metrics.json
        (ISO 8601 ``generated_at``; per-mode size/latency/AUC rows)
    Log file at runs/edge_evaluation_{generation}[_{method}].log
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.dataloader import build_dataloader  # noqa: E402
from src.evaluation.edge_eval import (  # noqa: E402
    SUPPORTED_MODES,
    run_edge_benchmark,
)
from src.models.students.mobilenetv3 import build_student  # noqa: E402
from src.utils.checkpoint import load_checkpoint  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

_TIMESTAMP_FMT = "%Y-%m-%dT%H:%M:%S"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_checkpoint_path(
    drive_root: Path, generation: str, method: str | None, override: str | None
) -> Path:
    if override:
        return Path(override)
    subdir = f"{generation}_{method}" if method else generation
    return drive_root / "checkpoints" / "students" / subdir / "best.pth"


def _resolve_run_tag(generation: str, method: str | None) -> str:
    return f"{generation}_{method}" if method else generation


def _parse_modes(text: str) -> tuple[str, ...]:
    candidates = tuple(m.strip().lower() for m in text.split(",") if m.strip())
    unknown = [m for m in candidates if m not in SUPPORTED_MODES]
    if unknown:
        raise SystemExit(
            f"Unknown TFLite modes: {unknown}. Expected subset of {SUPPORTED_MODES}."
        )
    return candidates


def _build_eval_loader(
    *,
    drive_root: Path,
    generation: str,
    batch_size: int,
    num_workers: int,
    image_size: int,
    aug_cfg: dict[str, Any],
):
    test_csv = drive_root / "datasets" / "splits" / f"{generation}_test.csv"
    if not test_csv.is_file():
        raise FileNotFoundError(f"Missing test split CSV: {test_csv}")
    return build_dataloader(
        csv_path=test_csv,
        mode="test",
        batch_size=batch_size,
        soft_label_path=None,
        image_size=image_size,
        aug_cfg=aug_cfg,
        num_workers=num_workers,
    )


def _build_calibration_loader(
    *,
    drive_root: Path,
    generation: str,
    batch_size: int,
    num_workers: int,
    image_size: int,
    aug_cfg: dict[str, Any],
    needed: bool,
):
    if not needed:
        return None
    # Use the val split for calibration — avoids leaking train augmentations
    # into the calibration statistics while staying in-distribution.
    calib_csv = drive_root / "datasets" / "splits" / f"{generation}_val.csv"
    if not calib_csv.is_file():
        raise FileNotFoundError(f"Missing calibration CSV: {calib_csv}")
    return build_dataloader(
        csv_path=calib_csv,
        mode="val",
        batch_size=batch_size,
        soft_label_path=None,
        image_size=image_size,
        aug_cfg=aug_cfg,
        num_workers=num_workers,
    )


def _write_summary(
    path: Path,
    *,
    generation: str,
    method: str | None,
    checkpoint_path: Path,
    results: list[Any],
    extras: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "generation": generation,
        "method": method,
        "checkpoint_path": str(checkpoint_path),
        "generated_at": datetime.now(timezone.utc).strftime(_TIMESTAMP_FMT),
        "modes": [r.as_dict() for r in results],
    }
    payload.update(extras)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _run(args: argparse.Namespace) -> int:
    run_tag = _resolve_run_tag(args.generation, args.method)
    logger = get_logger(
        f"edge_evaluation.{run_tag}",
        log_file=f"runs/edge_evaluation_{run_tag}.log",
    )
    _seed_everything(args.seed)

    cfg = load_config(args.config)
    drive_root = Path(cfg["paths"]["drive_root"])
    student_cfg = cfg["student"]
    aug_cfg = cfg["data"]["augmentation"]
    edge_cfg = cfg.get("edge", {})
    image_size = int(student_cfg.get("input_size", 224))
    batch_size = int(args.batch_size or cfg["training"]["initial_distillation"].get("batch_size", 64))

    modes = _parse_modes(args.modes) if args.modes else tuple(
        m.lower() for m in edge_cfg.get("tflite_quantization", ["fp32", "fp16", "int8"])
    )
    if not modes:
        logger.error("No TFLite modes selected — aborting")
        return 1
    logger.info("Requested TFLite modes: %s", list(modes))

    checkpoint_path = _resolve_checkpoint_path(
        drive_root, args.generation, args.method, args.checkpoint
    )
    if not checkpoint_path.is_file():
        logger.error(
            "Student checkpoint not found: %s (run scripts/04 or scripts/05 first)",
            checkpoint_path,
        )
        return 1

    # --- Student model + weights -------------------------------------------
    model = build_student(
        hidden_dim=int(student_cfg.get("head", {}).get("hidden_dim", 256)),
        dropout=float(student_cfg.get("head", {}).get("dropout", 0.2)),
        num_classes=int(student_cfg.get("num_classes", 2)),
        pretrained=False,
    )
    meta = load_checkpoint(checkpoint_path, model=model, strict=False, map_location="cpu")
    logger.info(
        "Loaded student from %s (epoch=%s, best_val_auc=%s)",
        checkpoint_path,
        meta.get("epoch"),
        meta.get("best_val_auc"),
    )

    # --- Loaders -----------------------------------------------------------
    eval_loader = _build_eval_loader(
        drive_root=drive_root,
        generation=args.generation,
        batch_size=batch_size,
        num_workers=args.num_workers,
        image_size=image_size,
        aug_cfg=aug_cfg,
    )
    representative_loader = _build_calibration_loader(
        drive_root=drive_root,
        generation=args.generation,
        batch_size=batch_size,
        num_workers=args.num_workers,
        image_size=image_size,
        aug_cfg=aug_cfg,
        needed=("int8" in modes),
    )

    # --- Edge benchmark ----------------------------------------------------
    output_dir = (
        Path(args.output_dir) if args.output_dir else drive_root / "edge" / run_tag
    )
    num_runs = int(args.num_runs or edge_cfg.get("benchmark_num_runs", 100))
    logger.info(
        "Running edge benchmark (runs=%d warmup=%d threads=%d modes=%s) -> %s",
        num_runs,
        args.num_warmup,
        args.num_threads,
        list(modes),
        output_dir,
    )
    results = run_edge_benchmark(
        model,
        eval_loader=eval_loader,
        output_dir=output_dir,
        representative_loader=representative_loader,
        modes=modes,
        input_size=image_size,
        num_runs=num_runs,
        num_warmup=args.num_warmup,
        num_threads=args.num_threads,
        calibration_batches=args.calibration_batches,
    )
    if not results:
        logger.error("No TFLite results produced — check onnx2tf logs")
        return 1

    # --- Summary JSON ------------------------------------------------------
    results_dir = (
        Path(args.results_dir) if args.results_dir else drive_root / "results" / "raw"
    )
    summary_path = results_dir / f"{run_tag}_edge_metrics.json"
    _write_summary(
        summary_path,
        generation=args.generation,
        method=args.method,
        checkpoint_path=checkpoint_path,
        results=results,
        extras={
            "best_val_auc": meta.get("best_val_auc"),
            "num_runs": num_runs,
            "num_warmup": int(args.num_warmup),
            "num_threads": int(args.num_threads),
            "calibration_batches": int(args.calibration_batches),
            "image_size": image_size,
            "output_dir": str(output_dir),
            "seed": int(args.seed),
        },
    )
    logger.info("Wrote edge metrics: %s", summary_path)
    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export student to TFLite and benchmark edge deployment"
    )
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument(
        "--generation",
        required=True,
        choices=["gen1", "gen2", "gen3"],
        help="Which student generation to export",
    )
    p.add_argument(
        "--method",
        default=None,
        choices=["ewc", "lwf", "replay"],
        help=(
            "For gen2/gen3 continual checkpoints, the anti-forgetting method "
            "(affects the default checkpoint path). Omit for initial gen1."
        ),
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Override the default student checkpoint path",
    )
    p.add_argument(
        "--modes",
        default=None,
        help=(
            "Comma-separated subset of {fp32,fp16,int8}. "
            "Defaults to ``edge.tflite_quantization`` from the config."
        ),
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--num-runs", type=int, default=None)
    p.add_argument("--num-warmup", type=int, default=10)
    p.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="CPU threads for the TFLite interpreter (typical edge: 1-4)",
    )
    p.add_argument(
        "--calibration-batches",
        type=int,
        default=16,
        help="Number of val-set batches used to calibrate INT8 quantization",
    )
    p.add_argument("--output-dir", default=None)
    p.add_argument("--results-dir", default=None)
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(_run(_parse_args()))
