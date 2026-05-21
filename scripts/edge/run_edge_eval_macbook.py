"""End-to-end edge evaluation orchestrator for MacBook M2 Max.

Runs the following over the 9 student checkpoints (3 gens x 3 seeds):

  1. Sanity-check conversion on one pilot checkpoint (PyTorch <-> TFLite
     fp32 <-> TFLite int8 <-> CoreML fp32 <-> CoreML int8 numerical match).
  2. Mass conversion for the remaining checkpoints.
  3. AUC evaluation on three test splits per checkpoint (gen3_test full
     + gen1_test 5K subset + gen2_test 5K subset).
  4. Latency benchmarks on one representative checkpoint x 4 formats.
  5. Aggregated JSON output ready for thesis tables.

CoreML is skipped automatically if not on macOS (logs a warning).

Usage:
    python scripts/edge/run_edge_eval_macbook.py \
        --config configs/macbook.yaml \
        --output-dir ~/ckd-edge/edge_results \
        --subset-rows 5000

Reads:
    {drive_root}/checkpoints/students/gen{1,2,3}_*_seed{0,1,2}/best.pth
    {drive_root}/datasets/splits/gen{1,2,3}_test.csv
    configs/macbook.yaml (drive_root override)

Writes:
    {output-dir}/conversions/<run_tag>/<format>.{tflite,mlpackage}
    {output-dir}/results/edge_eval_full.json
    {output-dir}/results/edge_eval_summary.md
    {output-dir}/sanity_report.json
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.dataloader import build_dataloader  # noqa: E402
from src.evaluation.edge_eval import (  # noqa: E402
    SUPPORTED_MODES,
    convert_to_tflite,
    evaluate_tflite,
    export_onnx,
    file_size_mb,
    benchmark_tflite,
)
from src.evaluation.edge_eval_sanity import (  # noqa: E402
    SanityReport,
    verify_coreml,
    verify_tflite,
)
from src.models.students.mobilenetv3 import build_student  # noqa: E402
from src.utils.checkpoint import load_checkpoint  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

ON_MACOS = platform.system() == "Darwin"

# Checkpoint definitions: (run_tag, ckpt_relative_path_under_students)
CHECKPOINT_TAGS: list[tuple[str, str]] = [
    ("gen1_seed0", "gen1_seed0/best.pth"),
    ("gen1_seed1", "gen1_seed1/best.pth"),
    ("gen1_seed2", "gen1_seed2/best.pth"),
    ("gen2_replay+ewc_seed0", "gen2_replay+ewc_seed0/best.pth"),
    ("gen2_replay+ewc_seed1", "gen2_replay+ewc_seed1/best.pth"),
    ("gen2_replay+ewc_seed2", "gen2_replay+ewc_seed2/best.pth"),
    ("gen3_replay+ewc_seed0", "gen3_replay+ewc_seed0/best.pth"),
    ("gen3_replay+ewc_seed1", "gen3_replay+ewc_seed1/best.pth"),
    ("gen3_replay+ewc_seed2", "gen3_replay+ewc_seed2/best.pth"),
]


@dataclass(frozen=True)
class EvalRow:
    run_tag: str
    runtime: str       # "pytorch" | "tflite" | "coreml"
    mode: str          # "fp32" | "int8" (or "n/a" for pytorch)
    test_split: str    # "gen1_test" | "gen2_test" | "gen3_test"
    num_samples: int
    auc: float
    log_loss: float
    accuracy: float
    artifact_path: str
    size_mb: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_tag": self.run_tag,
            "runtime": self.runtime,
            "mode": self.mode,
            "test_split": self.test_split,
            "num_samples": int(self.num_samples),
            "auc": float(self.auc),
            "log_loss": float(self.log_loss),
            "accuracy": float(self.accuracy),
            "artifact_path": self.artifact_path,
            "size_mb": float(self.size_mb),
        }


@dataclass(frozen=True)
class LatencyRow:
    runtime: str
    mode: str
    compute_unit: str
    latency_ms_mean: float
    latency_ms_p50: float
    latency_ms_p95: float
    latency_ms_p99: float
    num_runs: int
    num_warmup: int
    size_mb: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "runtime": self.runtime,
            "mode": self.mode,
            "compute_unit": self.compute_unit,
            "latency_ms_mean": float(self.latency_ms_mean),
            "latency_ms_p50": float(self.latency_ms_p50),
            "latency_ms_p95": float(self.latency_ms_p95),
            "latency_ms_p99": float(self.latency_ms_p99),
            "num_runs": int(self.num_runs),
            "num_warmup": int(self.num_warmup),
            "size_mb": float(self.size_mb),
        }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _stratified_subset_csv(
    src_csv: Path, dst_csv: Path, *, n_rows: int, seed: int = 0
) -> Path:
    """Write a stratified-by-label subsample of ``src_csv`` to ``dst_csv``.

    The full test CSVs are imbalanced (~22% real, ~78% fake); stratifying
    keeps the AUC estimator unbiased on the subset.
    """
    df = pd.read_csv(src_csv, low_memory=False)
    if len(df) <= n_rows:
        df_out = df
    else:
        # Stratify by label, preserving the original class ratios.
        per_label = []
        rng = np.random.default_rng(seed)
        for label, group in df.groupby("label", group_keys=False):
            frac = len(group) / len(df)
            target = max(1, int(round(n_rows * frac)))
            idx = rng.choice(group.index.values, size=min(target, len(group)), replace=False)
            per_label.append(group.loc[idx])
        df_out = pd.concat(per_label, axis=0).sample(frac=1, random_state=seed).reset_index(drop=True)

    dst_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(dst_csv, index=False)
    return dst_csv


def _make_loader(
    csv_path: Path,
    *,
    batch_size: int,
    num_workers: int,
    image_size: int,
    aug_cfg: dict[str, Any],
):
    return build_dataloader(
        csv_path=csv_path,
        mode="test",
        batch_size=batch_size,
        soft_label_path=None,
        image_size=image_size,
        aug_cfg=aug_cfg,
        num_workers=num_workers,
        pin_memory=False,  # CPU-only eval; pin_memory wastes RAM
    )


def _make_calib_loader(
    *,
    csv_dir: Path,
    output_dir: Path,
    batch_size: int,
    num_workers: int,
    image_size: int,
    aug_cfg: dict[str, Any],
    seed: int,
    samples_per_gen: int = 200,
):
    """Build a calibration CSV mixed across all 3 gens, then load it.

    Mixing ensures the int8 activation ranges generalize across all
    deepfake categories the deployed model is expected to see.
    """
    rng = np.random.default_rng(seed)
    chunks: list[pd.DataFrame] = []
    for gen in ("gen1", "gen2", "gen3"):
        src = csv_dir / f"{gen}_test.csv"
        if not src.is_file():
            logger.warning("Calibration source missing: %s — skipping %s", src, gen)
            continue
        df = pd.read_csv(src, low_memory=False)
        # Stratified per label
        per_label = []
        for label, group in df.groupby("label", group_keys=False):
            frac = len(group) / len(df)
            target = max(1, int(round(samples_per_gen * frac)))
            idx = rng.choice(group.index.values, size=min(target, len(group)), replace=False)
            per_label.append(group.loc[idx])
        chunks.append(pd.concat(per_label, axis=0))

    if not chunks:
        raise RuntimeError("No calibration data — all gen*_test.csv missing")
    calib_df = pd.concat(chunks, axis=0).sample(frac=1, random_state=seed).reset_index(drop=True)
    calib_csv = output_dir / "calibration_mixed.csv"
    calib_csv.parent.mkdir(parents=True, exist_ok=True)
    calib_df.to_csv(calib_csv, index=False)
    logger.info(
        "Calibration CSV: %d rows (mixed gen1/gen2/gen3) -> %s", len(calib_df), calib_csv
    )
    return _make_loader(
        calib_csv,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        aug_cfg=aug_cfg,
    )


def _load_student(
    ckpt_path: Path, *, student_cfg: dict[str, Any]
) -> tuple[torch.nn.Module, dict[str, Any]]:
    model = build_student(
        hidden_dim=int(student_cfg.get("head", {}).get("hidden_dim", 256)),
        dropout=float(student_cfg.get("head", {}).get("dropout", 0.2)),
        num_classes=int(student_cfg.get("num_classes", 2)),
        pretrained=False,
    )
    meta = load_checkpoint(ckpt_path, model=model, strict=False, map_location="cpu")
    return model.eval(), meta


def _evaluate_pytorch_baseline(
    model: torch.nn.Module, loader, *, fake_class_index: int = 1
) -> dict[str, float]:
    """PyTorch CPU baseline AUC for reference."""
    from src.evaluation.metrics import compute_binary_metrics

    model = model.eval().cpu()
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            images, hard_labels, *_ = batch
            logits = model(images.float().cpu())
            probs = torch.softmax(logits, dim=-1)[:, fake_class_index].cpu().numpy()
            all_probs.append(probs.astype(np.float32))
            labels_np = (
                hard_labels.cpu().numpy()
                if isinstance(hard_labels, torch.Tensor)
                else np.asarray(hard_labels)
            )
            all_labels.append(labels_np.astype(np.int64))
    if not all_probs:
        raise RuntimeError("PyTorch eval loader yielded zero batches")
    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    metrics = compute_binary_metrics(y_true, y_prob)
    return {
        "auc": float(metrics.auc),
        "log_loss": float(metrics.log_loss),
        "accuracy": float(metrics.accuracy),
        "num_samples": int(metrics.num_samples),
    }


def _convert_one_checkpoint(
    model: torch.nn.Module,
    *,
    run_tag: str,
    conversion_root: Path,
    representative_loader,
    image_size: int,
    calibration_batches: int,
    coreml_enabled: bool,
) -> dict[str, Path]:
    """Convert one checkpoint to TFLite fp32/int8 (+ CoreML if on macOS).

    Returns:
        ``{"tflite_fp32": Path, "tflite_int8": Path,
           "coreml_fp32": Path, "coreml_int8": Path}``
        (Keys for failed conversions are omitted.)
    """
    out_dir = conversion_root / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Path] = {}

    # --- TFLite ---
    onnx_path = export_onnx(model, out_dir / "model.onnx", input_size=image_size, opset=17)
    tflite_dir = out_dir / "tflite"
    try:
        tflite_paths = convert_to_tflite(
            onnx_path,
            tflite_dir,
            modes=("fp32", "int8"),
            representative_loader=representative_loader,
            calibration_batches=calibration_batches,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("TFLite conversion failed for %s: %s", run_tag, exc)
        tflite_paths = {}
    for mode, path in tflite_paths.items():
        artifacts[f"tflite_{mode}"] = path

    # --- CoreML ---
    if coreml_enabled:
        from src.evaluation.edge_eval_coreml import export_coreml

        coreml_dir = out_dir / "coreml"
        coreml_dir.mkdir(parents=True, exist_ok=True)
        for mode in ("fp32", "int8"):
            try:
                p = export_coreml(
                    model,
                    coreml_dir / f"{mode}.mlpackage",
                    input_size=image_size,
                    mode=mode,
                    compute_precision="FLOAT32" if mode == "fp32" else "FLOAT16",
                )
                artifacts[f"coreml_{mode}"] = p
            except Exception as exc:  # noqa: BLE001
                logger.error("CoreML conversion failed for %s/%s: %s", run_tag, mode, exc)
    return artifacts


def _run_sanity_check(
    model: torch.nn.Module,
    artifacts: dict[str, Path],
    *,
    image_size: int,
) -> SanityReport:
    """Verify numerical match between PyTorch and all converted artifacts."""
    tflite_paths = {
        m.split("_", 1)[1]: p
        for m, p in artifacts.items()
        if m.startswith("tflite_")
    }
    coreml_paths = {
        m.split("_", 1)[1]: p
        for m, p in artifacts.items()
        if m.startswith("coreml_")
    }
    report = SanityReport()
    if tflite_paths:
        tf_report = verify_tflite(model, tflite_paths, input_size=image_size)
        report.rows.extend(tf_report.rows)
    if coreml_paths and ON_MACOS:
        cm_report = verify_coreml(model, coreml_paths, input_size=image_size)
        report.rows.extend(cm_report.rows)
    return report


def _eval_artifact_on_split(
    artifact_path: Path,
    runtime: str,
    mode: str,
    loader,
    *,
    compute_unit: str = "all",
) -> dict[str, float]:
    if runtime == "tflite":
        metrics = evaluate_tflite(artifact_path, loader, num_threads=4)
    elif runtime == "coreml":
        from src.evaluation.edge_eval_coreml import evaluate_coreml

        metrics = evaluate_coreml(artifact_path, loader, compute_unit=compute_unit)
    else:
        raise ValueError(f"Unknown runtime: {runtime!r}")
    return {
        "auc": float(metrics.auc),
        "log_loss": float(metrics.log_loss),
        "accuracy": float(metrics.accuracy),
        "num_samples": int(metrics.num_samples),
    }


def _benchmark_latency(
    artifacts: dict[str, Path],
    *,
    image_size: int,
    num_runs: int,
    num_warmup: int,
    coreml_enabled: bool,
) -> list[LatencyRow]:
    """Benchmark latency on a single representative checkpoint."""
    rows: list[LatencyRow] = []
    for key, path in artifacts.items():
        if key.startswith("tflite_"):
            mode = key.split("_", 1)[1]
            lat = benchmark_tflite(
                path,
                input_size=image_size,
                num_runs=num_runs,
                num_warmup=num_warmup,
                num_threads=4,
            )
            rows.append(
                LatencyRow(
                    runtime="tflite",
                    mode=mode,
                    compute_unit="cpu_xnnpack",
                    latency_ms_mean=lat["latency_ms_mean"],
                    latency_ms_p50=lat["latency_ms_p50"],
                    latency_ms_p95=lat["latency_ms_p95"],
                    latency_ms_p99=lat.get("latency_ms_p99", lat["latency_ms_p95"]),
                    num_runs=lat["num_runs"],
                    num_warmup=lat["num_warmup"],
                    size_mb=file_size_mb(path),
                )
            )
        elif key.startswith("coreml_") and coreml_enabled and ON_MACOS:
            from src.evaluation.edge_eval_coreml import (
                _mlpackage_size_mb,
                benchmark_coreml,
            )

            mode = key.split("_", 1)[1]
            for cu_label in ("all", "cpu_only"):
                lat = benchmark_coreml(
                    path,
                    input_size=image_size,
                    num_runs=num_runs,
                    num_warmup=num_warmup,
                    compute_unit=cu_label,
                )
                rows.append(
                    LatencyRow(
                        runtime="coreml",
                        mode=mode,
                        compute_unit=cu_label,
                        latency_ms_mean=lat["latency_ms_mean"],
                        latency_ms_p50=lat["latency_ms_p50"],
                        latency_ms_p95=lat["latency_ms_p95"],
                        latency_ms_p99=lat["latency_ms_p99"],
                        num_runs=lat["num_runs"],
                        num_warmup=lat["num_warmup"],
                        size_mb=_mlpackage_size_mb(path),
                    )
                )
    return rows


def _write_summary_markdown(
    *,
    output_path: Path,
    eval_rows: list[EvalRow],
    latency_rows: list[LatencyRow],
    sanity_report: SanityReport,
    started_at: str,
    finished_at: str,
) -> None:
    """Render a Markdown summary that can be pasted into the thesis."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# CKD Edge Evaluation — MacBook M2 Max Summary")
    lines.append("")
    lines.append(f"- Started:  {started_at}")
    lines.append(f"- Finished: {finished_at}")
    lines.append(f"- Platform: {platform.platform()}")
    lines.append("")
    lines.append("## 1. Sanity check (numerical match vs PyTorch baseline)")
    lines.append("")
    lines.append("| Runtime | Mode | max_abs_diff | mean_abs_diff | cosine_sim | threshold | passed |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in sanity_report.rows:
        lines.append(
            f"| {row.runtime} | {row.mode} | "
            f"{row.max_abs_diff:.2e} | {row.mean_abs_diff:.2e} | "
            f"{row.cosine_similarity:.4f} | {row.threshold:.2e} | "
            f"{'OK' if row.passed else 'FAIL'} |"
        )
    lines.append("")

    lines.append("## 2. Latency (single-image, batch=1)")
    lines.append("")
    lines.append("| Runtime | Mode | Compute Unit | mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) | size (MB) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in latency_rows:
        lines.append(
            f"| {r.runtime} | {r.mode} | {r.compute_unit} | "
            f"{r.latency_ms_mean:.2f} | {r.latency_ms_p50:.2f} | "
            f"{r.latency_ms_p95:.2f} | {r.latency_ms_p99:.2f} | {r.size_mb:.2f} |"
        )
    lines.append("")

    lines.append("## 3. AUC per (checkpoint, runtime, mode, test_split)")
    lines.append("")
    lines.append("Mean +/- std aggregated across the 3 seeds of each generation:")
    lines.append("")

    # Aggregate: group by (gen_prefix, runtime, mode, test_split) then mean+std AUC.
    df = pd.DataFrame([r.as_dict() for r in eval_rows])
    if not df.empty:
        # Extract the "gen<X>" prefix from run_tag.
        df["ckpt_gen"] = df["run_tag"].str.extract(r"^(gen\d)")[0]
        agg = (
            df.groupby(["ckpt_gen", "runtime", "mode", "test_split"])["auc"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        lines.append("| Ckpt Gen | Runtime | Mode | Test Split | AUC mean | AUC std | n |")
        lines.append("|---|---|---|---|---|---|---|")
        for _, row in agg.iterrows():
            std = row["std"] if not pd.isna(row["std"]) else 0.0
            lines.append(
                f"| {row['ckpt_gen']} | {row['runtime']} | {row['mode']} | "
                f"{row['test_split']} | {row['mean']:.4f} | {std:.4f} | "
                f"{int(row['count'])} |"
            )
        lines.append("")

        # Full per-checkpoint table for appendix
        lines.append("## 4. Full per-checkpoint AUC table (appendix)")
        lines.append("")
        lines.append("| Run Tag | Runtime | Mode | Test Split | AUC | Log Loss | Accuracy | Samples |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in eval_rows:
            lines.append(
                f"| {r.run_tag} | {r.runtime} | {r.mode} | {r.test_split} | "
                f"{r.auc:.4f} | {r.log_loss:.4f} | {r.accuracy:.4f} | {r.num_samples} |"
            )
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-end edge evaluation orchestrator (MacBook)"
    )
    p.add_argument(
        "--config",
        default="configs/macbook.yaml",
        help="Path to the YAML config with drive_root pointing to the MacBook mirror",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "ckd-edge" / "edge_results",
        help="Where to write converted artifacts + results (default: ~/ckd-edge/edge_results)",
    )
    p.add_argument(
        "--subset-rows",
        type=int,
        default=5000,
        help="Subsample size for gen1/gen2 cross-gen evaluation (default: 5000)",
    )
    p.add_argument(
        "--gen3-full",
        action="store_true",
        default=True,
        help="Use the full gen3_test set for AUC (default: enabled)",
    )
    p.add_argument(
        "--no-coreml",
        action="store_true",
        help="Skip CoreML conversion + benchmarks (default: enabled on macOS)",
    )
    p.add_argument(
        "--num-latency-runs",
        type=int,
        default=200,
        help="Number of latency benchmark iterations (default: 200)",
    )
    p.add_argument("--num-warmup", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--calibration-batches", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--pilot-only",
        action="store_true",
        help="Convert and sanity-check only the first checkpoint (debugging)",
    )
    return p.parse_args()


def _run(args: argparse.Namespace) -> int:
    started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    overall_start = time.time()
    _seed_everything(args.seed)

    coreml_enabled = ON_MACOS and not args.no_coreml
    if not ON_MACOS:
        logger.warning(
            "Not running on macOS (%s) — CoreML benchmarks will be skipped.",
            platform.system(),
        )

    cfg = load_config(args.config)
    drive_root = Path(cfg["paths"]["drive_root"]).expanduser().resolve()
    student_cfg = cfg["student"]
    aug_cfg = cfg["data"]["augmentation"]
    image_size = int(student_cfg.get("input_size", 224))

    splits_dir = drive_root / "datasets" / "splits"
    ckpt_root = drive_root / "checkpoints" / "students"

    output_dir = args.output_dir.expanduser().resolve()
    conversion_root = output_dir / "conversions"
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- Test loaders ----------------------------------------------------
    # gen3 full + gen1/gen2 stratified subset.
    gen3_test_csv = splits_dir / "gen3_test.csv"
    if not gen3_test_csv.is_file():
        logger.error("Missing %s — run setup_macbook_mirror.py first", gen3_test_csv)
        return 1

    test_csvs: dict[str, Path] = {}
    test_csvs["gen3_test"] = gen3_test_csv  # full
    for gen in ("gen1", "gen2"):
        src = splits_dir / f"{gen}_test.csv"
        if not src.is_file():
            logger.warning("Missing %s — cross-gen %s eval will be skipped", src, gen)
            continue
        subset_csv = output_dir / "subsets" / f"{gen}_test_{args.subset_rows}.csv"
        _stratified_subset_csv(src, subset_csv, n_rows=args.subset_rows, seed=args.seed)
        test_csvs[f"{gen}_test"] = subset_csv

    test_loaders = {
        split: _make_loader(
            csv,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=image_size,
            aug_cfg=aug_cfg,
        )
        for split, csv in test_csvs.items()
    }

    # --- Calibration loader (mixed gen1+gen2+gen3) ------------------------
    calib_loader = _make_calib_loader(
        csv_dir=splits_dir,
        output_dir=output_dir / "subsets",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=image_size,
        aug_cfg=aug_cfg,
        seed=args.seed,
    )

    # --- Pilot conversion + sanity check ---------------------------------
    pilot_tag, pilot_rel = CHECKPOINT_TAGS[0]
    pilot_ckpt = ckpt_root / pilot_rel
    if not pilot_ckpt.is_file():
        logger.error("Pilot checkpoint missing: %s", pilot_ckpt)
        return 1

    logger.info("=" * 60)
    logger.info("Pilot conversion: %s", pilot_tag)
    logger.info("=" * 60)
    pilot_model, pilot_meta = _load_student(pilot_ckpt, student_cfg=student_cfg)
    logger.info(
        "Loaded pilot %s (epoch=%s, best_val_auc=%s)",
        pilot_ckpt.name,
        pilot_meta.get("epoch"),
        pilot_meta.get("best_val_auc"),
    )

    pilot_artifacts = _convert_one_checkpoint(
        pilot_model,
        run_tag=pilot_tag,
        conversion_root=conversion_root,
        representative_loader=calib_loader,
        image_size=image_size,
        calibration_batches=args.calibration_batches,
        coreml_enabled=coreml_enabled,
    )
    sanity_report = _run_sanity_check(
        pilot_model, pilot_artifacts, image_size=image_size
    )
    logger.info("Sanity check:\n%s", sanity_report.summary())
    (results_dir / "sanity_report.json").write_text(
        json.dumps(sanity_report.as_dict(), indent=2), encoding="utf-8"
    )

    if not sanity_report.all_passed():
        logger.warning(
            "Some sanity checks did NOT pass. Inspect conversion logs before "
            "trusting downstream numbers. Continuing anyway."
        )

    if args.pilot_only:
        logger.info("--pilot-only set — stopping after sanity check.")
        return 0

    # --- Mass conversion + eval ------------------------------------------
    all_artifacts: dict[str, dict[str, Path]] = {pilot_tag: pilot_artifacts}
    for run_tag, ckpt_rel in CHECKPOINT_TAGS[1:]:
        ckpt = ckpt_root / ckpt_rel
        if not ckpt.is_file():
            logger.warning("Checkpoint missing — skipping %s (%s)", run_tag, ckpt)
            continue
        logger.info("Converting %s ...", run_tag)
        model, _ = _load_student(ckpt, student_cfg=student_cfg)
        all_artifacts[run_tag] = _convert_one_checkpoint(
            model,
            run_tag=run_tag,
            conversion_root=conversion_root,
            representative_loader=calib_loader,
            image_size=image_size,
            calibration_batches=args.calibration_batches,
            coreml_enabled=coreml_enabled,
        )

    # --- AUC eval --------------------------------------------------------
    eval_rows: list[EvalRow] = []

    # PyTorch baseline (CPU) for the gen3 seed-0 final model — reference number.
    logger.info("PyTorch CPU baseline (reference) ...")
    pt_ckpt = ckpt_root / "gen3_replay+ewc_seed0/best.pth"
    if pt_ckpt.is_file():
        pt_model, _ = _load_student(pt_ckpt, student_cfg=student_cfg)
        for split, loader in test_loaders.items():
            pt = _evaluate_pytorch_baseline(pt_model, loader)
            eval_rows.append(
                EvalRow(
                    run_tag="gen3_replay+ewc_seed0",
                    runtime="pytorch",
                    mode="fp32",
                    test_split=split,
                    num_samples=pt["num_samples"],
                    auc=pt["auc"],
                    log_loss=pt["log_loss"],
                    accuracy=pt["accuracy"],
                    artifact_path=str(pt_ckpt),
                    size_mb=file_size_mb(pt_ckpt),
                )
            )
            logger.info(
                "  pytorch fp32 %s: auc=%.4f n=%d",
                split,
                pt["auc"],
                pt["num_samples"],
            )

    # Edge artifacts: TFLite + CoreML, all 9 ckpts x all splits.
    for run_tag, artifacts in all_artifacts.items():
        for key, art_path in artifacts.items():
            runtime, mode = key.split("_", 1)
            if runtime == "coreml" and not (ON_MACOS and coreml_enabled):
                continue
            size_mb = (
                file_size_mb(art_path)
                if runtime == "tflite"
                else _mlpackage_size_mb_safe(art_path)
            )
            for split, loader in test_loaders.items():
                try:
                    res = _eval_artifact_on_split(art_path, runtime, mode, loader)
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "Eval failed for %s/%s/%s/%s: %s",
                        run_tag,
                        runtime,
                        mode,
                        split,
                        exc,
                    )
                    continue
                eval_rows.append(
                    EvalRow(
                        run_tag=run_tag,
                        runtime=runtime,
                        mode=mode,
                        test_split=split,
                        num_samples=res["num_samples"],
                        auc=res["auc"],
                        log_loss=res["log_loss"],
                        accuracy=res["accuracy"],
                        artifact_path=str(art_path),
                        size_mb=size_mb,
                    )
                )
                logger.info(
                    "  %s %s %s %s: auc=%.4f n=%d",
                    run_tag,
                    runtime,
                    mode,
                    split,
                    res["auc"],
                    res["num_samples"],
                )

    # --- Latency benchmark on pilot only --------------------------------
    logger.info("Latency benchmark (pilot=%s) ...", pilot_tag)
    latency_rows = _benchmark_latency(
        pilot_artifacts,
        image_size=image_size,
        num_runs=args.num_latency_runs,
        num_warmup=args.num_warmup,
        coreml_enabled=coreml_enabled,
    )
    for r in latency_rows:
        logger.info(
            "  %s/%s/%s: mean=%.2fms p95=%.2fms",
            r.runtime,
            r.mode,
            r.compute_unit,
            r.latency_ms_mean,
            r.latency_ms_p95,
        )

    # --- Write outputs ---------------------------------------------------
    finished_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    elapsed_s = time.time() - overall_start

    payload: dict[str, Any] = {
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_seconds": float(elapsed_s),
        "platform": platform.platform(),
        "config": str(args.config),
        "coreml_enabled": bool(coreml_enabled),
        "num_checkpoints_converted": len(all_artifacts),
        "test_splits": {k: str(v) for k, v in test_csvs.items()},
        "args": vars(args).copy(),
        "sanity_report": sanity_report.as_dict(),
        "latency": [r.as_dict() for r in latency_rows],
        "evaluation": [r.as_dict() for r in eval_rows],
    }
    payload["args"]["output_dir"] = str(args.output_dir)
    json_path = results_dir / "edge_eval_full.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote results JSON: %s", json_path)

    md_path = results_dir / "edge_eval_summary.md"
    _write_summary_markdown(
        output_path=md_path,
        eval_rows=eval_rows,
        latency_rows=latency_rows,
        sanity_report=sanity_report,
        started_at=started_at,
        finished_at=finished_at,
    )
    logger.info("Wrote markdown summary: %s", md_path)
    logger.info("Total elapsed: %.1f min", elapsed_s / 60.0)
    return 0


def _mlpackage_size_mb_safe(path: Path) -> float:
    """Wrapper that imports the CoreML helper lazily."""
    try:
        from src.evaluation.edge_eval_coreml import _mlpackage_size_mb
        return _mlpackage_size_mb(path)
    except Exception:  # noqa: BLE001
        return 0.0


if __name__ == "__main__":
    raise SystemExit(_run(_parse_args()))
