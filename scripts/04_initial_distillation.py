"""Step 4: Train the student on Gen 1 via knowledge distillation.

CLI entrypoint. Not imported by other code.

Usage:
    python scripts/04_initial_distillation.py --config configs/default.yaml --seed 0
    python scripts/04_initial_distillation.py --generation gen1 --batch-size 32 --num-workers 4
    python scripts/04_initial_distillation.py --no-soft-labels   # ablation: CE only

Calls:
    src/models/students/mobilenetv3.py (build_student)
    src/training/trainer.py (DistillationTrainer, TrainerConfig)
    src/training/losses.py (DistillationLoss)
    src/evaluation/evaluator.py (evaluate_loaders, write_metrics_json)
    src/data/dataloader.py (build_dataloader)
    src/utils/config.py (load_config)
    src/utils/checkpoint.py (save_checkpoint)
    src/utils/logger.py (get_logger)
Reads:
    YAML config (--config)
    Split CSVs at {drive}/datasets/splits/{generation}_{train|val|test}.csv
    Face JPEGs referenced in the CSVs
    Ensemble soft labels at
        {drive}/soft_labels/{generation}/{train|val|test}/ensemble.npy
        (unless --no-soft-labels is passed)
Writes:
    Checkpoints at {drive}/checkpoints/students/{generation}/{best,last}.pth
        Keys: epoch, model_state_dict, optimizer_state_dict, scheduler_state_dict,
        best_val_auc, config, generation, metrics, timestamp (ISO 8601)
    Test metrics JSON at {drive}/results/raw/{generation}_initial_metrics.json
        ``generated_at`` in ISO 8601 ``%Y-%m-%dT%H:%M:%S``
    Log file at runs/initial_distillation_{generation}.log
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

# Make ``src`` importable when invoked as ``python scripts/04_initial_distillation.py``.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.dataloader import build_dataloader  # noqa: E402
from src.evaluation.evaluator import evaluate_loaders, write_metrics_json  # noqa: E402
from src.models.students.mobilenetv3 import build_student  # noqa: E402
from src.training.trainer import DistillationTrainer, TrainerConfig  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_soft_label_path(
    drive_root: Path, generation: str, split: str, use_soft: bool
) -> Path | None:
    if not use_soft:
        return None
    return drive_root / "soft_labels" / generation / split / "ensemble.npy"


def _build_trainer_config(training_cfg: dict) -> TrainerConfig:
    return TrainerConfig(
        learning_rate=float(training_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
        num_epochs=int(training_cfg.get("num_epochs", 30)),
        alpha=float(training_cfg.get("alpha", 0.7)),
        temperature=float(training_cfg.get("temperature", 4.0)),
        early_stopping_patience=int(
            training_cfg.get("early_stopping_patience", 5)
        ),
    )


def _run(args: argparse.Namespace) -> int:
    """Top-level entry: dispatch single-seed or multi-seed run.

    Multi-seed runs (``--num-seeds N`` with N>1) execute the training N
    times sequentially with seeds 0..N-1 and write per-seed checkpoints +
    metrics JSON. Use ``scripts/aggregate_seeds.py`` to compute mean/std
    across the resulting JSONs.
    """
    if args.num_seeds <= 1:
        return _run_single_seed(args, seed=args.seed)

    last_rc = 0
    for seed in range(args.num_seeds):
        rc = _run_single_seed(args, seed=seed, seed_suffix=f"_seed{seed}")
        last_rc = rc or last_rc
    return last_rc


def _run_single_seed(
    args: argparse.Namespace, *, seed: int, seed_suffix: str = ""
) -> int:
    """Run one initial-distillation training pass with the given seed.

    ``seed_suffix`` (e.g. ``"_seed0"``) is appended to checkpoint dir and
    metrics filename so multi-seed runs don't clobber each other. When the
    suffix is empty (single-seed run) outputs land at the historical paths
    so existing artefacts stay backward-compatible.
    """
    logger = get_logger(
        "initial_distillation",
        log_file=f"runs/initial_distillation_{args.generation}{seed_suffix}.log",
    )
    _seed_everything(seed)

    cfg = load_config(args.config)
    drive_root = Path(cfg["paths"]["drive_root"])
    training_cfg = cfg["training"]["initial_distillation"]
    student_cfg = cfg["student"]
    aug_cfg = cfg["data"]["augmentation"]
    image_size = int(student_cfg.get("input_size", 224))
    batch_size = int(args.batch_size or training_cfg.get("batch_size", 64))

    splits_dir = (
        Path(args.splits_dir)
        if args.splits_dir
        else drive_root / "datasets" / "splits"
    )
    split_csvs = {
        split: splits_dir / f"{args.generation}_{split}.csv"
        for split in ("train", "val", "test")
    }
    for split, path in split_csvs.items():
        if not path.is_file():
            logger.error(
                "Missing split CSV %s — run scripts/02_generate_splits.py first",
                path,
            )
            return 1

    use_soft = not args.no_soft_labels
    soft_label_paths = {
        split: _resolve_soft_label_path(drive_root, args.generation, split, use_soft)
        for split in split_csvs
    }
    if use_soft:
        for split, sl_path in soft_label_paths.items():
            if sl_path is None or not sl_path.is_file():
                logger.error(
                    "Missing ensemble soft labels for %s/%s: %s. "
                    "Run scripts/03_generate_soft_labels.py or pass --no-soft-labels.",
                    args.generation,
                    split,
                    sl_path,
                )
                return 1

    # --- DataLoaders ---------------------------------------------------
    logger.info(
        "Building dataloaders for %s (batch_size=%d, image_size=%d, soft=%s)",
        args.generation,
        batch_size,
        image_size,
        use_soft,
    )
    train_loader = build_dataloader(
        csv_path=split_csvs["train"],
        mode="train",
        batch_size=batch_size,
        soft_label_path=soft_label_paths["train"],
        image_size=image_size,
        aug_cfg=aug_cfg,
        num_workers=args.num_workers,
    )
    val_loader = build_dataloader(
        csv_path=split_csvs["val"],
        mode="val",
        batch_size=batch_size,
        soft_label_path=soft_label_paths["val"],
        image_size=image_size,
        aug_cfg=aug_cfg,
        num_workers=args.num_workers,
    )
    test_loader = build_dataloader(
        csv_path=split_csvs["test"],
        mode="test",
        batch_size=batch_size,
        soft_label_path=soft_label_paths["test"],
        image_size=image_size,
        aug_cfg=aug_cfg,
        num_workers=args.num_workers,
    )

    # --- Model ---------------------------------------------------------
    model = build_student(
        hidden_dim=int(student_cfg.get("head", {}).get("hidden_dim", 256)),
        dropout=float(student_cfg.get("head", {}).get("dropout", 0.2)),
        num_classes=int(student_cfg.get("num_classes", 2)),
        pretrained=(student_cfg.get("pretrained", "imagenet") is not None),
    )
    logger.info(
        "Student built: %s params (trainable=%d)",
        f"{model.num_parameters():,}",
        model.num_parameters(trainable_only=True),
    )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    trainer_cfg = _build_trainer_config(training_cfg)

    # --- Fit -----------------------------------------------------------
    checkpoint_dir = (
        Path(args.checkpoint_dir)
        if args.checkpoint_dir
        else drive_root / "checkpoints" / "students" / f"{args.generation}{seed_suffix}"
    )
    trainer = DistillationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trainer_cfg,
        device=device,
        checkpoint_dir=checkpoint_dir,
        generation=args.generation,
        run_config={
            "seed": seed,
            "use_soft_labels": use_soft,
            "training": training_cfg,
            "student": student_cfg,
        },
    )
    logger.info(
        "Starting initial distillation: generation=%s alpha=%.2f T=%.1f "
        "epochs=%d lr=%.2e patience=%d device=%s",
        args.generation,
        trainer_cfg.alpha,
        trainer_cfg.temperature,
        trainer_cfg.num_epochs,
        trainer_cfg.learning_rate,
        trainer_cfg.early_stopping_patience,
        device,
    )
    trainer.fit()
    logger.info(
        "Training done. Best val AUC: %.4f (checkpoints in %s)",
        trainer.best_val_auc,
        checkpoint_dir,
    )

    # --- Test evaluation ----------------------------------------------
    test_metrics = evaluate_loaders(
        model,
        {f"{args.generation}_test": test_loader},
        device=device,
    )

    results_dir = (
        Path(args.results_dir)
        if args.results_dir
        else drive_root / "results" / "raw"
    )
    metrics_path = (
        results_dir / f"{args.generation}_initial_metrics{seed_suffix}.json"
    )
    write_metrics_json(
        metrics_path,
        generation=args.generation,
        split_metrics=test_metrics,
        extra={
            "checkpoint_path": str(checkpoint_dir / "best.pth"),
            "best_val_auc": float(trainer.best_val_auc),
            "elapsed_seconds": float(getattr(trainer, "elapsed_seconds", 0.0)),
            "gpu_hours": float(getattr(trainer, "gpu_hours", 0.0)),
            "use_soft_labels": use_soft,
            "seed": seed,
        },
    )
    logger.info("Wrote test metrics: %s", metrics_path)
    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train student model on a single generation via KD"
    )
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument(
        "--generation",
        default="gen1",
        choices=["gen1", "gen2", "gen3"],
        help="Training generation (initial distillation uses gen1)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--no-soft-labels",
        action="store_true",
        help="Ablation: train with CE only, ignoring ensemble soft labels",
    )
    p.add_argument(
        "--splits-dir",
        default=None,
        help="Override splits dir. Default: {drive}/datasets/splits. "
        "Use this when reading from a local mirror to bypass Drive FUSE.",
    )
    p.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help="Run training N times with seeds 0..N-1, save metrics per-seed. "
        "Default 1 (backward compat). Pass 3 for thesis-grade reporting "
        "(mean +/- std via scripts/aggregate_seeds.py).",
    )
    p.add_argument("--checkpoint-dir", default=None)
    p.add_argument("--results-dir", default=None)
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(_run(_parse_args()))
