"""Step 5: Update student on Gen 2 / Gen 3 with anti-forgetting.

CLI entrypoint. Not imported by other code.

Usage:
    python scripts/05_continual_distillation.py --generation gen2 --method replay --seed 0
    python scripts/05_continual_distillation.py --generation gen3 --method lwf
    python scripts/05_continual_distillation.py --generation gen2 --method ewc --previous gen1

Calls:
    src/training/continual_trainer.py (ContinualTrainer, ContinualTrainerConfig)
    src/training/anti_forgetting/{ewc,lwf,replay}.py
    src/models/students/mobilenetv3.py (build_student)
    src/evaluation/evaluator.py (evaluate_loaders, summarize_cgrs, write_metrics_json)
    src/data/dataloader.py (build_dataloader)
    src/utils/config.py (load_config)
    src/utils/checkpoint.py (load_checkpoint)
    src/utils/logger.py (get_logger)
Reads:
    YAML config (--config)
    Previous checkpoint at
        {drive}/checkpoints/students/{previous_generation}/best.pth
    Split CSVs at {drive}/datasets/splits/{generation}_{train|val|test}.csv
    Split CSVs for all already-seen generations (for cross-gen test evaluation)
    Ensemble soft labels at
        {drive}/soft_labels/{generation}/{train|val|test}/ensemble.npy
Writes:
    Updated checkpoints at
        {drive}/checkpoints/students/{generation}_{method}/{best,last}.pth
        (ISO 8601 timestamps, full strategy_state in ``config`` block)
    Cross-gen metrics JSON at
        {drive}/results/raw/{generation}_{method}_continual_metrics.json
        Structure includes per-gen AUC, CGRS summary, and the previous-gen
        peak AUCs used for retention scoring.
    Log file at runs/continual_distillation_{generation}_{method}.log
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.dataloader import build_dataloader  # noqa: E402
from src.evaluation.evaluator import (  # noqa: E402
    evaluate_loaders,
    summarize_cgrs,
    write_metrics_json,
)
from src.models.students.mobilenetv3 import build_student  # noqa: E402
from src.training.anti_forgetting.base import AntiForgettingStrategy  # noqa: E402
from src.training.anti_forgetting.ewc import EWCStrategy  # noqa: E402
from src.training.anti_forgetting.lwf import LwFStrategy  # noqa: E402
from src.training.anti_forgetting.replay import ReplayStrategy  # noqa: E402
from src.training.continual_trainer import (  # noqa: E402
    ContinualTrainer,
    ContinualTrainerConfig,
)
from src.utils.checkpoint import load_checkpoint  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

_GENERATION_ORDER = ("gen1", "gen2", "gen3")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _previous_generation(generation: str) -> str:
    try:
        idx = _GENERATION_ORDER.index(generation)
    except ValueError as exc:
        raise ValueError(f"Unknown generation: {generation}") from exc
    if idx == 0:
        raise ValueError(
            f"'{generation}' is the first generation — use scripts/04 for initial distillation"
        )
    return _GENERATION_ORDER[idx - 1]


def _seen_generations(generation: str) -> list[str]:
    idx = _GENERATION_ORDER.index(generation)
    return list(_GENERATION_ORDER[: idx + 1])


def _build_trainer_config(training_cfg: dict) -> ContinualTrainerConfig:
    return ContinualTrainerConfig(
        learning_rate=float(training_cfg.get("learning_rate", 5e-5)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
        num_epochs=int(training_cfg.get("num_epochs", 10)),
        alpha=float(training_cfg.get("alpha", 0.5)),
        beta=float(training_cfg.get("beta", 0.3)),
        gamma=float(training_cfg.get("gamma", 0.2)),
        temperature=float(training_cfg.get("temperature", 4.0)),
        early_stopping_patience=int(
            training_cfg.get("early_stopping_patience", 3)
        ),
    )


def _build_strategy(
    method: str,
    *,
    cfg: dict,
    previous_generation: str,
    drive_root: Path,
    buffer_output_dir: Path,
    batch_size: int,
    num_workers: int,
    image_size: int,
    aug_cfg: dict,
) -> AntiForgettingStrategy:
    method = method.lower()
    af_cfg = cfg["training"]["anti_forgetting"]
    if method == "ewc":
        ewc_cfg = af_cfg.get("ewc", {})
        return EWCStrategy(
            lambda_=float(ewc_cfg.get("lambda", 5000)),
            fisher_samples=int(ewc_cfg.get("fisher_samples", 1000)),
        )
    if method == "lwf":
        lwf_cfg = af_cfg.get("lwf", {})
        return LwFStrategy(temperature=float(lwf_cfg.get("temperature", 2.0)))
    if method == "replay":
        replay_cfg = af_cfg.get("replay", {})
        return ReplayStrategy(
            previous_train_csv=(
                drive_root / "datasets" / "splits" / f"{previous_generation}_train.csv"
            ),
            previous_soft_label_path=(
                drive_root
                / "soft_labels"
                / previous_generation
                / "train"
                / "ensemble.npy"
            ),
            buffer_output_dir=buffer_output_dir,
            previous_generation=previous_generation,
            buffer_percentage=float(replay_cfg.get("buffer_percentage", 0.05)),
            selection=str(replay_cfg.get("selection", "herding")),
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=image_size,
            aug_cfg=aug_cfg,
        )
    raise ValueError(
        f"Unknown anti-forgetting method: {method!r}. Expected ewc / lwf / replay."
    )


def _peak_auc_from_previous_runs(
    results_dir: Path, generation: str
) -> float | None:
    """Look up the peak test AUC reported for ``generation`` in earlier rounds.

    Checks, in order: the initial-distillation metrics for gen1, then any
    continual-distillation metrics whose generation matches. Returns ``None``
    if no prior JSON contains the key (the caller falls back gracefully).
    """
    candidates = []
    candidates.extend(results_dir.glob(f"{generation}_initial_metrics.json"))
    candidates.extend(results_dir.glob(f"{generation}_*_continual_metrics.json"))

    best: float | None = None
    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        splits = payload.get("splits") or {}
        split_metrics = splits.get(f"{generation}_test")
        if not split_metrics:
            continue
        auc = split_metrics.get("auc")
        if auc is None:
            continue
        auc_f = float(auc)
        if best is None or auc_f > best:
            best = auc_f
    return best


def _run(args: argparse.Namespace) -> int:
    logger = get_logger(
        f"continual_distillation.{args.method}",
        log_file=f"runs/continual_distillation_{args.generation}_{args.method}.log",
    )
    _seed_everything(args.seed)

    cfg = load_config(args.config)
    drive_root = Path(cfg["paths"]["drive_root"])
    training_cfg = cfg["training"]["continual_distillation"]
    student_cfg = cfg["student"]
    aug_cfg = cfg["data"]["augmentation"]
    image_size = int(student_cfg.get("input_size", 224))
    batch_size = int(args.batch_size or training_cfg.get("batch_size", 64))

    previous_generation = args.previous or _previous_generation(args.generation)

    splits_dir = drive_root / "datasets" / "splits"
    new_split_csvs = {
        split: splits_dir / f"{args.generation}_{split}.csv"
        for split in ("train", "val", "test")
    }
    for split, path in new_split_csvs.items():
        if not path.is_file():
            logger.error(
                "Missing split CSV %s — run scripts/02_generate_splits.py first",
                path,
            )
            return 1

    soft_label_paths = {
        split: drive_root / "soft_labels" / args.generation / split / "ensemble.npy"
        for split in new_split_csvs
    }
    use_soft = not args.no_soft_labels
    if use_soft:
        for split, sl_path in soft_label_paths.items():
            if not sl_path.is_file():
                logger.error(
                    "Missing soft labels for %s/%s: %s "
                    "(run scripts/03_generate_soft_labels.py or pass --no-soft-labels)",
                    args.generation,
                    split,
                    sl_path,
                )
                return 1
    else:
        soft_label_paths = {split: None for split in soft_label_paths}

    # --- Loaders -------------------------------------------------------
    new_train_loader = build_dataloader(
        csv_path=new_split_csvs["train"],
        mode="train",
        batch_size=batch_size,
        soft_label_path=soft_label_paths["train"],
        image_size=image_size,
        aug_cfg=aug_cfg,
        num_workers=args.num_workers,
    )
    new_val_loader = build_dataloader(
        csv_path=new_split_csvs["val"],
        mode="val",
        batch_size=batch_size,
        soft_label_path=soft_label_paths["val"],
        image_size=image_size,
        aug_cfg=aug_cfg,
        num_workers=args.num_workers,
    )
    new_test_loader = build_dataloader(
        csv_path=new_split_csvs["test"],
        mode="test",
        batch_size=batch_size,
        soft_label_path=soft_label_paths["test"],
        image_size=image_size,
        aug_cfg=aug_cfg,
        num_workers=args.num_workers,
    )

    # Previous-generation val loader (needed by EWC for Fisher estimation).
    previous_val_csv = splits_dir / f"{previous_generation}_val.csv"
    if not previous_val_csv.is_file():
        logger.error(
            "Missing previous val CSV %s — cannot proceed with continual training",
            previous_val_csv,
        )
        return 1
    previous_val_loader = build_dataloader(
        csv_path=previous_val_csv,
        mode="val",
        batch_size=batch_size,
        soft_label_path=None,
        image_size=image_size,
        aug_cfg=aug_cfg,
        num_workers=args.num_workers,
    )

    # --- Student + previous checkpoint --------------------------------
    model = build_student(
        hidden_dim=int(student_cfg.get("head", {}).get("hidden_dim", 256)),
        dropout=float(student_cfg.get("head", {}).get("dropout", 0.2)),
        num_classes=int(student_cfg.get("num_classes", 2)),
        pretrained=False,  # we're overwriting with the previous checkpoint below
    )

    prev_ckpt_path = (
        Path(args.previous_checkpoint)
        if args.previous_checkpoint
        else drive_root / "checkpoints" / "students" / previous_generation / "best.pth"
    )
    if not prev_ckpt_path.is_file():
        logger.error(
            "Previous checkpoint not found: %s — run scripts/04 or an earlier scripts/05 first",
            prev_ckpt_path,
        )
        return 1

    prev_meta = load_checkpoint(prev_ckpt_path, model=model, strict=False)
    logger.info(
        "Loaded previous student from %s (epoch=%s, best_val_auc=%s)",
        prev_ckpt_path,
        prev_meta.get("epoch"),
        prev_meta.get("best_val_auc"),
    )
    if prev_meta.get("missing_keys"):
        logger.warning(
            "Missing %d keys when restoring previous student (first 3: %s)",
            len(prev_meta["missing_keys"]),
            prev_meta["missing_keys"][:3],
        )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # --- Strategy ------------------------------------------------------
    buffer_output_dir = (
        drive_root
        / "checkpoints"
        / "students"
        / f"{args.generation}_{args.method}"
        / "replay_buffer"
    )
    strategy = _build_strategy(
        args.method,
        cfg=cfg,
        previous_generation=previous_generation,
        drive_root=drive_root,
        buffer_output_dir=buffer_output_dir,
        batch_size=batch_size,
        num_workers=args.num_workers,
        image_size=image_size,
        aug_cfg=aug_cfg,
    )
    logger.info(
        "Anti-forgetting strategy: %s (previous_generation=%s)",
        strategy.name,
        previous_generation,
    )
    strategy.before_training(
        model,
        previous_checkpoint=prev_ckpt_path,
        previous_val_loader=previous_val_loader,
        device=device,
    )

    # --- Trainer -------------------------------------------------------
    trainer_cfg = _build_trainer_config(training_cfg)
    checkpoint_dir = (
        Path(args.checkpoint_dir)
        if args.checkpoint_dir
        else drive_root / "checkpoints" / "students" / f"{args.generation}_{args.method}"
    )
    trainer = ContinualTrainer(
        model=model,
        new_train_loader=new_train_loader,
        new_val_loader=new_val_loader,
        strategy=strategy,
        config=trainer_cfg,
        device=device,
        checkpoint_dir=checkpoint_dir,
        generation=args.generation,
        run_config={
            "seed": args.seed,
            "method": args.method,
            "previous_generation": previous_generation,
            "previous_checkpoint": str(prev_ckpt_path),
            "use_soft_labels": use_soft,
            "training": training_cfg,
            "student": student_cfg,
            "anti_forgetting": cfg["training"]["anti_forgetting"],
        },
    )
    logger.info(
        "Starting continual distillation: gen=%s method=%s alpha=%.2f beta=%.2f gamma=%.2f T=%.1f",
        args.generation,
        args.method,
        trainer_cfg.alpha,
        trainer_cfg.beta,
        trainer_cfg.gamma,
        trainer_cfg.temperature,
    )
    trainer.fit()
    logger.info(
        "Continual training done. Best val AUC on %s: %.4f",
        args.generation,
        trainer.best_val_auc,
    )

    # --- Cross-generation evaluation ----------------------------------
    seen = _seen_generations(args.generation)
    test_loaders = {
        f"{gen}_test": build_dataloader(
            csv_path=splits_dir / f"{gen}_test.csv",
            mode="test",
            batch_size=batch_size,
            soft_label_path=None,
            image_size=image_size,
            aug_cfg=aug_cfg,
            num_workers=args.num_workers,
        )
        for gen in seen
    }
    test_loaders[f"{args.generation}_test"] = new_test_loader  # reuse if already built

    all_metrics = evaluate_loaders(model, test_loaders, device=device)

    # --- CGRS against the peak AUCs from prior rounds ------------------
    results_dir = (
        Path(args.results_dir)
        if args.results_dir
        else drive_root / "results" / "raw"
    )
    auc_after = {gen: all_metrics[f"{gen}_test"].auc for gen in seen}
    auc_peak: dict[str, float] = {}
    for gen in seen:
        peak = _peak_auc_from_previous_runs(results_dir, gen)
        auc_peak[gen] = peak if peak is not None else auc_after[gen]
    cgrs_block = summarize_cgrs(auc_after, auc_peak)

    extra: dict[str, Any] = {
        "method": args.method,
        "previous_generation": previous_generation,
        "previous_checkpoint": str(prev_ckpt_path),
        "checkpoint_path": str(checkpoint_dir / "best.pth"),
        "best_val_auc": float(trainer.best_val_auc),
        "auc_after": auc_after,
        "auc_peak": auc_peak,
        **cgrs_block,
    }

    metrics_path = (
        results_dir
        / f"{args.generation}_{args.method}_continual_metrics.json"
    )
    write_metrics_json(
        metrics_path,
        generation=args.generation,
        split_metrics=all_metrics,
        extra=extra,
    )
    logger.info(
        "Wrote cross-gen metrics: %s (CGRS=%.4f)",
        metrics_path,
        cgrs_block["cgrs"],
    )
    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Continual distillation with anti-forgetting on Gen 2 / Gen 3"
    )
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument(
        "--generation",
        required=True,
        choices=["gen2", "gen3"],
        help="Which generation to train on (must be > gen1)",
    )
    p.add_argument(
        "--method",
        required=True,
        choices=["ewc", "lwf", "replay"],
        help="Anti-forgetting strategy to use",
    )
    p.add_argument(
        "--previous",
        default=None,
        help="Previous generation id. Defaults to the one just before --generation.",
    )
    p.add_argument(
        "--previous-checkpoint",
        default=None,
        help=(
            "Override the previous-student checkpoint path. Defaults to "
            "{drive}/checkpoints/students/{previous}/best.pth"
        ),
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--no-soft-labels",
        action="store_true",
        help="Ablation: skip the teacher KD term (gamma-only continual CE + retention)",
    )
    p.add_argument("--checkpoint-dir", default=None)
    p.add_argument("--results-dir", default=None)
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(_run(_parse_args()))
