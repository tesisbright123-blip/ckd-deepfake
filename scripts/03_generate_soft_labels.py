"""Step 3: Teacher inference to generate soft labels.

CLI entrypoint. Not imported by other code.

Usage:
    python scripts/03_generate_soft_labels.py --config configs/default.yaml --generation gen1
    python scripts/03_generate_soft_labels.py --generation gen2 --teachers efficientnet_b4,clip_vit_l14
    python scripts/03_generate_soft_labels.py --generation gen3 --batch-size 32 --num-workers 4

Calls:
    src/models/teachers/efficientnet_b4.py (EfficientNetB4Teacher)
    src/models/teachers/recce.py (RECCETeacher)
    src/models/teachers/clip_detector.py (CLIPDetectorTeacher)
    src/models/teachers/ensemble.py (compute_val_auc, softmax_weights, aggregate)
    src/data/dataset.py (not used directly — teachers need their own preprocessing)
    src/utils/config.py (load_config)
    src/utils/logger.py (get_logger)
Reads:
    YAML config (--config)
    Split CSVs at {drive}/datasets/splits/{generation}_{train|val|test}.csv
        columns: face_path, frame_idx, video_id, label, dataset, generation, technique
    Face JPEGs referenced by ``face_path``
    Teacher checkpoints declared under ``teacher.models`` in the YAML
        (e.g. {drive}/checkpoints/teachers/efficientnet_b4.pth)
Writes:
    Per-teacher soft labels at
        {drive}/soft_labels/{generation}/{train|val|test}/{teacher_name}.npy
        shape (N,), dtype float32, values in [0, 1] (P(fake))
    Ensemble soft labels at
        {drive}/soft_labels/{generation}/{train|val|test}/ensemble.npy
        shape (N,), dtype float32, values in [0, 1]
    Manifest at {drive}/soft_labels/{generation}/ensemble_weights.json
        ``generated_at`` in ISO 8601 ``%Y-%m-%dT%H:%M:%S``
    Log file at runs/generate_soft_labels_{generation}.log
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# Make ``src`` importable when invoked as ``python scripts/03_generate_soft_labels.py``.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.models.teachers.base import BaseTeacher  # noqa: E402
from src.models.teachers.clip_detector import CLIPDetectorTeacher  # noqa: E402
from src.models.teachers.efficientnet_b4 import EfficientNetB4Teacher  # noqa: E402
from src.models.teachers.ensemble import (  # noqa: E402
    aggregate,
    compute_val_auc,
    softmax_weights,
)
from src.models.teachers.recce import RECCETeacher  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

_SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")
_DEFAULT_TEACHERS: tuple[str, ...] = (
    "efficientnet_b4",
    "recce",
    "clip_vit_l14",
)
_WEIGHTING_TEMPERATURE: float = 1.0


# ---------------------------------------------------------------------- #
#  Dataset: a thin wrapper that applies the teacher's own preprocessing
#  pipeline to every face referenced in the split CSV, preserving CSV row
#  order so the saved .npy aligns with DeepfakeDataset downstream.
# ---------------------------------------------------------------------- #
class _TeacherInferenceDataset(Dataset):
    """Reads faces in CSV order and applies a teacher's preprocessing transform."""

    def __init__(self, csv_path: str | Path, transform):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path, usecols=["face_path"]).reset_index(
            drop=True
        )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Local import keeps cv2 out of the module import graph when the
        # script is only being inspected (e.g. by unit tests).
        import cv2

        path = str(self.df.iloc[idx]["face_path"])
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.transform(image=img)["image"]


# ---------------------------------------------------------------------- #
#  Teacher registry
# ---------------------------------------------------------------------- #
def _build_teacher(name: str, device: str) -> BaseTeacher:
    builders = {
        "efficientnet_b4": lambda: EfficientNetB4Teacher(device=device),
        "recce": lambda: RECCETeacher(device=device),
        "clip_vit_l14": lambda: CLIPDetectorTeacher(device=device),
    }
    if name not in builders:
        raise ValueError(
            f"Unknown teacher '{name}'. Supported: {sorted(builders.keys())}"
        )
    return builders[name]()


def _weight_path_for(name: str, cfg: dict) -> Path:
    """Resolve the teacher's checkpoint path from the YAML config."""
    for spec in cfg["teacher"]["models"]:
        if spec.get("name") == name:
            wp = spec.get("weight_path")
            if not wp:
                raise KeyError(
                    f"Teacher '{name}' has no 'weight_path' in teacher.models"
                )
            return Path(wp)
    raise KeyError(f"Teacher '{name}' is not declared in teacher.models")


# ---------------------------------------------------------------------- #
#  Inference core
# ---------------------------------------------------------------------- #
def _run_teacher_inference(
    teacher: BaseTeacher,
    csv_path: Path,
    *,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    """Run ``teacher`` on every row of ``csv_path`` and return a (N,) float32 array."""
    transform = teacher.get_preprocessing()
    dataset = _TeacherInferenceDataset(csv_path, transform)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )

    out = np.empty(len(dataset), dtype=np.float32)
    cursor = 0
    for batch in loader:
        probs = teacher.predict_proba(batch)
        n = int(probs.shape[0])
        out[cursor : cursor + n] = probs
        cursor += n
    if cursor != len(dataset):
        raise RuntimeError(
            f"Inference collected {cursor} samples but dataset has {len(dataset)}"
        )
    return out


def _load_val_labels(val_csv: Path) -> np.ndarray:
    df = pd.read_csv(val_csv, usecols=["label"])
    return df["label"].to_numpy().astype(np.int64)


# ---------------------------------------------------------------------- #
#  Orchestration
# ---------------------------------------------------------------------- #
def _run(args: argparse.Namespace) -> int:
    logger = get_logger(
        "generate_soft_labels",
        log_file=f"runs/generate_soft_labels_{args.generation}.log",
    )

    cfg = load_config(args.config)
    drive_root = Path(cfg["paths"]["drive_root"])

    splits_dir = (
        Path(args.splits_dir)
        if args.splits_dir
        else drive_root / "datasets" / "splits"
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else drive_root / "soft_labels" / args.generation
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    split_csvs: dict[str, Path] = {
        split: splits_dir / f"{args.generation}_{split}.csv"
        for split in _SPLIT_NAMES
    }
    for split, path in split_csvs.items():
        if not path.is_file():
            logger.error(
                "Missing split CSV for %s/%s: %s. Run scripts/02_generate_splits.py first.",
                args.generation,
                split,
                path,
            )
            return 1

    requested = (
        tuple(t.strip() for t in args.teachers.split(",") if t.strip())
        if args.teachers
        else _DEFAULT_TEACHERS
    )
    if not requested:
        logger.error("No teachers requested.")
        return 1

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        "Generating soft labels for %s: teachers=%s device=%s batch_size=%d",
        args.generation,
        list(requested),
        device,
        args.batch_size,
    )

    # per_teacher_soft[teacher_name][split] -> (N,) float32
    per_teacher_soft: dict[str, dict[str, np.ndarray]] = {}

    for teacher_name in requested:
        teacher: BaseTeacher | None = None
        try:
            teacher = _build_teacher(teacher_name, device)
            weight_path = _weight_path_for(teacher_name, cfg)
            teacher.load(weight_path)
        except (FileNotFoundError, ModuleNotFoundError, KeyError, AttributeError) as exc:
            logger.error(
                "Skipping teacher '%s' — load failed: %s", teacher_name, exc
            )
            if teacher is not None:
                teacher.unload()
            continue

        try:
            per_split: dict[str, np.ndarray] = {}
            for split, csv_path in split_csvs.items():
                logger.info(
                    "Teacher %s: inferring %s/%s (%s)",
                    teacher_name,
                    args.generation,
                    split,
                    csv_path,
                )
                probs = _run_teacher_inference(
                    teacher,
                    csv_path,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                )
                out_path = output_dir / split / f"{teacher_name}.npy"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(out_path, probs)
                logger.info(
                    "Saved %s (N=%d, mean=%.4f)",
                    out_path,
                    probs.shape[0],
                    float(probs.mean()) if probs.size else float("nan"),
                )
                per_split[split] = probs
            per_teacher_soft[teacher_name] = per_split
        finally:
            teacher.unload()

    if not per_teacher_soft:
        logger.error(
            "No teacher produced soft labels successfully. "
            "Check that checkpoints exist and dependencies are installed."
        )
        return 1

    # --- Weight teachers by val AUC ------------------------------------
    val_labels = _load_val_labels(split_csvs["val"])
    per_teacher_auc = {
        name: compute_val_auc(splits["val"], val_labels)
        for name, splits in per_teacher_soft.items()
    }
    weights = softmax_weights(
        per_teacher_auc, temperature=_WEIGHTING_TEMPERATURE
    )
    logger.info("Per-teacher val AUC: %s", {k: round(v, 4) for k, v in per_teacher_auc.items()})
    logger.info("Ensemble weights:   %s", {k: round(v, 4) for k, v in weights.items()})

    # --- Aggregate ensemble per split ----------------------------------
    num_samples_per_split: dict[str, int] = {}
    for split in _SPLIT_NAMES:
        per_teacher_split = {
            name: per_teacher_soft[name][split] for name in per_teacher_soft
        }
        ensemble = aggregate(per_teacher_split, weights)
        out_path = output_dir / split / "ensemble.npy"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, ensemble)
        num_samples_per_split[split] = int(ensemble.shape[0])
        logger.info(
            "Saved ensemble %s (N=%d, mean=%.4f)",
            out_path,
            ensemble.shape[0],
            float(ensemble.mean()) if ensemble.size else float("nan"),
        )

    # --- Manifest ------------------------------------------------------
    manifest = {
        "generation": args.generation,
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "teachers_used": list(per_teacher_soft.keys()),
        "weights": weights,
        "val_auc": per_teacher_auc,
        "weighting_temperature": _WEIGHTING_TEMPERATURE,
        "num_samples_per_split": num_samples_per_split,
    }
    manifest_path = output_dir / "ensemble_weights.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Wrote manifest: %s", manifest_path)
    logger.info("Done. Soft labels under %s", output_dir)
    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run teacher ensemble inference to generate soft labels"
    )
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument(
        "--generation",
        required=True,
        choices=["gen1", "gen2", "gen3"],
        help="Which generation to process",
    )
    p.add_argument(
        "--teachers",
        default=None,
        help=(
            "Comma-separated teacher names. Default: "
            "efficientnet_b4,recce,clip_vit_l14"
        ),
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument(
        "--device",
        default=None,
        help="Torch device. Default: cuda if available, else cpu",
    )
    p.add_argument(
        "--splits-dir",
        default=None,
        help="Override splits dir. Default: {drive}/datasets/splits",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Override output dir. Default: {drive}/soft_labels/{generation}",
    )
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(_run(_parse_args()))
