"""PyTorch Dataset class for deepfake detection.

Loads face images with hard labels, optional teacher soft labels, and
row-level metadata. Soft labels are supplied as a single ``.npy`` array
of shape ``(N,)`` dtype ``float32`` whose ordering matches the split
CSV row-by-row. If no soft-label file is given, the sentinel ``-1.0``
is returned so downstream code can detect the "hard-label-only" mode.

Called by: src/data/dataloader.py (build_dataloader),
           src/training/trainer.py, src/training/continual_trainer.py,
           tests/test_dataset.py
Reads:
    split CSV (columns: face_path, label, video_id, dataset, generation,
              technique)
    face JPEGs (224x224 BGR on disk)
    optional soft-label .npy (shape (N,), float32, ensemble fake-prob)
Writes: none
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.splits import REQUIRED_COLUMNS

SOFT_LABEL_MISSING = -1.0


class DeepfakeDataset(Dataset):
    """Row-aligned dataset that pairs faces with optional soft labels.

    Args:
        csv_path: Path to a split CSV produced by ``src/data/splits.py``.
        transform: Albumentations Compose (or any callable returning
            ``{"image": tensor}`` when given ``image=ndarray``).
        soft_label_path: Optional ``.npy`` file of shape ``(len(csv),)``
            containing float32 ensemble fake-probabilities aligned to
            CSV row order.
        mode: ``"train" | "val" | "test"`` — forwarded for bookkeeping
            only; behaviour is identical across modes (augmentation is
            controlled by ``transform``).
    """

    def __init__(
        self,
        csv_path: str | Path,
        transform: Any,
        soft_label_path: str | Path | None = None,
        mode: str = "train",
    ) -> None:
        self.csv_path = Path(csv_path)
        self.mode = mode
        if not self.csv_path.is_file():
            raise FileNotFoundError(f"Split CSV not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"{self.csv_path} is missing required columns: {missing}. "
                f"Present: {list(df.columns)}"
            )
        self.df = df.reset_index(drop=True)
        self.transform = transform

        self.soft_labels: np.ndarray | None = None
        if soft_label_path is not None:
            sl_path = Path(soft_label_path)
            if not sl_path.is_file():
                raise FileNotFoundError(f"Soft labels not found: {sl_path}")
            soft = np.load(sl_path)
            if soft.shape != (len(self.df),):
                raise ValueError(
                    f"Soft-label shape {soft.shape} does not match "
                    f"CSV length {len(self.df)} for {sl_path}"
                )
            self.soft_labels = soft.astype(np.float32, copy=False)

    # ------------------------------------------------------------------ #
    #  Dataset protocol
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, int, float, dict[str, Any]]:
        row = self.df.iloc[idx]
        image = self._load_image(row["face_path"])

        transformed = self.transform(image=image)
        image_tensor: torch.Tensor = transformed["image"]

        hard_label = int(row["label"])
        soft_label = (
            float(self.soft_labels[idx])
            if self.soft_labels is not None
            else SOFT_LABEL_MISSING
        )

        meta: dict[str, Any] = {
            "video_id": str(row["video_id"]),
            "dataset": str(row["dataset"]),
            "generation": str(row["generation"]),
            "technique": str(row["technique"]),
            "face_path": str(row["face_path"]),
        }
        return image_tensor, hard_label, soft_label, meta

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_image(path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        # cv2 returns BGR; albumentations' Normalize expects RGB ordering
        # because ImageNet statistics are defined over RGB channels.
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
