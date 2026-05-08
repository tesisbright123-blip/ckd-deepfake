"""Replay-buffer anti-forgetting strategy with herding selection.

Holds a small fraction (default: 5%) of the previous generation's train split,
chosen per-class via **herding** — the same greedy scheme used in iCaRL. The
buffer is a tiny CSV on disk plus an optional soft-label ``.npy``; at training
time it's spliced into the new-generation loader via
:func:`src.data.dataloader.build_replay_dataloader`.

Herding (per class):
    1. Compute feature vectors for every candidate via ``model.extract_features``.
    2. Compute the class mean ``mu_c``.
    3. Greedily pick the exemplar whose running-sum average is closest to
       ``mu_c``; repeat until the quota is filled.

Reference: Rebuffi et al., "iCaRL: Incremental Classifier and Representation
Learning" (CVPR 2017).

Called by:
    src/training/continual_trainer.py
    scripts/05_continual_distillation.py
Reads:
    Previous generation's train split CSV
    (``{drive}/datasets/splits/{previous}_train.csv``).
    Previous generation's ensemble soft labels
    (``{drive}/soft_labels/{previous}/train/ensemble.npy``), if available.
    Face JPEGs referenced by ``face_path``.
Writes:
    Buffer CSV at ``{output_dir}/replay_{previous}.csv``
        columns mirror :data:`src.data.splits.REQUIRED_COLUMNS` with the
        standard schema ``(face_path, frame_idx, video_id, label, dataset,
        generation, technique)``.
    Buffer soft labels at ``{output_dir}/replay_{previous}_soft.npy`` — an
    ``(N_buffer,) float32`` array aligned to the buffer CSV row order.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.augmentations import build_transforms
from src.data.dataloader import build_replay_dataloader
from src.data.splits import REQUIRED_COLUMNS
from src.training.anti_forgetting.base import AntiForgettingStrategy
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ReplayStrategy(AntiForgettingStrategy):
    """Herding-based replay buffer. See module docstring for the algorithm."""

    name = "replay"

    def __init__(
        self,
        *,
        previous_train_csv: str | Path,
        previous_soft_label_path: str | Path | None,
        buffer_output_dir: str | Path,
        previous_generation: str,
        buffer_percentage: float = 0.05,
        selection: str = "herding",
        replay_ratio: float = 0.5,
        batch_size: int = 64,
        num_workers: int = 2,
        image_size: int = 224,
        aug_cfg: dict | None = None,
        herding_candidate_cap: int | None = 4000,
    ) -> None:
        if not 0.0 < buffer_percentage < 1.0:
            raise ValueError(
                f"buffer_percentage must be in (0, 1), got {buffer_percentage}"
            )
        if selection not in {"herding", "random"}:
            raise ValueError(
                f"Unknown selection strategy: {selection!r} (expected 'herding' or 'random')"
            )

        self.previous_train_csv = Path(previous_train_csv)
        self.previous_soft_label_path = (
            Path(previous_soft_label_path)
            if previous_soft_label_path is not None
            else None
        )
        self.buffer_output_dir = Path(buffer_output_dir)
        self.previous_generation = previous_generation
        self.buffer_percentage = float(buffer_percentage)
        self.selection = selection
        self.replay_ratio = float(replay_ratio)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.image_size = int(image_size)
        self.aug_cfg = aug_cfg or {}
        self.herding_candidate_cap = herding_candidate_cap

        self._buffer_csv: Path | None = None
        self._buffer_soft_path: Path | None = None
        self._buffer_size: int = 0

    # ------------------------------------------------------------------ #
    #  Setup
    # ------------------------------------------------------------------ #
    def before_training(
        self,
        model: nn.Module,
        *,
        previous_checkpoint: str | Path | None = None,
        previous_val_loader: DataLoader | None = None,
        device: str | torch.device = "cuda",
    ) -> None:
        if not self.previous_train_csv.is_file():
            raise FileNotFoundError(
                f"Replay: previous train CSV not found: {self.previous_train_csv}"
            )
        self.buffer_output_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(self.previous_train_csv)
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Replay: previous CSV missing columns {missing} ({self.previous_train_csv})"
            )

        n_target = max(1, int(round(self.buffer_percentage * len(df))))
        logger.info(
            "Replay: selecting %d / %d previous-gen samples (%.1f%%, method=%s)",
            n_target,
            len(df),
            100.0 * self.buffer_percentage,
            self.selection,
        )

        soft_labels = self._load_soft_labels(len(df))
        if self.selection == "random":
            selected_indices = self._select_random(df, n_target)
        else:
            selected_indices = self._select_herding(df, model, device, n_target)

        self._write_buffer(df, soft_labels, selected_indices)

    def augment_dataloader(self, new_loader: DataLoader) -> DataLoader:
        if self._buffer_csv is None:
            logger.warning(
                "Replay: augment_dataloader called before before_training; passthrough."
            )
            return new_loader

        new_csv = _loader_csv_path(new_loader)
        new_soft = _loader_soft_path(new_loader)
        if new_csv is None:
            raise RuntimeError(
                "Replay: could not recover the new-generation CSV path from the "
                "provided loader. Pass a DataLoader built via build_dataloader()."
            )

        return build_replay_dataloader(
            new_csv_path=new_csv,
            replay_csv_path=self._buffer_csv,
            batch_size=self.batch_size,
            new_soft_label_path=new_soft,
            replay_soft_label_path=self._buffer_soft_path,
            image_size=self.image_size,
            aug_cfg=self.aug_cfg,
            num_workers=self.num_workers,
            replay_ratio=self.replay_ratio,
        )

    # ------------------------------------------------------------------ #
    #  Selection algorithms
    # ------------------------------------------------------------------ #
    def _load_soft_labels(self, expected_len: int) -> np.ndarray | None:
        if self.previous_soft_label_path is None:
            return None
        if not self.previous_soft_label_path.is_file():
            logger.warning(
                "Replay: previous soft labels not found at %s; buffer will be hard-label only.",
                self.previous_soft_label_path,
            )
            return None
        arr = np.load(self.previous_soft_label_path)
        if arr.shape != (expected_len,):
            raise ValueError(
                f"Replay: soft-label shape {arr.shape} does not match CSV length {expected_len}"
            )
        return arr.astype(np.float32, copy=False)

    def _select_random(self, df: pd.DataFrame, n_target: int) -> np.ndarray:
        rng = np.random.default_rng(0)
        # Stratify a bit: pick proportionally per class to avoid all-real buffers.
        picks: list[int] = []
        for label_value, bucket in df.groupby("label"):
            n_cls = max(1, int(round(n_target * len(bucket) / len(df))))
            idx = rng.choice(bucket.index.to_numpy(), size=min(n_cls, len(bucket)), replace=False)
            picks.extend(idx.tolist())
        return np.asarray(picks, dtype=np.int64)[:n_target]

    def _select_herding(
        self,
        df: pd.DataFrame,
        model: nn.Module,
        device: str | torch.device,
        n_target: int,
    ) -> np.ndarray:
        if not hasattr(model, "extract_features"):
            raise AttributeError(
                "Replay.herding requires ``model.extract_features(x) -> (B, D)``; "
                "MobileNetV3Student provides this."
            )

        picks: list[int] = []
        for label_value, bucket in df.groupby("label"):
            candidate_indices = bucket.index.to_numpy()
            n_cls = max(1, int(round(n_target * len(bucket) / len(df))))

            if self.herding_candidate_cap is not None and len(candidate_indices) > self.herding_candidate_cap:
                rng = np.random.default_rng(int(label_value))
                candidate_indices = rng.choice(
                    candidate_indices, size=self.herding_candidate_cap, replace=False
                )

            features = self._extract_features(
                df.loc[candidate_indices, "face_path"].tolist(), model, device
            )
            class_mean = features.mean(axis=0)
            class_mean /= max(np.linalg.norm(class_mean), 1e-8)

            selected_local = _greedy_herding(features, class_mean, k=min(n_cls, len(features)))
            picks.extend(int(candidate_indices[i]) for i in selected_local)

        return np.asarray(picks, dtype=np.int64)[:n_target]

    def _extract_features(
        self,
        face_paths: Sequence[str],
        model: nn.Module,
        device: str | torch.device,
    ) -> np.ndarray:
        transform = build_transforms(mode="val", image_size=self.image_size)
        model.eval()
        model.to(device)

        features: list[np.ndarray] = []
        batch: list[torch.Tensor] = []

        def _flush() -> None:
            if not batch:
                return
            tensor = torch.stack(batch, dim=0).to(device, non_blocking=True)
            with torch.inference_mode():
                feats = model.extract_features(tensor)
            feats_np = feats.detach().float().cpu().numpy()
            norms = np.linalg.norm(feats_np, axis=1, keepdims=True).clip(min=1e-8)
            features.append(feats_np / norms)
            batch.clear()

        for path in face_paths:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Replay: failed to read {path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            batch.append(transform(image=img)["image"])
            if len(batch) >= self.batch_size:
                _flush()
        _flush()

        if not features:
            raise RuntimeError("Replay: no features extracted")
        return np.concatenate(features, axis=0)

    # ------------------------------------------------------------------ #
    #  Logit extraction (used by DER++ subclass)
    # ------------------------------------------------------------------ #
    def _extract_stored_logits(
        self,
        df_selected: pd.DataFrame,
        *,
        student_model: nn.Module,
        device: str | torch.device,
    ) -> np.ndarray:
        """Forward selected exemplars through ``student_model`` (the
        previous-generation student, frozen) and return ``(N, num_classes)``
        logit array aligned to ``df_selected`` row order.

        Used by :class:`~src.training.anti_forgetting.der_plus_plus.DERReplayStrategy`
        to capture the logits the previous student produced on its own
        exemplars, which DER++ later forces the current student to match
        via MSE.
        """
        transform = build_transforms(mode="val", image_size=self.image_size)
        student_model.eval()
        student_model.to(device)

        chunks: list[np.ndarray] = []
        batch: list[torch.Tensor] = []

        @torch.inference_mode()
        def _flush() -> None:
            nonlocal batch
            if not batch:
                return
            x = torch.stack(batch, dim=0).to(device, non_blocking=True)
            out = student_model(x)
            chunks.append(out.detach().float().cpu().numpy())
            batch = []

        for path in df_selected["face_path"].tolist():
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"DER++: failed to read {path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            batch.append(transform(image=img)["image"])
            if len(batch) >= self.batch_size:
                _flush()
        _flush()

        if not chunks:
            raise RuntimeError("DER++: no logits extracted from buffer")
        return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)

    # ------------------------------------------------------------------ #
    #  Buffer serialization
    # ------------------------------------------------------------------ #
    def _write_buffer(
        self,
        df: pd.DataFrame,
        soft_labels: np.ndarray | None,
        selected: np.ndarray,
    ) -> None:
        buffer_df = df.loc[selected].reset_index(drop=True)
        csv_path = (
            self.buffer_output_dir / f"replay_{self.previous_generation}.csv"
        )
        buffer_df.to_csv(csv_path, index=False)
        self._buffer_csv = csv_path
        self._buffer_size = len(buffer_df)

        if soft_labels is not None:
            soft_slice = soft_labels[selected].astype(np.float32, copy=False)
            soft_path = (
                self.buffer_output_dir / f"replay_{self.previous_generation}_soft.npy"
            )
            np.save(soft_path, soft_slice)
            self._buffer_soft_path = soft_path

        logger.info(
            "Replay: wrote buffer (n=%d) to %s%s",
            self._buffer_size,
            csv_path,
            f" (+ soft: {self._buffer_soft_path})" if self._buffer_soft_path else "",
        )

    # ------------------------------------------------------------------ #
    #  Serialization
    # ------------------------------------------------------------------ #
    def state_dict(self) -> dict[str, object]:
        return {
            "previous_generation": self.previous_generation,
            "buffer_csv": str(self._buffer_csv) if self._buffer_csv else None,
            "buffer_soft_path": (
                str(self._buffer_soft_path) if self._buffer_soft_path else None
            ),
            "buffer_size": self._buffer_size,
            "selection": self.selection,
            "replay_ratio": self.replay_ratio,
        }


# ---------------------------------------------------------------------- #
#  Helpers
# ---------------------------------------------------------------------- #
def _greedy_herding(
    features: np.ndarray,
    class_mean: np.ndarray,
    *,
    k: int,
) -> list[int]:
    """Return ``k`` indices into ``features`` whose running mean tracks ``class_mean``.

    ``features`` is assumed to be L2-normalized (unit vectors).
    """
    n = features.shape[0]
    k = min(k, n)
    selected: list[int] = []
    running_sum = np.zeros_like(class_mean)
    available = np.ones(n, dtype=bool)

    for i in range(k):
        candidate_mean = (features + running_sum) / (i + 1)
        # We want to minimize || class_mean - candidate_mean ||.
        dist = np.linalg.norm(candidate_mean - class_mean, axis=1)
        dist[~available] = np.inf
        best = int(np.argmin(dist))
        selected.append(best)
        running_sum = running_sum + features[best]
        available[best] = False
    return selected


def _loader_csv_path(loader: DataLoader) -> str | Path | None:
    dataset = getattr(loader, "dataset", None)
    csv_path = getattr(dataset, "csv_path", None)
    return csv_path


def _loader_soft_path(loader: DataLoader) -> str | Path | None:
    """Best-effort recovery of the soft-label path from a DeepfakeDataset."""
    dataset = getattr(loader, "dataset", None)
    # DeepfakeDataset stores the loaded array (``soft_labels``) but not the
    # source path. Callers that need strict soft-label forwarding should pass
    # the path explicitly; we return None to signal "no file-backed path".
    if dataset is None or getattr(dataset, "soft_labels", None) is None:
        return None
    return None
