"""DataLoader factory with optional replay-buffer mixing.

Creates standard DataLoaders for initial distillation and mixed
DataLoaders (new generation + replay buffer) for continual distillation.

Called by: src/training/trainer.py, src/training/continual_trainer.py,
           scripts/03-06, tests/test_dataset.py
Data files: none directly. Wraps :class:`DeepfakeDataset` instances,
           which read CSV and .npy files declared in
           ``src/data/dataset.py``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

from src.data.augmentations import build_transforms
from src.data.dataset import DeepfakeDataset


def build_dataloader(
    csv_path: str | Path,
    mode: str,
    batch_size: int,
    *,
    soft_label_path: str | Path | None = None,
    image_size: int = 224,
    aug_cfg: dict[str, Any] | None = None,
    num_workers: int = 2,
    pin_memory: bool = True,
    shuffle: bool | None = None,
    drop_last: bool | None = None,
) -> DataLoader:
    """Build a DataLoader for a single split.

    ``shuffle`` / ``drop_last`` default to sensible values per mode:
    - train: shuffle=True,  drop_last=True
    - val/test: shuffle=False, drop_last=False
    """
    transform = build_transforms(mode=mode, image_size=image_size, aug_cfg=aug_cfg)
    dataset = DeepfakeDataset(
        csv_path=csv_path,
        transform=transform,
        soft_label_path=soft_label_path,
        mode=mode,
    )

    if shuffle is None:
        shuffle = mode == "train"
    if drop_last is None:
        drop_last = mode == "train"

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )


def build_replay_dataloader(
    new_csv_path: str | Path,
    replay_csv_path: str | Path,
    batch_size: int,
    *,
    new_soft_label_path: str | Path | None = None,
    replay_soft_label_path: str | Path | None = None,
    image_size: int = 224,
    aug_cfg: dict[str, Any] | None = None,
    num_workers: int = 2,
    pin_memory: bool = True,
    replay_ratio: float = 0.5,
) -> DataLoader:
    """Mix a new generation with a replay buffer into one training loader.

    Each mini-batch is drawn so that roughly ``replay_ratio`` of the
    samples come from the replay buffer — implemented via a
    :class:`WeightedRandomSampler` on top of a ``ConcatDataset``.

    Args:
        new_csv_path: CSV for the new generation (to be learned).
        replay_csv_path: CSV for exemplars kept from previous generations.
        replay_ratio: Fraction of each batch (in expectation) that should
            come from the replay set. Must be in (0, 1).
    """
    if not 0.0 < replay_ratio < 1.0:
        raise ValueError(f"replay_ratio must be in (0, 1), got {replay_ratio}")

    transform = build_transforms(
        mode="train", image_size=image_size, aug_cfg=aug_cfg
    )
    new_ds = DeepfakeDataset(
        new_csv_path, transform=transform,
        soft_label_path=new_soft_label_path, mode="train",
    )
    replay_ds = DeepfakeDataset(
        replay_csv_path, transform=transform,
        soft_label_path=replay_soft_label_path, mode="train",
    )

    combined: Dataset = ConcatDataset([new_ds, replay_ds])
    weights = _make_mixing_weights(
        n_new=len(new_ds),
        n_replay=len(replay_ds),
        replay_ratio=replay_ratio,
    )
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(combined),
        replacement=True,
    )

    return DataLoader(
        combined,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )


def _make_mixing_weights(
    n_new: int, n_replay: int, replay_ratio: float
) -> list[float]:
    """Per-sample weights that yield ``replay_ratio`` in expectation."""
    if n_new == 0 or n_replay == 0:
        raise ValueError(
            f"Both datasets must be non-empty (new={n_new}, replay={n_replay})"
        )
    w_new = (1.0 - replay_ratio) / n_new
    w_replay = replay_ratio / n_replay
    return [w_new] * n_new + [w_replay] * n_replay
