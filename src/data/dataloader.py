"""DataLoader factory with optional replay-buffer mixing.

Creates standard DataLoaders for initial distillation and mixed
DataLoaders (new generation + replay buffer) for continual distillation.

Three flavors are exposed:

* :func:`build_dataloader` — vanilla single-CSV loader.
* :func:`build_replay_dataloader` — mixes new + replay via
  :class:`WeightedRandomSampler` over a plain ``ConcatDataset``. Used by
  ``ReplayStrategy`` and ``CombinedReplayEWCStrategy``.
* :func:`build_replay_dataloader_indexed` — same as above but uses
  :class:`IndexedConcatDataset` and a custom collate that injects
  ``(stored_logits, mask)`` into each batch. Used by
  :class:`~src.training.anti_forgetting.der_plus_plus.DERReplayStrategy`,
  which needs to know which rows in the batch are replay samples (mask
  bit set) so the loss can apply MSE-on-stored-logits only to those rows.

Called by: src/training/trainer.py, src/training/continual_trainer.py,
           scripts/03-06, tests/test_dataset.py
Data files: none directly. Wraps :class:`DeepfakeDataset` instances,
           which read CSV and .npy files declared in
           ``src/data/dataset.py``.
"""
from __future__ import annotations

from bisect import bisect_right
from pathlib import Path
from typing import Any

import numpy as np
import torch
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


# --------------------------------------------------------------------------- #
#  DER++ support: indexed concat dataset + collate that injects stored logits
# --------------------------------------------------------------------------- #
class IndexedConcatDataset(ConcatDataset):
    """A :class:`ConcatDataset` that injects ``_sub_dataset_idx`` and
    ``_local_idx`` into each item's ``meta`` dict.

    Used by :func:`build_replay_dataloader_indexed` so the collate function
    can identify which rows of a mixed batch came from the replay buffer
    (and therefore have stored previous-gen logits) vs the new generation.

    Each underlying ``DeepfakeDataset.__getitem__`` returns a 4-tuple
    ``(image, hard_label, soft_label, meta)``. We pass that through
    unchanged except ``meta`` gets two extra integer keys:

    * ``_sub_dataset_idx``: index into ``self.datasets`` (0 = first dataset
      passed to constructor, 1 = second, ...). For DER++ we put new-gen at
      index 0 and replay at index 1.
    * ``_local_idx``: row index *within* the source dataset (0..N-1 where
      N = ``len(self.datasets[_sub_dataset_idx])``). Used to look up the
      matching stored-logits row in a separate aligned numpy array.
    """

    def __getitem__(self, idx: int) -> tuple[Any, Any, Any, dict[str, Any]]:
        if idx < 0:
            idx = len(self) + idx
        if idx < 0 or idx >= len(self):
            raise IndexError(f"IndexedConcatDataset index out of range: {idx}")
        sub_idx = bisect_right(self.cumulative_sizes, idx)
        local_idx = (
            idx if sub_idx == 0 else idx - self.cumulative_sizes[sub_idx - 1]
        )
        item = self.datasets[sub_idx][local_idx]
        # DeepfakeDataset returns (image, hard, soft, meta) — preserve shape
        if not (isinstance(item, tuple) and len(item) == 4):
            raise TypeError(
                "IndexedConcatDataset expects sub-datasets to return a "
                f"4-tuple (image, hard, soft, meta); got {type(item)}"
            )
        image, hard, soft, meta = item
        meta = dict(meta) if isinstance(meta, dict) else {"_meta_raw": meta}
        meta["_sub_dataset_idx"] = int(sub_idx)
        meta["_local_idx"] = int(local_idx)
        return image, hard, soft, meta


def build_replay_dataloader_indexed(
    new_csv_path: str | Path,
    replay_csv_path: str | Path,
    batch_size: int,
    *,
    new_soft_label_path: str | Path | None = None,
    replay_soft_label_path: str | Path | None = None,
    replay_logits_path: str | Path | None = None,
    image_size: int = 224,
    aug_cfg: dict[str, Any] | None = None,
    num_workers: int = 2,
    pin_memory: bool = True,
    replay_ratio: float = 0.5,
) -> DataLoader:
    """Build a DataLoader for DER++: mixed new + replay batches with
    per-row ``stored_logits`` and ``mask`` tensors appended to each batch.

    Each yielded batch is a 6-tuple::

        (images, hard_labels, soft_labels, metas,
         stored_logits, stored_logits_mask)

    where:

    * ``stored_logits`` has shape ``(B, num_classes)`` — float32, with
      replay-row entries filled from the previous-gen stored-logits NPY,
      and new-row entries set to zero (placeholder).
    * ``stored_logits_mask`` has shape ``(B,)`` — bool, True for replay
      rows, False for new rows. The continual loss uses this mask to
      apply the MSE-on-logits term only to replay samples.

    If ``replay_logits_path`` is ``None``, the function returns the
    plain replay loader (4-tuple batches) — equivalent to
    :func:`build_replay_dataloader` but using :class:`IndexedConcatDataset`
    so callers can still inspect sub-dataset origin via ``meta``.

    Args:
        new_csv_path: New-generation training CSV.
        replay_csv_path: Replay buffer CSV (selected exemplars).
        batch_size: Mini-batch size.
        new_soft_label_path: Optional ``.npy`` for new-gen soft labels.
        replay_soft_label_path: Optional ``.npy`` for replay soft labels.
        replay_logits_path: Optional ``.npy`` of stored previous-gen
            student logits, shape ``(N_replay, num_classes)``. Required
            for DER++.
        replay_ratio: Fraction of each batch (in expectation) that comes
            from the replay buffer.
    """
    if not 0.0 < replay_ratio < 1.0:
        raise ValueError(f"replay_ratio must be in (0, 1), got {replay_ratio}")

    transform = build_transforms(
        mode="train", image_size=image_size, aug_cfg=aug_cfg,
    )
    new_ds = DeepfakeDataset(
        new_csv_path, transform=transform,
        soft_label_path=new_soft_label_path, mode="train",
    )
    replay_ds = DeepfakeDataset(
        replay_csv_path, transform=transform,
        soft_label_path=replay_soft_label_path, mode="train",
    )
    # Order matters — collate uses _sub_dataset_idx (0 = new, 1 = replay).
    combined = IndexedConcatDataset([new_ds, replay_ds])

    weights = _make_mixing_weights(
        n_new=len(new_ds), n_replay=len(replay_ds), replay_ratio=replay_ratio,
    )
    sampler = WeightedRandomSampler(
        weights=weights, num_samples=len(combined), replacement=True,
    )

    # Lazy-load stored logits (kept in driver memory, not per-worker).
    # Stored logits are small: e.g. 30k samples × 2 classes × 4 bytes ~ 0.25 MB.
    stored_logits: np.ndarray | None = None
    num_classes = 2
    if replay_logits_path is not None:
        sl_path = Path(replay_logits_path)
        if not sl_path.is_file():
            raise FileNotFoundError(
                f"DER++ stored logits not found: {sl_path}"
            )
        stored_logits = np.load(sl_path).astype(np.float32, copy=False)
        if stored_logits.ndim != 2:
            raise ValueError(
                f"Expected stored logits to have shape (N, C); got {stored_logits.shape}"
            )
        if stored_logits.shape[0] != len(replay_ds):
            raise ValueError(
                f"Stored logits length {stored_logits.shape[0]} does not match "
                f"replay buffer length {len(replay_ds)}"
            )
        num_classes = int(stored_logits.shape[1])

    def _collate(batch: list[tuple[Any, int, float, dict[str, Any]]]) -> tuple[Any, ...]:
        images = torch.stack([b[0] for b in batch], dim=0)
        hards = torch.tensor([int(b[1]) for b in batch], dtype=torch.long)
        softs = torch.tensor([float(b[2]) for b in batch], dtype=torch.float32)
        metas = [b[3] for b in batch]

        if stored_logits is None:
            return images, hards, softs, metas

        sl_batch = torch.zeros(len(batch), num_classes, dtype=torch.float32)
        mask = torch.zeros(len(batch), dtype=torch.bool)
        for i, m in enumerate(metas):
            if m.get("_sub_dataset_idx") == 1:  # replay sub-dataset
                local = int(m.get("_local_idx", -1))
                if 0 <= local < stored_logits.shape[0]:
                    sl_batch[i] = torch.from_numpy(stored_logits[local])
                    mask[i] = True
        return images, hards, softs, metas, sl_batch, mask

    return DataLoader(
        combined,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=_collate,
        persistent_workers=num_workers > 0,
    )
