"""DER++ (Dark Experience Replay++) anti-forgetting strategy.

Reference: Buzzega, Boschini, Porrello, Abati, Calderara — "Dark Experience
for General Continual Learning: a Strong, Simple Baseline." NeurIPS 2020.

Extends :class:`~src.training.anti_forgetting.replay.ReplayStrategy` by
storing the **logits** the previous-generation student produced on the
selected exemplars at the time the buffer was built. During replay, the
loss adds an MSE term

    L_DER = MSE(student_current(x_replay), stored_logits(x_replay))

on top of the standard CE + KD-from-teacher pipeline. This forces the
student to remain functionally close to its previous-gen incarnation on
exemplars from the prior generation, providing a stronger retention
signal than plain replay (which only enforces output match on hard labels
and the *current* teacher's soft labels).

Key implementation choices:

* Inherit from ``ReplayStrategy`` so we get herding selection +
  buffer CSV / soft-label NPY for free.
* In ``before_training()``, after the parent runs, load the
  previous-generation student from ``previous_checkpoint``, run the
  buffered exemplars through it once, and persist the resulting logits
  to ``replay_<gen>_logits.npy`` next to the existing buffer files.
* ``augment_dataloader()`` returns a
  :func:`~src.data.dataloader.build_replay_dataloader_indexed` that uses
  :class:`~src.data.dataloader.IndexedConcatDataset` and a custom collate
  to inject ``(stored_logits, mask)`` into every batch.
* The MSE term itself is computed inside
  :class:`~src.training.losses.ContinualDistillationLoss` (with
  ``alpha_der`` weighting) — kept there so all three loss components
  (KD / retention / DER) live in one place and share the renormalisation
  logic when ``has_retention=False``.
* ``provides_retention_logits = False`` because DER++ does NOT produce
  per-batch frozen-model logits the way LwF does (the stored logits are
  pre-computed once and looked up by row index, not by re-running a
  frozen model on the current images).

Called by:
    scripts/05_continual_distillation.py (when --method = "der++")
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataloader import build_replay_dataloader_indexed
from src.training.anti_forgetting.replay import (
    ReplayStrategy,
    _loader_csv_path,
    _loader_soft_path,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DERReplayStrategy(ReplayStrategy):
    """DER++ — stores previous-gen student logits for the replay buffer
    and adds an MSE-on-logits term during replay.

    Args:
        alpha_der: Weight on the MSE-on-logits term in the continual loss.
            Default 0.5 matches the original DER++ paper's setting.
        **replay_kwargs: Forwarded to :class:`ReplayStrategy`'s constructor
            (``previous_train_csv``, ``buffer_percentage``, etc.).
    """

    name = "der++"
    # DER++ does NOT supply per-batch logits via previous_logits() — the
    # logits are pre-computed once and read from disk via the dataloader's
    # collate fn. So provides_retention_logits = False, and the loss will
    # renormalise alpha+gamma+alpha_der to sum 1 (renormalisation in
    # ContinualDistillationLoss.__init__ handles this case).
    provides_retention_logits = False
    retention_temperature: float | None = None

    def __init__(self, *, alpha_der: float = 0.5, **replay_kwargs: Any) -> None:
        super().__init__(**replay_kwargs)
        if alpha_der < 0:
            raise ValueError(f"alpha_der must be >= 0, got {alpha_der}")
        self.alpha_der = float(alpha_der)
        self._buffer_logits_path: Path | None = None

    # ------------------------------------------------------------------ #
    #  Lifecycle
    # ------------------------------------------------------------------ #
    def before_training(
        self,
        model: nn.Module,
        *,
        previous_checkpoint: str | Path | None = None,
        previous_val_loader: DataLoader | None = None,
        device: str | torch.device = "cuda",
    ) -> None:
        # 1) Run base replay logic (build CSV + soft labels)
        super().before_training(
            model,
            previous_checkpoint=previous_checkpoint,
            previous_val_loader=previous_val_loader,
            device=device,
        )

        if self._buffer_csv is None:
            logger.warning(
                "DER++: buffer was not built (super().before_training did not "
                "set _buffer_csv); skipping logits extraction. Replay-only "
                "fallback will be used."
            )
            return

        # 2) Load PREVIOUS-generation student to extract logits
        if previous_checkpoint is None:
            logger.warning(
                "DER++: no previous_checkpoint supplied — falling back to "
                "vanilla replay (no stored logits, no MSE term)."
            )
            return

        prev_ckpt = Path(previous_checkpoint)
        if not prev_ckpt.is_file():
            logger.warning(
                "DER++: previous_checkpoint not found at %s — falling back "
                "to vanilla replay.", prev_ckpt,
            )
            return

        # Lazy-import to avoid pulling timm/torch into module import graph
        # for non-DER++ runs.
        from src.models.students.mobilenetv3 import build_student
        from src.utils.checkpoint import load_checkpoint

        prev_student = build_student(pretrained=False)
        load_checkpoint(prev_ckpt, model=prev_student, strict=False)
        prev_student.eval()

        # 3) Extract logits for every row in the buffer CSV (already sliced
        # to the herding-selected exemplars). Order is preserved.
        buffer_df = pd.read_csv(self._buffer_csv)
        logger.info(
            "DER++: extracting %d logits from previous-gen student %s",
            len(buffer_df), prev_ckpt,
        )
        try:
            stored = self._extract_stored_logits(
                buffer_df,
                student_model=prev_student,
                device=device,
            )
        finally:
            # Free GPU memory immediately — the previous student was only
            # needed to compute these logits once.
            del prev_student
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 4) Save logits next to the buffer CSV / soft-labels NPY
        logits_path = (
            self.buffer_output_dir
            / f"replay_{self.previous_generation}_logits.npy"
        )
        import numpy as np  # local — only needed in this branch
        np.save(logits_path, stored)
        self._buffer_logits_path = logits_path
        logger.info(
            "DER++: stored logits saved to %s (shape=%s, alpha_der=%.3f)",
            logits_path, stored.shape, self.alpha_der,
        )

    # ------------------------------------------------------------------ #
    #  Per-batch hooks
    # ------------------------------------------------------------------ #
    def augment_dataloader(self, new_loader: DataLoader) -> DataLoader:
        if self._buffer_csv is None:
            logger.warning(
                "DER++: augment_dataloader called before buffer built; "
                "passing through new_loader unchanged."
            )
            return new_loader

        new_csv = _loader_csv_path(new_loader)
        new_soft = _loader_soft_path(new_loader)
        if new_csv is None:
            raise RuntimeError(
                "DER++: could not recover the new-generation CSV path from "
                "the provided loader. Pass a DataLoader built via "
                "build_dataloader()."
            )

        return build_replay_dataloader_indexed(
            new_csv_path=new_csv,
            replay_csv_path=self._buffer_csv,
            batch_size=self.batch_size,
            new_soft_label_path=new_soft,
            replay_soft_label_path=self._buffer_soft_path,
            replay_logits_path=self._buffer_logits_path,
            image_size=self.image_size,
            aug_cfg=self.aug_cfg,
            num_workers=self.num_workers,
            replay_ratio=self.replay_ratio,
        )

    # ------------------------------------------------------------------ #
    #  Serialization
    # ------------------------------------------------------------------ #
    def state_dict(self) -> dict[str, Any]:
        s = super().state_dict()
        s["alpha_der"] = self.alpha_der
        s["buffer_logits_path"] = (
            str(self._buffer_logits_path) if self._buffer_logits_path else None
        )
        return s
