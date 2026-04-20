"""Abstract interface for anti-forgetting mechanisms.

Three concrete strategies plug into continual distillation:

* :class:`~src.training.anti_forgetting.ewc.EWCStrategy`
* :class:`~src.training.anti_forgetting.lwf.LwFStrategy`
* :class:`~src.training.anti_forgetting.replay.ReplayStrategy`

They all subclass :class:`AntiForgettingStrategy` below. The base class is
deliberately stateless beyond the hooks it defines; each concrete strategy
owns whatever state it needs (Fisher matrix, frozen model, exemplar buffer).

Typical lifecycle inside :class:`~src.training.continual_trainer.ContinualTrainer`::

    strategy.before_training(model, previous_checkpoint_path, prev_val_loader)
    for epoch in range(num_epochs):
        for batch in train_loader:
            ...
            loss_main = criterion(...)
            loss_ret, prev_logits = strategy.retention_loss(model, batch)
            loss = loss_main + loss_ret           # or use ContinualDistillationLoss
            loss.backward()
    strategy.after_training(model)

Called by:
    src/training/continual_trainer.py
    src/training/anti_forgetting/ewc.py (subclass)
    src/training/anti_forgetting/lwf.py (subclass)
    src/training/anti_forgetting/replay.py (subclass)
Reads / Writes: none directly — subclasses may read checkpoints / CSVs.
"""
from __future__ import annotations

from abc import ABC
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class AntiForgettingStrategy(ABC):
    """Base hook interface for continual-learning anti-forgetting methods.

    Every hook has a safe no-op default so simple strategies only need to
    override what they care about:

    * EWC overrides :meth:`before_training` (compute Fisher) and
      :meth:`penalty` (add quadratic term to the loss).
    * LwF overrides :meth:`before_training` (snapshot frozen model) and
      :meth:`previous_logits` (produce target distributions for KD).
    * Replay overrides :meth:`before_training` (build exemplar buffer) and
      :meth:`augment_dataloader` (return a mixed loader at fit time).
    """

    name: str = "base"

    def before_training(
        self,
        model: nn.Module,
        *,
        previous_checkpoint: str | Path | None = None,
        previous_val_loader: DataLoader | None = None,
        device: str | torch.device = "cuda",
    ) -> None:
        """Prepare any state needed before the continual-training run."""
        return None

    def after_training(self, model: nn.Module) -> None:
        """Optional cleanup after the training run."""
        return None

    # ------------------------------------------------------------------ #
    #  Retention hooks — subclasses override the ones they implement.
    # ------------------------------------------------------------------ #
    def penalty(self, model: nn.Module) -> torch.Tensor:
        """Return a scalar regularizer (e.g. the EWC Fisher term).

        Default: zero. Must live on the model's device when non-zero.
        """
        device = next(model.parameters()).device
        return torch.zeros((), device=device)

    def previous_logits(
        self,
        model: nn.Module,
        images: torch.Tensor,
    ) -> torch.Tensor | None:
        """Return the frozen previous student's logits for ``images``.

        Returning ``None`` disables the retention KD term for this batch. LwF
        overrides this to do a forward pass through its snapshot.
        """
        return None

    def augment_dataloader(
        self,
        new_loader: DataLoader,
    ) -> DataLoader:
        """Optionally replace ``new_loader`` with one that mixes in exemplars.

        Default: pass-through. Replay overrides this to splice in its
        buffer loader via :func:`src.data.dataloader.build_replay_dataloader`.
        """
        return new_loader

    # ------------------------------------------------------------------ #
    #  Serialization hooks for checkpoints (optional)
    # ------------------------------------------------------------------ #
    def state_dict(self) -> dict[str, Any]:
        """Return strategy-specific state worth serializing. Default empty."""
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore strategy state from :meth:`state_dict`. Default no-op."""
        return None
