"""Learning without Forgetting (LwF) anti-forgetting strategy.

Snapshots the previous student *right before* the continual training round
starts, freezes it, and supplies its logits on the current batch as the target
distribution for a KD term inside
:class:`~src.training.losses.ContinualDistillationLoss`:

    L_retention = T**2 * KL( softmax(logits_prev / T) || softmax(logits_new / T) )

The trainer calls :meth:`previous_logits` once per batch and forwards the
returned tensor to the loss. The frozen model stays on the same device as the
active model so there's no per-batch device ping-pong.

Reference: Li & Hoiem, "Learning without Forgetting" (ECCV 2016).

Called by:
    src/training/continual_trainer.py
    scripts/05_continual_distillation.py
Reads:
    Previous checkpoint ``.pth`` (via :mod:`src.utils.checkpoint`) used to
    populate the frozen model.
Writes: none — the snapshot lives in memory and is rebuilt each run.
"""
from __future__ import annotations

import copy
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.anti_forgetting.base import AntiForgettingStrategy
from src.utils.checkpoint import load_checkpoint
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LwFStrategy(AntiForgettingStrategy):
    """LwF with a frozen snapshot of the previous student.

    Args:
        temperature: Softmax temperature used by the retention KD term in
            :class:`~src.training.losses.ContinualDistillationLoss`. Default
            ``2.0`` follows Li & Hoiem 2017. The trainer reads this via the
            :attr:`retention_temperature` class slot and forwards it to the
            loss as its ``retention_temperature`` argument, so the LwF KD
            uses τ=2 even when the main teacher KD uses τ=4.
    """

    name = "lwf"
    provides_retention_logits = True

    def __init__(self, temperature: float = 2.0) -> None:
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.temperature = float(temperature)
        self.retention_temperature = float(temperature)
        self._frozen_model: nn.Module | None = None

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
        frozen = copy.deepcopy(model)
        if previous_checkpoint is not None:
            load_checkpoint(previous_checkpoint, model=frozen, map_location=device)
            logger.info("LwF: loaded frozen snapshot from %s", previous_checkpoint)
        else:
            logger.info(
                "LwF: no previous_checkpoint provided — using current model weights as snapshot"
            )

        for param in frozen.parameters():
            param.requires_grad_(False)
        frozen.eval()
        frozen.to(device)
        self._frozen_model = frozen

    def after_training(self, model: nn.Module) -> None:
        # Freeing the frozen model releases its share of GPU memory.
        self._frozen_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    #  Retention hook
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def previous_logits(
        self,
        model: nn.Module,
        images: torch.Tensor,
    ) -> torch.Tensor | None:
        if self._frozen_model is None:
            return None
        frozen_device = next(self._frozen_model.parameters()).device
        return self._frozen_model(images.to(frozen_device, non_blocking=True))

    # ------------------------------------------------------------------ #
    #  Serialization
    # ------------------------------------------------------------------ #
    def state_dict(self) -> dict[str, object]:
        return {
            "temperature": self.temperature,
            "frozen_model_state": (
                self._frozen_model.state_dict()
                if self._frozen_model is not None
                else None
            ),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.temperature = float(state.get("temperature", self.temperature))
        # Caller must have already instantiated ``_frozen_model`` via
        # before_training before calling this; here we just restore weights.
        frozen_state = state.get("frozen_model_state")
        if frozen_state is not None and self._frozen_model is not None:
            self._frozen_model.load_state_dict(frozen_state)  # type: ignore[arg-type]
