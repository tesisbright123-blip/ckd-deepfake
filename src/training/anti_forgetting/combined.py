"""Combined Replay + EWC anti-forgetting strategy.

Wraps :class:`~src.training.anti_forgetting.replay.ReplayStrategy` and
:class:`~src.training.anti_forgetting.ewc.EWCStrategy` so that a single
continual-distillation run gets BOTH:

* **Data-level protection** via Replay — small herding-selected exemplar
  buffer from the previous generation is mixed into every training batch.
* **Weight-level protection** via EWC — Fisher Information Matrix penalty
  pulls important parameters back toward their post-Gen-N-1 values.

These two mechanisms are **complementary**:

* Replay alone keeps the model exposed to old data but allows weights to
  drift if the new-gen loss surface is steep enough.
* EWC alone freezes important weights but can't show the model what those
  weights were *for* without exemplars.

Combined, the strategy stays close to its previous parameters AND keeps
seeing what those parameters were trained on. Empirically (Fontana et al.
2025, Kirkpatrick et al. 2017) this combination beats either alone on
chronological CL benchmarks.

Lifecycle hook orchestration::

    before_training()       -> replay first (build buffer), then EWC (Fisher)
    augment_dataloader()    -> delegate to replay (EWC has no loader hook)
    previous_logits()       -> None (neither sub-strategy provides retention logits)
    penalty()               -> sum of EWC + Replay penalties
    after_training()        -> EWC then replay (cleanup order mirrors construction)
    state_dict / load       -> nested dict {"replay": ..., "ewc": ...}

Loss renormalization: ``provides_retention_logits = False`` so
:class:`~src.training.losses.ContinualDistillationLoss` automatically
renormalises ``alpha + gamma`` to sum 1, keeping gradient magnitudes
comparable across LwF / Replay / EWC / Replay+EWC ablation cells.

Called by:
    scripts/05_continual_distillation.py (when --method = "replay+ewc")
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.anti_forgetting.base import AntiForgettingStrategy
from src.training.anti_forgetting.ewc import EWCStrategy
from src.training.anti_forgetting.replay import ReplayStrategy
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CombinedReplayEWCStrategy(AntiForgettingStrategy):
    """Composite strategy = ReplayStrategy + EWCStrategy applied jointly.

    Args:
        replay: Pre-constructed :class:`ReplayStrategy` instance.
        ewc: Pre-constructed :class:`EWCStrategy` instance.

    Example:
        replay = ReplayStrategy(buffer_percentage=0.10, ...)
        ewc = EWCStrategy(lambda_=5000, fisher_samples=5000)
        strategy = CombinedReplayEWCStrategy(replay=replay, ewc=ewc)
    """

    name = "replay+ewc"
    # Neither sub-strategy provides per-batch frozen-model logits, so the
    # continual loss should drop the LwF retention term and renormalise
    # alpha/gamma to sum 1 (handled in ContinualDistillationLoss).
    provides_retention_logits = False
    retention_temperature: float | None = None

    def __init__(
        self, *, replay: ReplayStrategy, ewc: EWCStrategy,
    ) -> None:
        if not isinstance(replay, ReplayStrategy):
            raise TypeError(
                f"replay must be a ReplayStrategy, got {type(replay).__name__}"
            )
        if not isinstance(ewc, EWCStrategy):
            raise TypeError(
                f"ewc must be an EWCStrategy, got {type(ewc).__name__}"
            )
        self._replay = replay
        self._ewc = ewc

    # ------------------------------------------------------------------ #
    #  Lifecycle hooks
    # ------------------------------------------------------------------ #
    def before_training(
        self,
        model: nn.Module,
        *,
        previous_checkpoint: str | Path | None = None,
        previous_val_loader: DataLoader | None = None,
        device: str | torch.device = "cuda",
    ) -> None:
        # Replay first: builds exemplar buffer (uses model's
        # ``extract_features`` for herding — must run before EWC mutates
        # any state).
        logger.info(
            "%s: setting up replay buffer (previous_generation=%s, buffer_percentage=%.3f)",
            self.name,
            getattr(self._replay, "previous_generation", "?"),
            getattr(self._replay, "buffer_percentage", float("nan")),
        )
        self._replay.before_training(
            model,
            previous_checkpoint=previous_checkpoint,
            previous_val_loader=previous_val_loader,
            device=device,
        )

        # EWC second: snapshots theta* + computes Fisher diagonal on
        # previous_val_loader. Requires a non-None previous_val_loader.
        if previous_val_loader is None:
            raise ValueError(
                f"{self.name} requires previous_val_loader for EWC's Fisher "
                "estimation. Pass --previous to scripts/05 to provide it."
            )
        logger.info(
            "%s: computing EWC Fisher (lambda=%.1f, fisher_samples=%s)",
            self.name,
            getattr(self._ewc, "lambda_", float("nan")),
            getattr(self._ewc, "fisher_samples", "?"),
        )
        self._ewc.before_training(
            model,
            previous_checkpoint=previous_checkpoint,
            previous_val_loader=previous_val_loader,
            device=device,
        )

    def after_training(self, model: nn.Module) -> None:
        # Mirror construction order in reverse for cleanup.
        self._ewc.after_training(model)
        self._replay.after_training(model)

    # ------------------------------------------------------------------ #
    #  Per-batch hooks
    # ------------------------------------------------------------------ #
    def penalty(self, model: nn.Module) -> torch.Tensor:
        # Sum of two scalars, both already on the model's device.
        return self._ewc.penalty(model) + self._replay.penalty(model)

    def previous_logits(
        self, model: nn.Module, images: torch.Tensor,
    ) -> torch.Tensor | None:
        # Neither sub-strategy provides retention logits. Returning None
        # makes ContinualDistillationLoss skip the retention KD term and
        # the trainer use the alpha/gamma-renormalised loss path.
        return None

    def augment_dataloader(self, new_loader: DataLoader) -> DataLoader:
        # Replay handles the WeightedRandomSampler ConcatDataset; EWC has
        # no loader-level transformation.
        return self._replay.augment_dataloader(new_loader)

    # ------------------------------------------------------------------ #
    #  Serialization
    # ------------------------------------------------------------------ #
    def state_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "replay": self._replay.state_dict(),
            "ewc": self._ewc.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        replay_state = state.get("replay", {}) if isinstance(state, dict) else {}
        ewc_state = state.get("ewc", {}) if isinstance(state, dict) else {}
        self._replay.load_state_dict(replay_state)
        self._ewc.load_state_dict(ewc_state)
