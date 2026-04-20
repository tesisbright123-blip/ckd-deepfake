"""Elastic Weight Consolidation (EWC) anti-forgetting strategy.

Computes the diagonal Fisher Information Matrix on the previous generation's
validation set right before the continual-training run starts, then penalizes
changes to parameters the model considered "important":

    L_EWC = (lambda / 2) * sum_i F_i * (theta_i - theta_star_i) ** 2

``theta_star`` is the parameter snapshot at the start of the new round;
``F_i`` is the per-parameter Fisher estimate. Only trainable float parameters
are tracked — BatchNorm buffers, int buffers, and frozen weights are skipped.

Reference: Kirkpatrick et al., "Overcoming catastrophic forgetting in neural
networks" (PNAS 2017).

Called by:
    src/training/continual_trainer.py
    scripts/05_continual_distillation.py
Reads:
    Previous-generation val loader (for Fisher estimation).
Writes: none — Fisher/theta_star live in CPU tensors on this object and are
    optionally serialised via :meth:`state_dict` into the training checkpoint.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.training.anti_forgetting.base import AntiForgettingStrategy
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EWCStrategy(AntiForgettingStrategy):
    """EWC with diagonal Fisher matrix estimated on the previous-gen val loader.

    Args:
        lambda_: Strength of the quadratic penalty. Typical range 10–5000.
        fisher_samples: Maximum number of samples to iterate through when
            estimating the Fisher matrix. ``None`` means "use the whole
            loader".
    """

    name = "ewc"

    def __init__(
        self,
        lambda_: float = 5000.0,
        fisher_samples: int | None = 1000,
    ) -> None:
        if lambda_ < 0:
            raise ValueError(f"lambda_ must be >= 0, got {lambda_}")
        if fisher_samples is not None and fisher_samples <= 0:
            raise ValueError(
                f"fisher_samples must be positive or None, got {fisher_samples}"
            )
        self.lambda_ = float(lambda_)
        self.fisher_samples = fisher_samples

        # Populated by ``before_training``. Keys are parameter names.
        self._fisher: dict[str, torch.Tensor] = {}
        self._theta_star: dict[str, torch.Tensor] = {}

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
        if previous_val_loader is None:
            raise ValueError(
                "EWCStrategy requires previous_val_loader to estimate the Fisher matrix"
            )
        model.to(device)
        model.eval()

        fisher = {
            name: torch.zeros_like(p, device=device)
            for name, p in model.named_parameters()
            if p.requires_grad and p.is_floating_point()
        }

        seen = 0
        for batch in previous_val_loader:
            images, hard_labels, *_ = batch
            images = images.to(device, non_blocking=True)
            hard_labels = hard_labels.to(device, non_blocking=True).long()

            model.zero_grad(set_to_none=True)
            logits = model(images)
            # Use log-likelihood of the true label so E[grad^2] approximates Fisher.
            log_probs = F.log_softmax(logits, dim=1)
            nll = F.nll_loss(log_probs, hard_labels, reduction="sum")
            nll.backward()

            batch_size = images.size(0)
            for name, param in model.named_parameters():
                if name in fisher and param.grad is not None:
                    fisher[name] += param.grad.detach() ** 2
            seen += batch_size

            if self.fisher_samples is not None and seen >= self.fisher_samples:
                break

        if seen == 0:
            raise RuntimeError("EWC Fisher estimation saw zero samples")

        # Normalize by the number of samples, then move to CPU to save VRAM.
        self._fisher = {name: (f / float(seen)).detach().cpu() for name, f in fisher.items()}
        self._theta_star = {
            name: p.detach().clone().cpu()
            for name, p in model.named_parameters()
            if name in self._fisher
        }
        model.zero_grad(set_to_none=True)
        logger.info(
            "EWC: Fisher estimated over %d samples across %d parameter tensors (lambda=%g)",
            seen,
            len(self._fisher),
            self.lambda_,
        )

    # ------------------------------------------------------------------ #
    #  Penalty term
    # ------------------------------------------------------------------ #
    def penalty(self, model: nn.Module) -> torch.Tensor:
        device = next(model.parameters()).device
        if not self._fisher or self.lambda_ == 0.0:
            return torch.zeros((), device=device)

        total = torch.zeros((), device=device)
        named = dict(model.named_parameters())
        for name, fisher_cpu in self._fisher.items():
            if name not in named:
                continue  # parameter dropped between checkpoints
            param = named[name]
            theta_star = self._theta_star[name].to(device)
            fisher = fisher_cpu.to(device)
            total = total + (fisher * (param - theta_star) ** 2).sum()
        return 0.5 * self.lambda_ * total

    # ------------------------------------------------------------------ #
    #  Serialization
    # ------------------------------------------------------------------ #
    def state_dict(self) -> dict[str, object]:
        return {
            "lambda_": self.lambda_,
            "fisher_samples": self.fisher_samples,
            "fisher": {k: v.clone() for k, v in self._fisher.items()},
            "theta_star": {k: v.clone() for k, v in self._theta_star.items()},
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.lambda_ = float(state.get("lambda_", self.lambda_))
        self.fisher_samples = state.get("fisher_samples", self.fisher_samples)  # type: ignore[assignment]
        self._fisher = {
            k: v.detach().clone()
            for k, v in (state.get("fisher") or {}).items()  # type: ignore[union-attr]
        }
        self._theta_star = {
            k: v.detach().clone()
            for k, v in (state.get("theta_star") or {}).items()  # type: ignore[union-attr]
        }
