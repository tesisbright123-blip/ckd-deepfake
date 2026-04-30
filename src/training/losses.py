"""Loss functions for knowledge distillation and continual learning.

Two public classes:

* :class:`DistillationLoss` — initial (single-generation) distillation.
      L = alpha * L_KD(T^2) + (1 - alpha) * L_CE

* :class:`ContinualDistillationLoss` — continual (later-generation) training.
      L = alpha * L_KD + beta * L_retention + gamma * L_CE
      where ``L_retention`` is a KD term against the previous student's logits
      (supplied by LwF / replay orchestrators).

Both use temperature-scaled KL divergence. Soft labels arrive as per-sample
``P(fake)`` floats in ``[0, 1]`` (the ensemble output from scripts/03); internally
they are expanded to ``(B, 2)`` probability distributions before the KD step.

The sentinel ``-1.0`` (imported from ``src.data.dataset.SOFT_LABEL_MISSING``)
marks "no soft label available for this sample" — rows carrying it are masked
out of the KD term so that hard-label-only batches still train cleanly.

Called by:
    src/training/trainer.py
    src/training/continual_trainer.py
Reads / Writes: none.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.dataset import SOFT_LABEL_MISSING

# Small epsilon used when converting P(fake) scalars into (real, fake) distributions.
_EPS = 1.0e-7


def soft_labels_to_distribution(
    soft_fake_prob: torch.Tensor,
    num_classes: int = 2,
) -> torch.Tensor:
    """Turn a ``(B,)`` ``P(fake)`` tensor into a ``(B, num_classes)`` distribution.

    Only ``num_classes == 2`` is supported (index 1 = fake). Values are clamped
    to ``[EPS, 1-EPS]`` to avoid ``log(0)`` downstream.
    """
    if num_classes != 2:
        raise ValueError(
            f"soft_labels_to_distribution: only num_classes=2 supported, got {num_classes}"
        )
    p_fake = soft_fake_prob.clamp(min=_EPS, max=1.0 - _EPS)
    return torch.stack((1.0 - p_fake, p_fake), dim=1)


def _kd_kl(
    student_logits: torch.Tensor,
    teacher_prob: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Temperature-scaled KL(teacher || student) averaged over the batch.

    Matches the classical Hinton formulation: the student's log-softmax is
    computed at temperature ``T``, the teacher distribution is treated as the
    target, and the result is scaled by ``T**2``.
    """
    log_p_student = F.log_softmax(student_logits / temperature, dim=1)
    log_p_teacher = teacher_prob.clamp(min=_EPS).log()
    # ``F.kl_div`` expects log-probabilities as input and probabilities as target.
    kl = F.kl_div(log_p_student, teacher_prob, reduction="batchmean", log_target=False)
    return kl * (temperature ** 2)


class DistillationLoss(nn.Module):
    """Initial distillation: alpha * KD + (1 - alpha) * CE.

    Args:
        alpha: Weight on the KD term. Must be in ``[0, 1]``.
        temperature: Softmax temperature for both student and teacher.
    """

    def __init__(self, alpha: float = 0.7, temperature: float = 4.0) -> None:
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.alpha = float(alpha)
        self.temperature = float(temperature)
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        student_logits: torch.Tensor,
        hard_labels: torch.Tensor,
        soft_fake_prob: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Combined loss + per-component diagnostics.

        Args:
            student_logits: ``(B, 2)`` raw logits.
            hard_labels: ``(B,)`` int64 class labels.
            soft_fake_prob: ``(B,)`` float32 teacher ensemble ``P(fake)``; rows
                equal to :data:`~src.data.dataset.SOFT_LABEL_MISSING` are masked
                out of the KD term.

        Returns:
            ``(loss, metrics)`` where ``metrics`` has float entries
            ``{"loss", "loss_kd", "loss_ce", "kd_coverage"}``.
        """
        ce_loss = self.ce(student_logits, hard_labels.long())

        has_soft = soft_fake_prob != SOFT_LABEL_MISSING
        coverage = float(has_soft.float().mean().item())

        if has_soft.any() and self.alpha > 0.0:
            teacher_prob = soft_labels_to_distribution(soft_fake_prob[has_soft])
            kd_loss = _kd_kl(
                student_logits[has_soft], teacher_prob, self.temperature
            )
        else:
            kd_loss = student_logits.new_zeros(())

        total = self.alpha * kd_loss + (1.0 - self.alpha) * ce_loss
        metrics = {
            "loss": float(total.detach().item()),
            "loss_kd": float(kd_loss.detach().item()),
            "loss_ce": float(ce_loss.detach().item()),
            "kd_coverage": coverage,
        }
        return total, metrics


class ContinualDistillationLoss(nn.Module):
    """Continual distillation: alpha * KD + beta * L_retention + gamma * CE.

    ``L_retention`` is another temperature-scaled KL divergence, but the target
    distribution is the *previous* student's output (passed in as logits by the
    LwF / replay orchestrator in :mod:`src.training.continual_trainer`).

    Effective-weight renormalisation
    --------------------------------
    When the active anti-forgetting strategy does not provide previous-student
    logits (EWC, plain Replay) the retention term is zero on every batch. Naively
    keeping ``beta`` non-zero in that case would mean the loss carries only
    ``alpha + gamma`` of total weight and EWC/Replay implicitly receive a 30%
    smaller update step than LwF — making the cross-method ablation unfair.

    To keep methods apples-to-apples we **detect at construction time** whether
    a retention-providing strategy will be used (via ``has_retention``) and
    renormalise ``alpha`` and ``gamma`` to sum to 1 when retention is off:

        if not has_retention:
            alpha_eff = alpha / (alpha + gamma)
            gamma_eff = gamma / (alpha + gamma)
            beta_eff  = 0.0

    so EWC/Replay gets the same total loss magnitude as LwF.

    Args:
        alpha: Weight on the teacher KD term.
        beta:  Weight on the previous-student retention term.
        gamma: Weight on the hard-label CE term.
        temperature: Softmax temperature for the teacher KD term.
        retention_temperature: Separate temperature for the retention term
            (LwF, Li & Hoiem 2017, used T=2). If ``None`` the same temperature
            as the teacher KD term is used.
        has_retention: Whether the active strategy will supply
            ``previous_student_logits`` on every batch. When ``False`` (e.g.
            EWC, plain Replay) ``alpha`` and ``gamma`` are renormalised so the
            two-term loss carries unit total weight.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.2,
        temperature: float = 4.0,
        retention_temperature: float | None = None,
        has_retention: bool = True,
    ) -> None:
        super().__init__()
        for name, value in (("alpha", alpha), ("beta", beta), ("gamma", gamma)):
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if retention_temperature is not None and retention_temperature <= 0:
            raise ValueError(
                f"retention_temperature must be > 0, got {retention_temperature}"
            )

        if has_retention:
            self.alpha = float(alpha)
            self.beta = float(beta)
            self.gamma = float(gamma)
        else:
            denom = float(alpha + gamma)
            if denom <= 0:
                raise ValueError(
                    "When has_retention=False, alpha + gamma must be > 0 "
                    f"(got alpha={alpha}, gamma={gamma})"
                )
            self.alpha = float(alpha) / denom
            self.beta = 0.0
            self.gamma = float(gamma) / denom

        self.temperature = float(temperature)
        self.retention_temperature = (
            float(retention_temperature)
            if retention_temperature is not None
            else float(temperature)
        )
        self.has_retention = bool(has_retention)
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        student_logits: torch.Tensor,
        hard_labels: torch.Tensor,
        soft_fake_prob: torch.Tensor,
        previous_student_logits: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Combined loss + per-component diagnostics.

        Args:
            student_logits: ``(B, 2)`` logits from the current student.
            hard_labels: ``(B,)`` int64 labels.
            soft_fake_prob: ``(B,)`` teacher ensemble ``P(fake)`` (or sentinel).
            previous_student_logits: ``(B, 2)`` logits from the frozen previous
                student, or ``None`` if retention is disabled for this batch.
        """
        ce_loss = self.ce(student_logits, hard_labels.long())

        has_soft = soft_fake_prob != SOFT_LABEL_MISSING
        coverage = float(has_soft.float().mean().item())
        if has_soft.any() and self.alpha > 0.0:
            teacher_prob = soft_labels_to_distribution(soft_fake_prob[has_soft])
            kd_loss = _kd_kl(
                student_logits[has_soft], teacher_prob, self.temperature
            )
        else:
            kd_loss = student_logits.new_zeros(())

        if previous_student_logits is not None and self.beta > 0.0:
            prev_prob = F.softmax(
                previous_student_logits.detach() / self.retention_temperature,
                dim=1,
            )
            retention_loss = _kd_kl(
                student_logits, prev_prob, self.retention_temperature
            )
        else:
            retention_loss = student_logits.new_zeros(())

        total = (
            self.alpha * kd_loss
            + self.beta * retention_loss
            + self.gamma * ce_loss
        )
        metrics = {
            "loss": float(total.detach().item()),
            "loss_kd": float(kd_loss.detach().item()),
            "loss_retention": float(retention_loss.detach().item()),
            "loss_ce": float(ce_loss.detach().item()),
            "kd_coverage": coverage,
        }
        return total, metrics
