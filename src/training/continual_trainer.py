"""Continual-distillation training loop (Gen 2 onwards).

Takes the student that finished Gen N, a new-generation loader, and an
:class:`~src.training.anti_forgetting.base.AntiForgettingStrategy` instance,
and does an abbreviated KD run that tries not to catastrophically forget
earlier generations.

Loss breakdown (mirrors ``training.continual_distillation`` in the YAML):

    L = alpha * L_KD_teacher
      + beta  * L_retention          (LwF KD against frozen prev-student logits)
      + gamma * L_CE
      + L_penalty                    (EWC quadratic term, if any)

Anti-forgetting strategies plug in via their hooks:

* ``strategy.augment_dataloader(new_loader)`` — Replay swaps in a mixed loader.
* ``strategy.previous_logits(model, images)`` — LwF returns frozen logits.
* ``strategy.penalty(model)`` — EWC returns the Fisher regularizer scalar.

Called by:
    scripts/05_continual_distillation.py
Reads: DataLoaders from :mod:`src.data.dataloader`, previous student
    checkpoint ``.pth`` (for LwF snapshot), strategy-specific inputs.
Writes:
    ``{checkpoint_dir}/best.pth`` — best val AUC on the new generation.
    ``{checkpoint_dir}/last.pth`` — latest epoch.
    Checkpoints carry the standard schema defined in
    :mod:`src.utils.checkpoint` (``timestamp`` ISO 8601 ``%Y-%m-%dT%H:%M:%S``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.evaluation.evaluator import evaluate_loader
from src.evaluation.metrics import BinaryMetrics
from src.training.anti_forgetting.base import AntiForgettingStrategy
from src.training.losses import ContinualDistillationLoss
from src.utils.checkpoint import save_checkpoint
from src.utils.logger import get_logger


@dataclass(frozen=True)
class ContinualTrainerConfig:
    """Hyperparameters for :class:`ContinualTrainer`.

    Defaults mirror ``training.continual_distillation`` in ``configs/default.yaml``.
    """

    learning_rate: float = 5.0e-5
    weight_decay: float = 1.0e-4
    num_epochs: int = 10
    alpha: float = 0.5
    beta: float = 0.3
    gamma: float = 0.2
    temperature: float = 4.0
    early_stopping_patience: int = 3
    grad_clip_norm: float | None = 1.0
    log_every_n_steps: int = 50


@dataclass
class ContinualEpochStats:
    """Per-epoch rollup for continual training — includes retention diagnostics."""

    epoch: int
    train_loss: float
    train_kd: float
    train_retention: float
    train_ce: float
    train_penalty: float
    val_auc: float
    val_log_loss: float
    val_accuracy: float
    lr: float
    kd_coverage: float = 0.0
    extra: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float]:
        out = {
            "epoch": int(self.epoch),
            "train_loss": float(self.train_loss),
            "train_kd": float(self.train_kd),
            "train_retention": float(self.train_retention),
            "train_ce": float(self.train_ce),
            "train_penalty": float(self.train_penalty),
            "val_auc": float(self.val_auc),
            "val_log_loss": float(self.val_log_loss),
            "val_accuracy": float(self.val_accuracy),
            "lr": float(self.lr),
            "kd_coverage": float(self.kd_coverage),
        }
        out.update(self.extra)
        return out


class ContinualTrainer:
    """Anti-forgetting-aware trainer for Gen N > 1."""

    def __init__(
        self,
        *,
        model: nn.Module,
        new_train_loader: DataLoader,
        new_val_loader: DataLoader,
        strategy: AntiForgettingStrategy,
        config: ContinualTrainerConfig,
        device: str | torch.device = "cuda",
        checkpoint_dir: str | Path | None = None,
        generation: str = "gen2",
        run_config: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.new_train_loader = new_train_loader
        self.new_val_loader = new_val_loader
        self.strategy = strategy
        self.config = config
        self.device = torch.device(device)
        self.generation = generation
        self.run_config = run_config or {}

        self.checkpoint_dir = (
            Path(checkpoint_dir) if checkpoint_dir is not None else None
        )
        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.loss_fn = ContinualDistillationLoss(
            alpha=config.alpha,
            beta=config.beta,
            gamma=config.gamma,
            temperature=config.temperature,
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=max(config.num_epochs, 1)
        )

        self.logger = get_logger(f"continual_trainer.{generation}.{strategy.name}")
        self.best_val_auc: float = 0.0
        self.history: list[ContinualEpochStats] = []

    # ------------------------------------------------------------------ #
    #  Public entrypoint
    # ------------------------------------------------------------------ #
    def fit(self) -> list[ContinualEpochStats]:
        """Run the continual-distillation training loop."""
        self.model.to(self.device)
        # Let the strategy rewrite the train loader (Replay mixes in buffer).
        train_loader = self.strategy.augment_dataloader(self.new_train_loader)

        patience = self.config.early_stopping_patience
        epochs_without_improvement = 0

        for epoch in range(self.config.num_epochs):
            train_metrics = self._train_one_epoch(epoch, train_loader)
            val_metrics = evaluate_loader(
                self.model, self.new_val_loader, device=self.device
            )
            self.scheduler.step()

            stats = ContinualEpochStats(
                epoch=epoch,
                train_loss=train_metrics["loss"],
                train_kd=train_metrics["loss_kd"],
                train_retention=train_metrics["loss_retention"],
                train_ce=train_metrics["loss_ce"],
                train_penalty=train_metrics["loss_penalty"],
                val_auc=val_metrics.auc,
                val_log_loss=val_metrics.log_loss,
                val_accuracy=val_metrics.accuracy,
                kd_coverage=train_metrics["kd_coverage"],
                lr=self._current_lr(),
            )
            self.history.append(stats)

            self.logger.info(
                "epoch=%d train_loss=%.4f (kd=%.4f ret=%.4f ce=%.4f pen=%.4f) "
                "val_auc=%.4f val_ll=%.4f val_acc=%.4f lr=%.2e",
                stats.epoch,
                stats.train_loss,
                stats.train_kd,
                stats.train_retention,
                stats.train_ce,
                stats.train_penalty,
                stats.val_auc,
                stats.val_log_loss,
                stats.val_accuracy,
                stats.lr,
            )

            if val_metrics.auc > self.best_val_auc:
                self.best_val_auc = val_metrics.auc
                epochs_without_improvement = 0
                self._save("best.pth", epoch=epoch, metrics=val_metrics)
            else:
                epochs_without_improvement += 1

            self._save("last.pth", epoch=epoch, metrics=val_metrics)

            if patience > 0 and epochs_without_improvement >= patience:
                self.logger.info(
                    "Early stopping at epoch %d (patience=%d, best_auc=%.4f)",
                    epoch,
                    patience,
                    self.best_val_auc,
                )
                break

        # Let strategies free GPU memory (e.g. LwF's frozen model).
        self.strategy.after_training(self.model)
        return self.history

    # ------------------------------------------------------------------ #
    #  Internals
    # ------------------------------------------------------------------ #
    def _train_one_epoch(
        self, epoch: int, loader: DataLoader
    ) -> dict[str, float]:
        self.model.train()
        running = {
            "loss": 0.0,
            "loss_kd": 0.0,
            "loss_retention": 0.0,
            "loss_ce": 0.0,
            "loss_penalty": 0.0,
            "kd_coverage": 0.0,
        }
        n_batches = 0

        for step, batch in enumerate(loader):
            images, hard_labels, soft_labels, _meta = batch
            images = images.to(self.device, non_blocking=True)
            hard_labels = hard_labels.to(self.device, non_blocking=True)
            soft_labels = soft_labels.to(self.device, non_blocking=True).float()

            prev_logits = self.strategy.previous_logits(self.model, images)
            logits = self.model(images)

            core_loss, metrics = self.loss_fn(
                logits, hard_labels, soft_labels, prev_logits
            )
            penalty = self.strategy.penalty(self.model)
            loss = core_loss + penalty

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.config.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip_norm
                )
            self.optimizer.step()

            running["loss"] += float(loss.detach().item())
            running["loss_kd"] += metrics["loss_kd"]
            running["loss_retention"] += metrics["loss_retention"]
            running["loss_ce"] += metrics["loss_ce"]
            running["loss_penalty"] += float(penalty.detach().item())
            running["kd_coverage"] += metrics["kd_coverage"]
            n_batches += 1

            if (step + 1) % self.config.log_every_n_steps == 0:
                self.logger.info(
                    "epoch=%d step=%d loss=%.4f kd=%.4f ret=%.4f ce=%.4f pen=%.4f cov=%.2f",
                    epoch,
                    step + 1,
                    float(loss.detach().item()),
                    metrics["loss_kd"],
                    metrics["loss_retention"],
                    metrics["loss_ce"],
                    float(penalty.detach().item()),
                    metrics["kd_coverage"],
                )

        if n_batches == 0:
            raise RuntimeError("Empty continual-train loader — nothing was iterated.")
        return {k: v / n_batches for k, v in running.items()}

    def _current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _save(
        self,
        filename: str,
        *,
        epoch: int,
        metrics: BinaryMetrics,
    ) -> Path | None:
        if self.checkpoint_dir is None:
            return None
        return save_checkpoint(
            self.checkpoint_dir / filename,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=epoch,
            best_val_auc=self.best_val_auc,
            config={
                **self.run_config,
                "strategy_state": self.strategy.state_dict(),
            },
            generation=self.generation,
            metrics={
                "val": metrics.as_dict(),
                "history": [s.as_dict() for s in self.history],
            },
        )
