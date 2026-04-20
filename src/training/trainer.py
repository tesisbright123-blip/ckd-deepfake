"""Training loop for initial knowledge distillation (Gen 1).

:class:`DistillationTrainer` wires together:

* a :class:`~src.models.students.mobilenetv3.MobileNetV3Student`
* a :class:`~src.training.losses.DistillationLoss` (alpha·KD + (1-alpha)·CE)
* AdamW + cosine-annealing LR schedule
* val-based early stopping on AUC
* per-epoch + best-checkpoint saving via :mod:`src.utils.checkpoint`

The class is deliberately framework-light — no Lightning — so it can run
equally in Colab and the laptop dev box.

Called by:
    scripts/04_initial_distillation.py
Reads: DataLoader objects built by :mod:`src.data.dataloader` (which in turn
    read split CSVs, face JPEGs, and soft-label ``.npy``).
Writes:
    ``{checkpoint_dir}/best.pth`` — best val AUC checkpoint.
    ``{checkpoint_dir}/last.pth`` — most recent epoch checkpoint.
    Both carry the keys defined in :mod:`src.utils.checkpoint`
    (``timestamp`` in ISO 8601 ``%Y-%m-%dT%H:%M:%S``).
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
from src.training.losses import DistillationLoss
from src.utils.checkpoint import save_checkpoint
from src.utils.logger import get_logger


@dataclass(frozen=True)
class TrainerConfig:
    """Hyperparameters for :class:`DistillationTrainer`.

    Defaults mirror ``training.initial_distillation`` in ``configs/default.yaml``.
    """

    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-4
    num_epochs: int = 30
    alpha: float = 0.7
    temperature: float = 4.0
    early_stopping_patience: int = 5
    grad_clip_norm: float | None = 1.0
    log_every_n_steps: int = 50


@dataclass
class EpochStats:
    """Per-epoch rollup — populated by the trainer and stored in checkpoints."""

    epoch: int
    train_loss: float
    train_kd: float
    train_ce: float
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
            "train_ce": float(self.train_ce),
            "val_auc": float(self.val_auc),
            "val_log_loss": float(self.val_log_loss),
            "val_accuracy": float(self.val_accuracy),
            "lr": float(self.lr),
            "kd_coverage": float(self.kd_coverage),
        }
        out.update(self.extra)
        return out


class DistillationTrainer:
    """Trainer for initial (single-generation) knowledge distillation."""

    def __init__(
        self,
        *,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainerConfig,
        device: str | torch.device = "cuda",
        checkpoint_dir: str | Path | None = None,
        generation: str = "gen1",
        run_config: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device(device)
        self.generation = generation
        self.run_config = run_config or {}

        self.checkpoint_dir = (
            Path(checkpoint_dir) if checkpoint_dir is not None else None
        )
        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.loss_fn = DistillationLoss(
            alpha=config.alpha, temperature=config.temperature
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=max(config.num_epochs, 1)
        )

        self.logger = get_logger(f"trainer.{generation}")
        self.best_val_auc: float = 0.0
        self.history: list[EpochStats] = []

    # ------------------------------------------------------------------ #
    #  Public entrypoint
    # ------------------------------------------------------------------ #
    def fit(self) -> list[EpochStats]:
        """Run the full training loop and return the per-epoch history."""
        self.model.to(self.device)
        patience = self.config.early_stopping_patience
        epochs_without_improvement = 0

        for epoch in range(self.config.num_epochs):
            train_metrics = self._train_one_epoch(epoch)
            val_metrics = evaluate_loader(
                self.model, self.val_loader, device=self.device
            )
            self.scheduler.step()

            stats = EpochStats(
                epoch=epoch,
                train_loss=train_metrics["loss"],
                train_kd=train_metrics["loss_kd"],
                train_ce=train_metrics["loss_ce"],
                val_auc=val_metrics.auc,
                val_log_loss=val_metrics.log_loss,
                val_accuracy=val_metrics.accuracy,
                kd_coverage=train_metrics["kd_coverage"],
                lr=self._current_lr(),
            )
            self.history.append(stats)

            self.logger.info(
                "epoch=%d train_loss=%.4f (kd=%.4f ce=%.4f) val_auc=%.4f "
                "val_ll=%.4f val_acc=%.4f lr=%.2e",
                stats.epoch,
                stats.train_loss,
                stats.train_kd,
                stats.train_ce,
                stats.val_auc,
                stats.val_log_loss,
                stats.val_accuracy,
                stats.lr,
            )

            improved = val_metrics.auc > self.best_val_auc
            if improved:
                self.best_val_auc = val_metrics.auc
                epochs_without_improvement = 0
                self._save("best.pth", epoch=epoch, metrics=val_metrics)
            else:
                epochs_without_improvement += 1

            self._save("last.pth", epoch=epoch, metrics=val_metrics)

            if patience > 0 and epochs_without_improvement >= patience:
                self.logger.info(
                    "Early stopping triggered at epoch %d (patience=%d, best_auc=%.4f)",
                    epoch,
                    patience,
                    self.best_val_auc,
                )
                break

        return self.history

    # ------------------------------------------------------------------ #
    #  Internals
    # ------------------------------------------------------------------ #
    def _train_one_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        n_batches = 0
        running = {"loss": 0.0, "loss_kd": 0.0, "loss_ce": 0.0, "kd_coverage": 0.0}

        for step, batch in enumerate(self.train_loader):
            images, hard_labels, soft_labels, _meta = batch
            images = images.to(self.device, non_blocking=True)
            hard_labels = hard_labels.to(self.device, non_blocking=True)
            soft_labels = soft_labels.to(self.device, non_blocking=True).float()

            logits = self.model(images)
            loss, metrics = self.loss_fn(logits, hard_labels, soft_labels)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.config.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip_norm
                )
            self.optimizer.step()

            for k in running:
                running[k] += metrics[k]
            n_batches += 1

            if (step + 1) % self.config.log_every_n_steps == 0:
                self.logger.info(
                    "epoch=%d step=%d loss=%.4f kd=%.4f ce=%.4f cov=%.2f",
                    epoch,
                    step + 1,
                    metrics["loss"],
                    metrics["loss_kd"],
                    metrics["loss_ce"],
                    metrics["kd_coverage"],
                )

        if n_batches == 0:
            raise RuntimeError(
                "Empty train loader — nothing was iterated during training"
            )
        return {k: v / n_batches for k, v in running.items()}

    def _current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _save(
        self, filename: str, *, epoch: int, metrics: BinaryMetrics
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
            config=self.run_config,
            generation=self.generation,
            metrics={
                "val": metrics.as_dict(),
                "history": [s.as_dict() for s in self.history],
            },
        )
