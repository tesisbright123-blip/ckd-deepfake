"""MobileNetV3-Large student model with a custom 2-class classifier head.

The backbone is ``timm``'s ``mobilenetv3_large_100`` (ImageNet-pretrained by
default), with its original classifier replaced by:

    GAP -> Linear(D, hidden_dim) -> ReLU -> Dropout(p) -> Linear(hidden_dim, 2)

where ``D`` is auto-detected from the timm backbone via a dummy forward pass
(960 in timm < 1.0, 1280 in timm >= 1.0). Hardcoding either value crashes the
other version; the auto-detect approach makes the student portable across
environments without touching code.

Total params ≈ 4.4M (timm >= 1.0) or 5.4M (timm < 1.0) — cheap enough to
distill to and ship to edge devices (TFLite FP16 / INT8 quantization
downstream).

Called by:
    src/training/trainer.py (initial distillation)
    src/training/continual_trainer.py (continual distillation)
    scripts/04_initial_distillation.py
    scripts/05_continual_distillation.py
    scripts/07_edge_evaluation.py (exports to ONNX/TFLite)
Reads: ImageNet pretrained weights via ``timm`` (no local files).
Writes: none (training loop handles checkpoints).
Input: ``(B, 3, 224, 224)`` float32 (ImageNet-normalized).
Output: ``(B, 2)`` raw logits — index 1 = fake (matches teacher convention).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


def _detect_feature_dim(backbone: nn.Module, *, image_size: int = 224) -> int:
    """Run a 1x3xHxW dummy through ``backbone`` and return output channel count.

    Lets the student adapt automatically to whichever timm version is
    installed (mobilenetv3_large_100 returns 960 in timm<1.0 and 1280 in
    timm>=1.0 with ``num_classes=0`` + ``global_pool="avg"``).
    """
    was_training = backbone.training
    backbone.eval()
    try:
        with torch.inference_mode():
            dummy = torch.zeros(1, 3, image_size, image_size)
            out = backbone(dummy)
    finally:
        backbone.train(was_training)
    return int(out.shape[1])


@dataclass(frozen=True)
class StudentConfig:
    """Hyperparameters for the MobileNetV3 student head."""

    hidden_dim: int = 256
    dropout: float = 0.2
    num_classes: int = 2
    pretrained: bool = True


class MobileNetV3Student(nn.Module):
    """MobileNetV3-Large backbone + custom classification head.

    Args:
        config: :class:`StudentConfig` bundling ``hidden_dim``, ``dropout``,
            ``num_classes``, and ``pretrained`` flag.
    """

    def __init__(self, config: StudentConfig | None = None) -> None:
        super().__init__()
        self.config = config or StudentConfig()
        self.backbone = self._build_backbone(pretrained=self.config.pretrained)
        # Auto-detect feature dim so the head matches whichever timm version
        # is installed (960 vs 1280). This avoids the hardcoded-960 trap that
        # silently crashes on timm>=1.0 with a "mat1 and mat2 shapes cannot be
        # multiplied" error in the head's first Linear layer.
        self.feature_dim = _detect_feature_dim(self.backbone)
        self.head = self._build_head(
            in_features=self.feature_dim,
            hidden_dim=self.config.hidden_dim,
            dropout=self.config.dropout,
            num_classes=self.config.num_classes,
        )

    # ------------------------------------------------------------------ #
    #  Construction helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_backbone(*, pretrained: bool) -> nn.Module:
        """Create a MobileNetV3-Large feature extractor (no classifier).

        Output dim is detected at runtime via :func:`_detect_feature_dim` —
        timm < 1.0 returns 960, timm >= 1.0 returns 1280 for the same
        ``mobilenetv3_large_100`` config.
        """
        import timm  # lazy import — keeps unit tests that stub the model free of timm.

        return timm.create_model(
            "mobilenetv3_large_100",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )

    @staticmethod
    def _build_head(
        *, in_features: int, hidden_dim: int, dropout: float, num_classes: int
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    # ------------------------------------------------------------------ #
    #  Forward / feature extraction
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape ``(B, num_classes)``."""
        features = self.backbone(x)
        return self.head(features)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the pooled backbone embedding of shape ``(B, D)``.

        ``D`` matches whatever the timm backbone outputs for this version
        (960 for timm < 1.0, 1280 for timm >= 1.0). Read it via
        ``self.feature_dim`` if downstream code needs the exact value.

        Useful for replay-buffer herding selection and feature-level
        distillation experiments.
        """
        return self.backbone(x)

    # ------------------------------------------------------------------ #
    #  Convenience
    # ------------------------------------------------------------------ #
    def num_parameters(self, *, trainable_only: bool = False) -> int:
        params = (p for p in self.parameters() if not trainable_only or p.requires_grad)
        return sum(p.numel() for p in params)


def build_student(
    *,
    hidden_dim: int = 256,
    dropout: float = 0.2,
    num_classes: int = 2,
    pretrained: bool = True,
) -> MobileNetV3Student:
    """Factory matching the ``student`` block in ``configs/default.yaml``."""
    return MobileNetV3Student(
        StudentConfig(
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_classes=num_classes,
            pretrained=pretrained,
        )
    )
