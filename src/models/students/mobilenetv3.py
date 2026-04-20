"""MobileNetV3-Large student model with a custom 2-class classifier head.

The backbone is ``timm``'s ``mobilenetv3_large_100`` (ImageNet-pretrained by
default), with its original classifier replaced by:

    GAP -> Linear(960, hidden_dim) -> ReLU -> Dropout(p) -> Linear(hidden_dim, 2)

Total params ≈ 5.4M — cheap enough to distill to and ship to edge devices
(TFLite FP16 / INT8 quantization downstream).

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

# timm's ``mobilenetv3_large_100`` feature-extractor output dim.
_BACKBONE_FEATURE_DIM = 960


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
        self.head = self._build_head(
            in_features=_BACKBONE_FEATURE_DIM,
            hidden_dim=self.config.hidden_dim,
            dropout=self.config.dropout,
            num_classes=self.config.num_classes,
        )

    # ------------------------------------------------------------------ #
    #  Construction helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_backbone(*, pretrained: bool) -> nn.Module:
        """Create a MobileNetV3-Large feature extractor (no classifier)."""
        import timm  # lazy import — keeps unit tests that stub the model free of timm.

        # ``num_classes=0`` + ``global_pool="avg"`` returns a (B, 960) feature vector.
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
        """Return the pooled backbone embedding of shape ``(B, 960)``.

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
