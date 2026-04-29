"""Vendored RECCE (Reconstruction-Classification) architecture.

Provides ``build_recce()`` consumed by :class:`src.models.teachers.recce.RECCETeacher`.

RECCE (Cao et al., CVPR 2022, "End-to-End Reconstruction-Classification Learning
for Face Forgery Detection") couples a reconstruction-driven encoder with a
binary classifier head. The full DeepfakeBench reference implementation also
ships a reconstruction decoder, a graph-reasoning module, and a recurrence
block; for *teacher inference only* (which is what this repo uses RECCE for —
generating soft labels) we only need the classification path. We therefore
build a minimal, weight-compatible variant:

    Xception backbone (1x3x224x224 -> features)
       + Global Average Pool
       + Linear head -> (B, 2) logits

Compatibility with DeepfakeBench's ``recce_best.pth``
-----------------------------------------------------
Their checkpoint stores the Xception backbone weights under names like
``backbone.<layer>.weight`` and the classifier under ``head.<...>``. Our
:func:`safe_load_state_dict` strips the ``backbone.`` prefix so the Xception
weights map directly onto timm's keys. The ``head.*`` keys land on our
``Linear(2048, 2)`` head when the shape matches; otherwise the head stays
randomly initialised (a warning is logged in that case so it isn't silent).

If you have the full RECCE ckpt with a Conv2d head (e.g. shape (2, 2048, 1, 1))
this loader is intentionally *forgiving*: it will load every key it can match
and skip the rest, biasing toward "produce some signal" rather than crashing
hard. The val-AUC weighting in
:func:`src.models.teachers.ensemble.softmax_weights` will down-weight RECCE
automatically if the loaded weights underperform.

Called by:
    src/models/teachers/recce.py (RECCETeacher._build_architecture)

Notes
-----
* timm exposes Xception as ``"legacy_xception"``. It outputs a (B, 2048)
  feature vector after global-average pooling.
* This module is intentionally self-contained — no DeepfakeBench import,
  no external download beyond what timm itself does for the (unused, since
  ``pretrained=False``) ImageNet weights.
"""
from __future__ import annotations

import torch
import torch.nn as nn

_XCEPTION_FEATURE_DIM = 2048


class _MinimalRECCE(nn.Module):
    """Xception backbone + 2-class classifier — the inference subset of RECCE."""

    def __init__(self, *, num_classes: int = 2) -> None:
        super().__init__()
        import timm  # lazy import, matches the rest of the teacher loaders.

        # ``num_classes=0`` + ``global_pool="avg"`` -> (B, 2048) feature vector.
        # legacy_xception is the Chollet Xception used by DeepfakeBench's RECCE.
        self.backbone = timm.create_model(
            "legacy_xception",
            pretrained=False,
            num_classes=0,
            global_pool="avg",
        )
        self.head = nn.Linear(_XCEPTION_FEATURE_DIM, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)


def build_recce(*, num_classes: int = 2) -> nn.Module:
    """Factory consumed by :class:`RECCETeacher`."""
    return _MinimalRECCE(num_classes=num_classes)
