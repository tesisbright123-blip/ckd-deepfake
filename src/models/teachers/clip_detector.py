"""CLIPping the Deception teacher loader.

Wraps CLIP ViT-L/14 with an adaptation head (fine-tuned on deepfake data by
the CLIPping-the-Deception paper) into the :class:`BaseTeacher` interface.

The backbone is open_clip's official ``ViT-L-14`` visual tower; only the
adaptation head weights are expected to live in the checkpoint at
``{drive}/checkpoints/teachers/clip_clipping.pth``. If the checkpoint also
contains visual-tower weights, they are loaded non-strictly.

IMPORTANT: CLIP uses a DIFFERENT normalization from the other teachers
(``CLIP_MEAN`` / ``CLIP_STD`` in ``base.py`` — do NOT reuse ImageNet stats).

Called by:
    src/models/teachers/ensemble.py
    scripts/03_generate_soft_labels.py
Reads:
    Weight file (default: {drive}/checkpoints/teachers/clip_clipping.pth)
Preprocessing: LongestMaxSize -> PadIfNeeded(224) -> CLIP Normalize -> ToTensor.
Output: per-image ``P(fake)`` as ``np.ndarray`` shape ``(B,)``, dtype ``float32``.
"""
from __future__ import annotations

from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2

from src.models.teachers.base import (
    CLIP_MEAN,
    CLIP_STD,
    BaseTeacher,
    safe_load_state_dict,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# open_clip's ViT-L/14 visual-tower output dimension.
_CLIP_VIT_L14_EMBED_DIM = 768


class _CLIPVisualWithHead(nn.Module):
    """Visual tower + binary/softmax adaptation head."""

    def __init__(
        self,
        visual: nn.Module,
        embed_dim: int,
        num_outputs: int,
    ) -> None:
        super().__init__()
        self.visual = visual
        self.head = nn.Linear(embed_dim, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.visual(x)
        # open_clip ``visual(x)`` returns either (features,) or a plain tensor.
        if isinstance(features, (tuple, list)):
            features = features[0]
        return self.head(features)


class CLIPDetectorTeacher(BaseTeacher):
    """CLIP ViT-L/14 backbone + adaptation head (CLIPping-the-Deception).

    The number of head outputs is auto-detected from the checkpoint's
    ``head.weight`` shape on load. A 1-output head is treated as a single
    BCE logit; a 2-output head is treated as (real, fake) softmax.
    """

    name = "clip_vit_l14"
    input_size = 224

    def __init__(
        self,
        device: str | torch.device = "cuda",
        pretrained_tag: str = "openai",
    ) -> None:
        super().__init__(device=device)
        self.pretrained_tag = pretrained_tag
        self._num_outputs: int = 1  # updated in ``load`` once head shape is known.

    # ------------------------------------------------------------------ #
    #  Loading
    # ------------------------------------------------------------------ #

    def load(self, weight_path: str | Path) -> None:
        try:
            import open_clip
        except ImportError as exc:
            raise ModuleNotFoundError(
                "open_clip_torch is required for CLIPDetectorTeacher. "
                "Install with: pip install open_clip_torch"
            ) from exc

        # Build visual tower from open_clip's registry.
        clip_model, _, _ = open_clip.create_model_and_transforms(
            "ViT-L-14",
            pretrained=self.pretrained_tag,
        )
        visual = clip_model.visual  # ViT-L/14 image tower.

        # Peek at the checkpoint to figure out head output size.
        state_dict = safe_load_state_dict(weight_path)
        num_outputs = self._infer_head_outputs(state_dict)
        self._num_outputs = num_outputs

        model = _CLIPVisualWithHead(
            visual=visual,
            embed_dim=_CLIP_VIT_L14_EMBED_DIM,
            num_outputs=num_outputs,
        )

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(
                "%s: %d missing keys when loading %s (first 3: %s)",
                self.name,
                len(missing),
                weight_path,
                missing[:3],
            )
        if unexpected:
            logger.warning(
                "%s: %d unexpected keys when loading %s (first 3: %s)",
                self.name,
                len(unexpected),
                weight_path,
                unexpected[:3],
            )

        model.eval()
        model.to(self.device)
        self.model = model
        logger.info(
            "%s: loaded weights from %s (head_outputs=%d)",
            self.name,
            weight_path,
            num_outputs,
        )

    @staticmethod
    def _infer_head_outputs(state_dict: dict[str, torch.Tensor]) -> int:
        """Inspect checkpoint for ``head.weight`` and return its row count."""
        for candidate in ("head.weight", "classifier.weight", "fc.weight"):
            if candidate in state_dict and state_dict[candidate].ndim == 2:
                return int(state_dict[candidate].shape[0])
        # Default to 1-logit BCE head if the checkpoint has no classifier layer.
        logger.warning(
            "CLIPDetectorTeacher: no head weight found in checkpoint; "
            "defaulting to single-logit BCE head."
        )
        return 1

    # ------------------------------------------------------------------ #
    #  Inference
    # ------------------------------------------------------------------ #

    @torch.inference_mode()
    def predict_proba(self, images: torch.Tensor) -> np.ndarray:
        model = self._assert_loaded()
        images = images.to(self.device, non_blocking=True)
        logits = model(images)

        if self._num_outputs == 1:
            # BCE: sigmoid of the single logit is P(fake).
            probs = torch.sigmoid(logits.squeeze(-1))
        elif self._num_outputs == 2:
            probs = F.softmax(logits, dim=1)[:, 1]
        else:
            raise RuntimeError(
                f"{self.name}: unsupported head output dim {self._num_outputs}"
            )
        return probs.detach().float().cpu().numpy()

    def get_preprocessing(self) -> A.Compose:
        return A.Compose(
            [
                A.Resize(height=self.input_size, width=self.input_size),
                A.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
                ToTensorV2(),
            ]
        )
