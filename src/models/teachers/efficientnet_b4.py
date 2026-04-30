"""DeepfakeBench EfficientNet-B4 teacher loader.

Wraps the pretrained EfficientNet-B4 checkpoint from DeepfakeBench
into the :class:`BaseTeacher` interface so ``scripts/03_generate_soft_labels.py``
and ``src/models/teachers/ensemble.py`` can use it interchangeably with
other teachers.

Called by:
    src/models/teachers/ensemble.py
    scripts/03_generate_soft_labels.py
Reads:
    EfficientNet-B4 weight file (default: {drive}/checkpoints/teachers/efficientnet_b4.pth)
Preprocessing: Albumentations LongestMaxSize -> PadIfNeeded(224) -> ImageNet Normalize -> ToTensor.
Output: per-image ``P(fake)`` as ``np.ndarray`` shape ``(B,)``, dtype ``float32``.
"""
from __future__ import annotations

from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2

from src.models.teachers.base import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    BaseTeacher,
    safe_load_state_dict,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EfficientNetB4Teacher(BaseTeacher):
    """Binary EfficientNet-B4 teacher (DeepfakeBench-trained).

    Args:
        device: Torch device string or object. Defaults to ``"cuda"``.
        num_classes: Output head dim. DeepfakeBench uses 2 (real/fake).
    """

    name = "efficientnet_b4"
    input_size = 224

    def __init__(
        self,
        device: str | torch.device = "cuda",
        num_classes: int = 2,
    ) -> None:
        super().__init__(device=device)
        self.num_classes = num_classes

    # ------------------------------------------------------------------ #
    #  Loading
    # ------------------------------------------------------------------ #

    def load(self, weight_path: str | Path) -> None:
        """Build EfficientNet-B4 via ``timm`` and load DeepfakeBench weights."""
        # Lazy import so unit tests that stub the teacher don't need timm.
        import timm

        model = timm.create_model(
            "efficientnet_b4",
            pretrained=False,
            num_classes=self.num_classes,
        )
        state_dict = safe_load_state_dict(weight_path)
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
        logger.info("%s: loaded weights from %s", self.name, weight_path)

    # ------------------------------------------------------------------ #
    #  Inference
    # ------------------------------------------------------------------ #

    @torch.inference_mode()
    def predict_proba(self, images: torch.Tensor) -> np.ndarray:
        """Return the fake-class probability for each image in the batch."""
        model = self._assert_loaded()
        images = images.to(self.device, non_blocking=True)
        logits = model(images)

        # DeepfakeBench convention: index 1 = fake.
        if logits.ndim != 2 or logits.size(1) != self.num_classes:
            raise RuntimeError(
                f"{self.name}: expected logits of shape (B, {self.num_classes}), "
                f"got {tuple(logits.shape)}"
            )
        probs = F.softmax(logits, dim=1)[:, 1]
        return probs.detach().float().cpu().numpy()

    def get_preprocessing(self) -> A.Compose:
        return A.Compose(
            [
                A.Resize(height=self.input_size, width=self.input_size),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ]
        )
