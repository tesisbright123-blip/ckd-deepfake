"""DeepfakeBench RECCE teacher loader.

RECCE ("Reconstruction-Classification") is a custom architecture from
https://github.com/SCLBD/DeepfakeBench. Its definition is not available
on PyPI; the model class must be vendored into this repo.

How to vendor the architecture:
    1. Copy DeepfakeBench's RECCE model file into
       ``src/models/teachers/recce_arch.py``
    2. Expose a zero-arg factory named ``build_recce()`` that returns a
       ``torch.nn.Module`` whose forward returns ``(B, 2)`` logits with
       index 1 = fake (matching DeepfakeBench convention).
    3. Place the checkpoint at
       ``{drive}/checkpoints/teachers/recce.pth``.

Called by:
    src/models/teachers/ensemble.py
    scripts/03_generate_soft_labels.py
Reads:
    RECCE weight file (default: {drive}/checkpoints/teachers/recce.pth)
Preprocessing: LongestMaxSize -> PadIfNeeded(224) -> ImageNet Normalize -> ToTensor.
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


_VENDORING_INSTRUCTIONS = (
    "RECCE architecture not found. Copy DeepfakeBench's RECCE model "
    "definition into src/models/teachers/recce_arch.py and expose a "
    "``build_recce()`` factory. See module docstring for details."
)


class RECCETeacher(BaseTeacher):
    """DeepfakeBench RECCE teacher wrapper.

    Args:
        device: Torch device.
        num_classes: Head dimension (2 = real/fake).
    """

    name = "recce"
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
        model = self._build_architecture()
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

    def _build_architecture(self) -> torch.nn.Module:
        """Try to import a vendored RECCE architecture."""
        try:
            from src.models.teachers import recce_arch  # type: ignore[attr-defined]
        except ImportError as exc:
            raise ModuleNotFoundError(_VENDORING_INSTRUCTIONS) from exc

        if not hasattr(recce_arch, "build_recce"):
            raise AttributeError(_VENDORING_INSTRUCTIONS)

        return recce_arch.build_recce()  # type: ignore[no-any-return]

    # ------------------------------------------------------------------ #
    #  Inference
    # ------------------------------------------------------------------ #

    @torch.inference_mode()
    def predict_proba(self, images: torch.Tensor) -> np.ndarray:
        model = self._assert_loaded()
        images = images.to(self.device, non_blocking=True)
        output = model(images)

        # DeepfakeBench RECCE forward often returns (logits, recon) during
        # training — keep only the classification logits here.
        if isinstance(output, (tuple, list)):
            output = output[0]

        if output.ndim != 2 or output.size(1) != self.num_classes:
            raise RuntimeError(
                f"{self.name}: expected logits of shape (B, {self.num_classes}), "
                f"got {tuple(output.shape)}"
            )
        probs = F.softmax(output, dim=1)[:, 1]
        return probs.detach().float().cpu().numpy()

    def get_preprocessing(self) -> A.Compose:
        return A.Compose(
            [
                A.LongestMaxSize(max_size=self.input_size),
                A.PadIfNeeded(
                    min_height=self.input_size,
                    min_width=self.input_size,
                    border_mode=0,
                    value=0,
                ),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ]
        )
