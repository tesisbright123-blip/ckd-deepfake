"""DeepfakeBench EfficientNet-B4 teacher loader.

Wraps the pretrained EfficientNet-B4 checkpoint from DeepfakeBench
into the :class:`BaseTeacher` interface so ``scripts/03_generate_soft_labels.py``
and ``src/models/teachers/ensemble.py`` can use it interchangeably with
other teachers.

Architecture compatibility (IMPORTANT)
--------------------------------------
DeepfakeBench's ``effnb4_best.pth`` was trained with the
``efficientnet_pytorch`` library (lukemelas/EfficientNet-PyTorch), **not**
``timm``. The two libraries have:

* Different parameter names: lukemelas uses ``_conv_stem``, ``_bn0``,
  ``_blocks.N._depthwise_conv``, ``_fc`` (with leading underscores and
  ``_bn0``/``_bn1``/``_fc`` instead of ``bn1``/``bn2``/``classifier``);
  timm uses ``conv_stem``, ``bn1``, ``blocks.N.X``, ``classifier``.
* Different block internal structure (slightly different MBConvBlock
  implementations).

Loading a DeepfakeBench checkpoint into a timm model produces ~600 missing
and ~700 unexpected keys, leaving the backbone effectively random-init.
We therefore use ``efficientnet_pytorch`` here to match the checkpoint
exactly.

Called by:
    src/models/teachers/ensemble.py
    scripts/03_generate_soft_labels.py
Reads:
    EfficientNet-B4 weight file (default: {drive}/checkpoints/teachers/efficientnet_b4.pth)
Preprocessing: Albumentations Resize(224) -> ImageNet Normalize -> ToTensor.
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
        """Build EfficientNet-B4 via ``efficientnet_pytorch`` (lukemelas) and
        load DeepfakeBench weights.

        Loader strips the ``efficientnet.`` prefix that DeepfakeBench's
        detector wrapper adds (their saved state dict looks like
        ``{'efficientnet._conv_stem.weight': ..., ..., '_fc.weight': ...}``
        because their detector class holds the model as ``self.efficientnet``).
        After stripping we have keys matching the lukemelas model
        directly: ``_conv_stem.weight``, ``_bn0.weight``, ..., ``_fc.weight``.
        """
        try:
            from efficientnet_pytorch import EfficientNet
        except ImportError as exc:
            raise ModuleNotFoundError(
                "efficientnet_pytorch is required for the DeepfakeBench-trained "
                "EfficientNet-B4 teacher. Install with: pip install efficientnet-pytorch"
            ) from exc

        model = EfficientNet.from_name(
            "efficientnet-b4",
            num_classes=self.num_classes,
        )

        # safe_load_state_dict already strips 'module.', 'backbone.', 'model.',
        # 'net.' prefixes. We additionally strip 'efficientnet.' which is
        # specific to DeepfakeBench's detector wrapper.
        raw_state = safe_load_state_dict(weight_path)
        state_dict = {
            (k[len("efficientnet."):] if k.startswith("efficientnet.") else k): v
            for k, v in raw_state.items()
        }

        # DeepfakeBench's detector class names the classifier head
        # ``last_layer`` (a separate ``nn.Linear(1792, 2)`` outside their
        # ``self.efficientnet`` submodule). The lukemelas EffNet calls the
        # equivalent layer ``_fc``. Without remapping the classifier stays
        # random-init -> garbage P(fake) predictions despite a valid backbone.
        head_remap = {
            "last_layer.weight": "_fc.weight",
            "last_layer.bias":   "_fc.bias",
            # Also handle the rare case where the ckpt stores a ``head.*``
            # naming (some DFB forks).
            "head.weight":       "_fc.weight",
            "head.bias":         "_fc.bias",
        }
        for src_key, dst_key in head_remap.items():
            if src_key in state_dict and dst_key not in state_dict:
                state_dict[dst_key] = state_dict.pop(src_key)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        if missing:
            logger.warning(
                "%s: %d missing keys when loading %s (first 5: %s)",
                self.name,
                len(missing),
                weight_path,
                missing[:5],
            )
        if unexpected:
            logger.warning(
                "%s: %d unexpected keys when loading %s (first 5: %s)",
                self.name,
                len(unexpected),
                weight_path,
                unexpected[:5],
            )

        model.eval()
        model.to(self.device)
        self.model = model
        logger.info(
            "%s: loaded weights from %s (missing=%d, unexpected=%d)",
            self.name,
            weight_path,
            len(missing),
            len(unexpected),
        )

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
