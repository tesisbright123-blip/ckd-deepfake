"""Shared teacher interface for soft-label generation.

All teacher wrappers (EfficientNet-B4, RECCE, CLIP ViT-L/14) subclass
:class:`BaseTeacher` so that the ensemble and
``scripts/03_generate_soft_labels.py`` can treat them uniformly.

Called by:
    src/models/teachers/efficientnet_b4.py (subclass)
    src/models/teachers/recce.py (subclass)
    src/models/teachers/clip_detector.py (subclass)
    src/models/teachers/ensemble.py (iterates teachers)
    scripts/03_generate_soft_labels.py (orchestration)
Reads: none directly. Subclasses and ``safe_load_state_dict`` read
    ``.pth`` checkpoint files produced by DeepfakeBench / CLIPping.
Writes: none.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import albumentations as A
import numpy as np
import torch

# Per-teacher normalization stats. Kept here so each wrapper imports the
# right constants without duplication.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# CLIP's original normalization, used by open_clip's ViT-L/14.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class BaseTeacher(ABC):
    """Abstract interface shared by every teacher wrapper.

    A concrete teacher must set ``name`` and ``input_size``, then implement
    :meth:`load`, :meth:`predict_proba`, and :meth:`get_preprocessing`.

    Attributes:
        name: Short identifier matching the config (e.g. ``"efficientnet_b4"``).
        input_size: Square edge length expected by the model (e.g. 224).
        device: Torch device the model currently lives on.
    """

    name: str = "base"
    input_size: int = 224

    def __init__(self, device: str | torch.device = "cuda") -> None:
        self.device: torch.device = torch.device(device)
        self.model: torch.nn.Module | None = None

    # ------------------------------------------------------------------ #
    #  Required interface
    # ------------------------------------------------------------------ #

    @abstractmethod
    def load(self, weight_path: str | Path) -> None:
        """Build the architecture, load weights, move to ``self.device``."""

    @abstractmethod
    def predict_proba(self, images: torch.Tensor) -> np.ndarray:
        """Return the per-sample probability that each image is *fake*.

        Args:
            images: Tensor of shape ``(B, 3, H, W)`` already on the target
                device and already preprocessed by :meth:`get_preprocessing`.
        Returns:
            ``np.ndarray`` of shape ``(B,)``, dtype ``float32``, values in ``[0, 1]``.
        """

    @abstractmethod
    def get_preprocessing(self) -> A.Compose:
        """Return an Albumentations pipeline matching the teacher's training."""

    # ------------------------------------------------------------------ #
    #  Optional: default eager-unload implementation
    # ------------------------------------------------------------------ #

    def unload(self) -> None:
        """Free the GPU memory held by this teacher."""
        if self.model is not None:
            del self.model
            self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    #  Shared helpers
    # ------------------------------------------------------------------ #

    def _assert_loaded(self) -> torch.nn.Module:
        if self.model is None:
            raise RuntimeError(
                f"{self.name}: .load(weight_path) must be called before inference."
            )
        return self.model


def safe_load_state_dict(
    checkpoint_path: str | Path,
    *,
    prefixes_to_strip: tuple[str, ...] = ("module.", "backbone.", "model.", "net."),
) -> dict[str, torch.Tensor]:
    """Load a ``.pth`` checkpoint and normalize its ``state_dict`` shape.

    DeepfakeBench and CLIPping checkpoints use a few different conventions:

    * bare ``state_dict`` mapping (param -> tensor)
    * ``{"state_dict": {...}, "epoch": int, ...}``
    * ``{"model": {...}}`` or ``{"net": {...}}``

    This helper returns a flat ``param_name -> tensor`` dict, stripping the
    first matching prefix from each key (so ``module.conv.weight`` and
    ``backbone.conv.weight`` both collapse to ``conv.weight``).

    Args:
        checkpoint_path: Path to a PyTorch checkpoint file.
        prefixes_to_strip: Ordered prefixes to try removing from each key.
            Only the first match is stripped per key.

    Returns:
        Flat ``state_dict`` suitable for ``model.load_state_dict(..., strict=False)``.
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Teacher checkpoint not found: {path}")

    obj: Any = torch.load(path, map_location="cpu")

    # Unwrap common container formats.
    if isinstance(obj, dict):
        for key in ("state_dict", "model", "net", "model_state_dict"):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break

    if not isinstance(obj, dict):
        raise ValueError(
            f"Unexpected checkpoint layout in {path}: "
            f"expected dict, got {type(obj).__name__}"
        )

    normalized: dict[str, torch.Tensor] = {}
    for raw_key, tensor in obj.items():
        new_key = raw_key
        for prefix in prefixes_to_strip:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
                break
        normalized[new_key] = tensor
    return normalized
