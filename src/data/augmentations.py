"""Albumentations augmentation pipelines for training and validation.

The train pipeline mimics distortions common in real-world deepfake
scenarios: re-encoding, mild noise, blur, and geometric jitter. The val
pipeline only normalises with ImageNet statistics.

Called by: src/data/dataset.py (DeepfakeDataset), src/data/dataloader.py,
           scripts/03_generate_soft_labels.py (val), scripts/04/05 (train)
Data files: none. Returns Albumentations Compose objects only.
"""
from __future__ import annotations

from typing import Any

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet statistics — matches pretrained backbones (MobileNetV3, EfficientNet, CLIP).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(
    image_size: int = 224,
    aug_cfg: dict[str, Any] | None = None,
) -> A.Compose:
    """Training-time augmentations.

    Args:
        image_size: Output edge length (square).
        aug_cfg: Optional dict of overrides. Expected keys match the
            ``data.augmentation`` block in ``configs/default.yaml``:
                image_compression: [q_lo, q_hi]
                gauss_noise_sigma: [lo, hi]       # converted to var_limit
                gauss_blur_kernel: [k_lo, k_hi]
                horizontal_flip_p: float
                brightness_contrast: float
                shift_scale_rotate: [shift, scale, rotate_deg]
    """
    cfg = aug_cfg or {}
    compression = cfg.get("image_compression", [40, 100])
    sigma_lo, sigma_hi = cfg.get("gauss_noise_sigma", [0, 20])
    blur_lo, blur_hi = cfg.get("gauss_blur_kernel", [3, 7])
    hflip_p = cfg.get("horizontal_flip_p", 0.5)
    bc = cfg.get("brightness_contrast", 0.2)
    ssr_shift, ssr_scale, ssr_rotate = cfg.get("shift_scale_rotate", [0.0625, 0.1, 10])

    # var_limit is sigma^2 for GaussNoise
    var_limit = (float(sigma_lo) ** 2, float(sigma_hi) ** 2)

    return A.Compose(
        [
            # DF40 ships pre-aligned 224x224 face crops, so Resize is a no-op
            # for in-distribution data. Using Resize (not LongestMaxSize+Pad)
            # avoids black-padding artefacts on any non-square outliers and
            # matches DeepfakeBench / DFDC-winner conventions.
            A.Resize(height=image_size, width=image_size),
            A.ImageCompression(
                quality_lower=int(compression[0]),
                quality_upper=int(compression[1]),
                p=0.5,
            ),
            A.GaussNoise(var_limit=var_limit, p=0.3),
            A.GaussianBlur(blur_limit=(int(blur_lo), int(blur_hi)), p=0.3),
            A.HorizontalFlip(p=float(hflip_p)),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=float(bc),
                        contrast_limit=float(bc),
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=10,
                        sat_shift_limit=20,
                        val_shift_limit=20,
                    ),
                ],
                p=0.5,
            ),
            A.ShiftScaleRotate(
                shift_limit=float(ssr_shift),
                scale_limit=float(ssr_scale),
                rotate_limit=int(ssr_rotate),
                p=0.3,
                border_mode=0,
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_val_transforms(image_size: int = 224) -> A.Compose:
    """Validation / test / teacher-inference transforms (no augmentation)."""
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def build_transforms(
    mode: str,
    image_size: int = 224,
    aug_cfg: dict[str, Any] | None = None,
) -> A.Compose:
    """Dispatch helper used by the dataloader factory."""
    if mode == "train":
        return get_train_transforms(image_size=image_size, aug_cfg=aug_cfg)
    if mode in {"val", "test"}:
        return get_val_transforms(image_size=image_size)
    raise ValueError(
        f"Unknown transform mode: {mode!r} (expected 'train', 'val', or 'test')"
    )
