"""Unit tests for src/data/dataset.py and src/data/augmentations.py.

Run: pytest tests/test_dataset.py -v

Targets:
    src/data/dataset.py:DeepfakeDataset.__len__, __getitem__
        (returns image tensor + label + soft label)
    src/data/augmentations.py:build_transforms
        (train vs val, output shape (3, 224, 224))

Expected split CSV schema (read-only reference, no data written here):
    face_path  (str)   e.g. "faces/gen1/real/000_00.jpg"
    label      (int)   0 = real, 1 = fake
    video_id   (str)   e.g. "000_003"
    generation (str)   e.g. "gen1"
    technique  (str)   e.g. "Face2Face"
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

# Make ``src`` importable when pytest is invoked from the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.augmentations import build_transforms  # noqa: E402
from src.data.dataset import SOFT_LABEL_MISSING, DeepfakeDataset  # noqa: E402


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #


def _write_face_jpeg(path: Path, size: int = 224) -> None:
    """Write a deterministic synthetic BGR face image to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(abs(hash(str(path))) % (2**32))
    img = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    ok = cv2.imwrite(str(path), img)
    assert ok, f"failed to write synthetic face JPEG: {path}"


def _make_split_dataset(tmp_path: Path, n_rows: int = 4) -> Path:
    """Create a tiny split CSV + matching face JPEGs; return the CSV path."""
    rows: list[dict[str, object]] = []
    for i in range(n_rows):
        face = tmp_path / "faces" / f"{i:03d}.jpg"
        _write_face_jpeg(face)
        rows.append(
            {
                "face_path": str(face),
                "frame_idx": i,
                "video_id": f"vid_{i // 2:03d}",  # pair frames per video
                "label": i % 2,
                "dataset": "synthetic",
                "generation": "gen1",
                "technique": "real" if i % 2 == 0 else "DeepFakes",
            }
        )
    csv_path = tmp_path / "gen1_train.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


# --------------------------------------------------------------------------- #
#  build_transforms
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_build_transforms_train_returns_tensor_shape() -> None:
    transform = build_transforms(mode="train", image_size=224)
    img = np.random.default_rng(0).integers(
        0, 256, size=(256, 256, 3), dtype=np.uint8
    )

    out = transform(image=img)["image"]

    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 224, 224)
    assert out.dtype == torch.float32


@pytest.mark.unit
def test_build_transforms_val_returns_tensor_shape() -> None:
    transform = build_transforms(mode="val", image_size=224)
    img = np.random.default_rng(1).integers(
        0, 256, size=(300, 200, 3), dtype=np.uint8
    )

    out = transform(image=img)["image"]

    assert out.shape == (3, 224, 224)


@pytest.mark.unit
def test_build_transforms_train_is_stochastic() -> None:
    """Train pipeline applies random augmentations, so two calls should differ."""
    transform = build_transforms(mode="train", image_size=224)
    img = np.random.default_rng(2).integers(
        0, 256, size=(224, 224, 3), dtype=np.uint8
    )

    a = transform(image=img)["image"]
    b = transform(image=img)["image"]

    # Probability of equality is effectively zero with multiple random ops.
    assert not torch.equal(a, b)


@pytest.mark.unit
def test_build_transforms_invalid_mode_raises() -> None:
    with pytest.raises(ValueError, match="Unknown transform mode"):
        build_transforms(mode="bogus")


# --------------------------------------------------------------------------- #
#  DeepfakeDataset
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_deepfake_dataset_len_matches_csv_rows(tmp_path: Path) -> None:
    csv = _make_split_dataset(tmp_path, n_rows=6)
    ds = DeepfakeDataset(
        csv_path=csv, transform=build_transforms(mode="val"), mode="val"
    )

    assert len(ds) == 6


@pytest.mark.unit
def test_deepfake_dataset_getitem_shape_and_types(tmp_path: Path) -> None:
    csv = _make_split_dataset(tmp_path, n_rows=4)
    ds = DeepfakeDataset(
        csv_path=csv, transform=build_transforms(mode="val"), mode="val"
    )

    image, hard, soft, meta = ds[0]

    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 224, 224)
    assert isinstance(hard, int) and hard in (0, 1)
    assert isinstance(soft, float)
    assert set(meta) == {
        "video_id",
        "dataset",
        "generation",
        "technique",
        "face_path",
    }


@pytest.mark.unit
def test_deepfake_dataset_returns_sentinel_when_soft_labels_missing(
    tmp_path: Path,
) -> None:
    csv = _make_split_dataset(tmp_path, n_rows=3)
    ds = DeepfakeDataset(csv_path=csv, transform=build_transforms("val"))

    _, _, soft, _ = ds[1]

    assert soft == SOFT_LABEL_MISSING


@pytest.mark.unit
def test_deepfake_dataset_reads_soft_labels_aligned_to_csv(tmp_path: Path) -> None:
    csv = _make_split_dataset(tmp_path, n_rows=4)
    soft = np.array([0.10, 0.85, 0.42, 0.99], dtype=np.float32)
    soft_path = tmp_path / "soft.npy"
    np.save(soft_path, soft)

    ds = DeepfakeDataset(
        csv_path=csv,
        transform=build_transforms("val"),
        soft_label_path=soft_path,
    )

    for idx, expected in enumerate(soft):
        _, _, actual, _ = ds[idx]
        assert actual == pytest.approx(float(expected), abs=1e-6)


@pytest.mark.unit
def test_deepfake_dataset_csv_not_found_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Split CSV not found"):
        DeepfakeDataset(
            csv_path=tmp_path / "does_not_exist.csv",
            transform=build_transforms("val"),
        )


@pytest.mark.unit
def test_deepfake_dataset_missing_required_columns_raises(
    tmp_path: Path,
) -> None:
    bad_csv = tmp_path / "bad.csv"
    # Missing "technique", "dataset", etc.
    pd.DataFrame(
        [{"face_path": "x.jpg", "label": 0, "video_id": "v0"}]
    ).to_csv(bad_csv, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        DeepfakeDataset(csv_path=bad_csv, transform=build_transforms("val"))


@pytest.mark.unit
def test_deepfake_dataset_soft_label_shape_mismatch_raises(
    tmp_path: Path,
) -> None:
    csv = _make_split_dataset(tmp_path, n_rows=4)
    wrong_soft = np.zeros(3, dtype=np.float32)  # csv has 4 rows
    soft_path = tmp_path / "wrong.npy"
    np.save(soft_path, wrong_soft)

    with pytest.raises(ValueError, match="does not match"):
        DeepfakeDataset(
            csv_path=csv,
            transform=build_transforms("val"),
            soft_label_path=soft_path,
        )


@pytest.mark.unit
def test_deepfake_dataset_soft_label_file_missing_raises(tmp_path: Path) -> None:
    csv = _make_split_dataset(tmp_path, n_rows=2)
    missing = tmp_path / "nope.npy"

    with pytest.raises(FileNotFoundError, match="Soft labels not found"):
        DeepfakeDataset(
            csv_path=csv,
            transform=build_transforms("val"),
            soft_label_path=missing,
        )


@pytest.mark.unit
def test_deepfake_dataset_unreadable_image_raises(tmp_path: Path) -> None:
    """If the face JPEG is missing on disk, __getitem__ must not silently fail."""
    csv = _make_split_dataset(tmp_path, n_rows=2)
    # Delete the face JPEG referenced by the first row.
    df = pd.read_csv(csv)
    Path(df.loc[0, "face_path"]).unlink()

    ds = DeepfakeDataset(csv_path=csv, transform=build_transforms("val"))
    with pytest.raises(FileNotFoundError, match="Failed to read image"):
        _ = ds[0]
