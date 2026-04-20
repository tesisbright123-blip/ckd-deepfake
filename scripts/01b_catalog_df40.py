"""Step 1b: Catalog DF40 pre-processed face images (gen3).

Entrypoint script — invoked via CLI, not imported by other code.

DF40 ships as already face-cropped images (not raw videos), so the normal
face-extractor in scripts/01 is not needed. This script enumerates the
DF40 image tree, borrows REAL rows from an earlier generation's metadata
(typically ``gen1`` = FF++ real), and writes a combined ``metadata_gen3.csv``
that downstream scripts (02 splits, 03 soft labels) can consume.

Expected DF40 layout on Drive (after ``gdown`` download):

    {drive}/datasets/raw/df40/
        <technique>/
            ff/                   or   cdf/
                <video_id>/
                    frame_00.png
                    frame_01.png
                    ...

Accepted image extensions: .png, .jpg, .jpeg, .webp.

Usage:
    python scripts/01b_catalog_df40.py                                     # defaults
    python scripts/01b_catalog_df40.py --borrow-real-from gen1 \
        --techniques sd_2_1,ddpm,rddm
    python scripts/01b_catalog_df40.py --no-borrow-real                   # only DF40 fakes

Reads:
    YAML config (--config).
    DF40 preprocessed images at {drive}/datasets/raw/df40/**/*.png
    (Optional) prior metadata at
        {drive}/datasets/faces/metadata_{borrow_from}.csv   for real rows.
Writes:
    {drive}/datasets/faces/metadata_gen3.csv
        columns: face_path, frame_idx, video_id, label, dataset,
                 generation, technique
    Log file at runs/catalog_df40.log
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.utils.config import load_config  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

_IMAGE_EXTS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp")

# Canonical column order — must match src/data/splits.REQUIRED_COLUMNS.
_COLUMNS: tuple[str, ...] = (
    "face_path",
    "frame_idx",
    "video_id",
    "label",
    "dataset",
    "generation",
    "technique",
)


def _iter_images(folder: Path) -> list[Path]:
    out: list[Path] = []
    for ext in _IMAGE_EXTS:
        out.extend(folder.rglob(f"*{ext}"))
    return sorted(p for p in out if p.is_file())


def _group_by_video(images: list[Path], domain_dir: Path) -> dict[str, list[Path]]:
    """Cluster images by immediate parent folder (= ``video_id``).

    If images live directly under ``domain_dir`` (flat layout) we treat each
    file as its own single-frame video.
    """
    grouped: dict[str, list[Path]] = {}
    for img in images:
        rel_parent = img.parent
        if rel_parent == domain_dir:
            vid = img.stem
        else:
            vid = rel_parent.name
        grouped.setdefault(vid, []).append(img)
    for vid in grouped:
        grouped[vid].sort()
    return grouped


def catalog_df40_preprocessed(
    dataset_root: Path,
    *,
    allow_techniques: set[str] | None = None,
    domains: tuple[str, ...] = ("ff", "cdf"),
) -> list[dict[str, Any]]:
    """Enumerate DF40 face-cropped images and emit metadata rows (all FAKE)."""
    rows: list[dict[str, Any]] = []
    if not dataset_root.is_dir():
        return rows

    for tech_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        technique = tech_dir.name
        if allow_techniques and technique not in allow_techniques:
            continue
        for domain in domains:
            domain_dir = tech_dir / domain
            if not domain_dir.is_dir():
                continue
            imgs = _iter_images(domain_dir)
            if not imgs:
                continue
            grouped = _group_by_video(imgs, domain_dir)
            for vid_id, frame_paths in grouped.items():
                for frame_idx, face_path in enumerate(frame_paths):
                    rows.append(
                        {
                            "face_path": str(face_path),
                            "frame_idx": frame_idx,
                            "video_id": f"{technique}_{domain}_{vid_id}",
                            "label": 1,  # all DF40 images are fake
                            "dataset": "df40_diffusion_subset",
                            "generation": "gen3",
                            "technique": technique,
                        }
                    )
    return rows


def _borrow_real_rows(
    source_csv: Path,
    *,
    target_generation: str,
    logger,
) -> list[dict[str, Any]]:
    """Pull real (``label == 0``) rows from an earlier generation's metadata."""
    if not source_csv.is_file():
        logger.warning(
            "Borrow-real source not found: %s (continuing without real rows)",
            source_csv,
        )
        return []
    df = pd.read_csv(source_csv)
    real = df[df["label"] == 0].copy()
    if real.empty:
        logger.warning("No real rows in %s", source_csv)
        return []
    # Retag generation so downstream splits recognize these as gen3 real.
    real["generation"] = target_generation
    logger.info(
        "Borrowed %d real rows from %s (%d videos)",
        len(real),
        source_csv.name,
        real["video_id"].nunique(),
    )
    return real[list(_COLUMNS)].to_dict(orient="records")


def _parse_techniques(text: str | None) -> set[str] | None:
    if not text:
        return None
    techs = {item.strip() for item in text.split(",") if item.strip()}
    return techs or None


def _run(args: argparse.Namespace) -> int:
    logger = get_logger("catalog_df40", log_file="runs/catalog_df40.log")

    cfg = load_config(args.config)
    drive_root = Path(cfg["paths"]["drive_root"])

    df40_root = (
        Path(args.df40_root)
        if args.df40_root
        else drive_root / "datasets" / "raw" / "df40"
    )
    if not df40_root.is_dir():
        logger.error(
            "DF40 root not found: %s. Download via gdown folder first.", df40_root
        )
        return 1

    allow_techniques = _parse_techniques(args.techniques)
    domains = tuple(d.strip() for d in args.domains.split(",") if d.strip())
    logger.info(
        "Scanning DF40 at %s (techniques=%s domains=%s)",
        df40_root,
        sorted(allow_techniques) if allow_techniques else "all",
        list(domains),
    )

    fake_rows = catalog_df40_preprocessed(
        df40_root, allow_techniques=allow_techniques, domains=domains
    )
    if not fake_rows:
        logger.error(
            "No DF40 images found under %s (checked extensions %s). "
            "Verify download completed.",
            df40_root,
            list(_IMAGE_EXTS),
        )
        return 2

    # Summary per technique for sanity check.
    counts: dict[str, int] = {}
    for r in fake_rows:
        counts[r["technique"]] = counts.get(r["technique"], 0) + 1
    for tech, n in sorted(counts.items()):
        logger.info("  technique=%s frames=%d", tech, n)
    logger.info("DF40 fake rows total: %d", len(fake_rows))

    real_rows: list[dict[str, Any]] = []
    if args.borrow_real_from:
        faces_dir = drive_root / "datasets" / "faces"
        source_csv = faces_dir / f"metadata_{args.borrow_real_from}.csv"
        real_rows = _borrow_real_rows(
            source_csv, target_generation=args.generation, logger=logger
        )

    all_rows = real_rows + fake_rows
    output_path = (
        Path(args.output)
        if args.output
        else drive_root / "datasets" / "faces" / f"metadata_{args.generation}.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(all_rows)[list(_COLUMNS)]
    df.to_csv(output_path, index=False)
    logger.info(
        "Wrote %s  rows=%d  real=%d  fake=%d  videos=%d",
        output_path,
        len(df),
        int((df["label"] == 0).sum()),
        int((df["label"] == 1).sum()),
        df["video_id"].nunique(),
    )
    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Catalog DF40 preprocessed face images for gen3"
    )
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument(
        "--generation",
        default="gen3",
        help="Generation tag to write (default: gen3)",
    )
    p.add_argument(
        "--df40-root",
        default=None,
        help="Override {drive}/datasets/raw/df40",
    )
    p.add_argument(
        "--techniques",
        default=None,
        help=(
            "Comma-separated technique folder names to keep "
            "(e.g. 'sd_2_1,ddpm,rddm'). Default: all folders under df40/."
        ),
    )
    p.add_argument(
        "--domains",
        default="ff,cdf",
        help="Comma-separated DF40 domains to include (default: ff,cdf)",
    )
    p.add_argument(
        "--borrow-real-from",
        default="gen1",
        help=(
            "Source generation tag to borrow REAL rows from "
            "(default: gen1). Pass empty string to disable."
        ),
    )
    p.add_argument(
        "--no-borrow-real",
        action="store_true",
        help="Shortcut for --borrow-real-from '' (only DF40 fakes)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Override output metadata CSV path",
    )
    args = p.parse_args()
    if args.no_borrow_real:
        args.borrow_real_from = ""
    return args


if __name__ == "__main__":
    raise SystemExit(_run(_parse_args()))
