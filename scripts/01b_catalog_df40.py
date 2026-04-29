"""Step 1b: Catalog DF40 pre-processed face images.

CLI entrypoint. Not imported by other code.

DF40 ships as already face-cropped images (not raw videos). This script
enumerates the DF40 image tree and writes a ``metadata_<generation>.csv``
that downstream scripts (02 splits, 03 soft labels) can consume.

Because DF40 contains 40 techniques spanning the full deepfake history,
the CKD pipeline partitions DF40 into three generational buckets
(gen1 = classic face-swap, gen2 = reenactment, gen3 = diffusion & modern).
The technique list for each generation is read from
``configs/default.yaml -> data.generations.<gen>.techniques``.

Expected DF40 layout on Drive (after ``gdown`` download):

    {drive}/datasets/raw/df40/
        <technique>/
            ff/   or  cdf/   or  fake/      <- fake crops
                [<video_id>/]
                    frame_00.png
                    frame_01.png
                    ...

Real faces come from a separate DF40 real pool:

    {drive}/datasets/raw/df40_real/
        ff_real/<video_id>/*.png
        cdf_real/<video_id>/*.png

(or any folder whose leaves contain face images — we just glob and label 0).

Accepted image extensions: .png, .jpg, .jpeg, .webp.

Usage:
    # Let config drive the technique list + domains (typical).
    python scripts/01b_catalog_df40.py --config configs/default.yaml \
        --generation gen1

    # Override techniques from CLI.
    python scripts/01b_catalog_df40.py --generation gen2 \
        --techniques fomm,facevid2vid,wav2lip

    # Only write DF40 fakes, no reals.
    python scripts/01b_catalog_df40.py --generation gen3 --no-real

Reads:
    YAML config (--config).
    DF40 fake images at {drive}/datasets/raw/df40/**/*.png
    DF40 real images at {drive}/datasets/raw/df40_real/**/*.png
Writes:
    {drive}/datasets/faces/metadata_<generation>.csv
        columns: face_path, frame_idx, video_id, label, dataset,
                 generation, technique
    Log file at runs/catalog_df40_<generation>.log
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

# DF40 fake-crop subfolders we try under each technique directory.
# Most techniques use ff/cdf; a handful (MidJourney, StarGAN, e4e, CollabDiff,
# whichfaceisreal) use fake/real; some have no domain folder at all.
_FAKE_DOMAINS: tuple[str, ...] = ("ff", "cdf", "fake")
_REAL_DOMAINS_IN_TECH: tuple[str, ...] = ("real",)


def _iter_images(folder: Path) -> list[Path]:
    out: list[Path] = []
    for ext in _IMAGE_EXTS:
        out.extend(folder.rglob(f"*{ext}"))
    return sorted(p for p in out if p.is_file())


def _group_by_video(images: list[Path], domain_dir: Path) -> dict[str, list[Path]]:
    """Cluster images by immediate parent folder (= ``video_id``).

    If images live directly under ``domain_dir`` (flat layout) each file
    becomes its own single-frame video.
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


def _catalog_technique_dir(
    tech_dir: Path,
    *,
    technique: str,
    generation: str,
    label: int,
    logger,
) -> list[dict[str, Any]]:
    """Walk one technique folder and emit rows.

    Handles three layouts:
      1) <tech_dir>/<ff|cdf|fake>/<video_id>/*.png  (typical)
      2) <tech_dir>/<video_id>/*.png                (no domain folder)
      3) <tech_dir>/*.png                           (flat)
    """
    rows: list[dict[str, Any]] = []
    if not tech_dir.is_dir():
        return rows

    fake_domains = [d for d in _FAKE_DOMAINS if (tech_dir / d).is_dir()]
    real_domains = [d for d in _REAL_DOMAINS_IN_TECH if (tech_dir / d).is_dir()]
    domains_present = fake_domains if label == 1 else real_domains

    if domains_present:
        for domain in domains_present:
            domain_dir = tech_dir / domain
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
                            "label": label,
                            "dataset": "df40",
                            "generation": generation,
                            "technique": technique,
                        }
                    )
        return rows

    # Flat fallback only applies when NO domain dirs (ff/cdf/fake/real) exist
    # under this tech. Otherwise we'd accidentally re-walk fake images as real
    # (or vice versa) for techniques like faceswap that use ff/cdf with no
    # explicit real/ subdir — those should yield zero inline-real rows here.
    if fake_domains or real_domains:
        return rows

    # Truly flat layout: only emit for label=1 (we don't have a way to tell
    # real-only flat layouts apart from fake-only ones).
    if label != 1:
        return rows

    imgs = _iter_images(tech_dir)
    if imgs:
        grouped = _group_by_video(imgs, tech_dir)
        for vid_id, frame_paths in grouped.items():
            for frame_idx, face_path in enumerate(frame_paths):
                rows.append(
                    {
                        "face_path": str(face_path),
                        "frame_idx": frame_idx,
                        "video_id": f"{technique}_{vid_id}",
                        "label": label,
                        "dataset": "df40",
                        "generation": generation,
                        "technique": technique,
                    }
                )
    if not rows and label == 1:
        logger.warning("Technique '%s' at %s produced 0 fake rows.", technique, tech_dir)
    return rows


def catalog_df40_fakes(
    dataset_root: Path,
    *,
    techniques: list[str],
    generation: str,
    logger,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Enumerate DF40 fake face-crops for the requested technique list.

    Returns ``(fake_rows, inline_real_rows)``. The second list contains real
    frames that ship inside method folders (``<tech>/real/``) for the
    9 "unknown" methods (MidJourney, stargan, starganv2, styleclip, e4e,
    CollabDiff, whichfaceisreal).
    """
    fake_rows: list[dict[str, Any]] = []
    inline_real_rows: list[dict[str, Any]] = []
    if not dataset_root.is_dir():
        logger.error("DF40 fake root not found: %s", dataset_root)
        return fake_rows, inline_real_rows

    # Build case-insensitive lookup of on-disk technique folders.
    on_disk = {p.name.lower(): p for p in dataset_root.iterdir() if p.is_dir()}
    for tech in techniques:
        key = tech.lower()
        if key not in on_disk:
            logger.warning(
                "Technique folder '%s' not found under %s — skipping.",
                tech,
                dataset_root,
            )
            continue
        tech_dir = on_disk[key]
        f_rows = _catalog_technique_dir(
            tech_dir, technique=tech, generation=generation, label=1, logger=logger,
        )
        r_rows = _catalog_technique_dir(
            tech_dir, technique=tech, generation=generation, label=0, logger=logger,
        )
        fake_rows.extend(f_rows)
        inline_real_rows.extend(r_rows)
        logger.info(
            "  technique=%s fake_frames=%d inline_real_frames=%d",
            tech,
            len(f_rows),
            len(r_rows),
        )
    return fake_rows, inline_real_rows


def catalog_df40_reals(
    real_root: Path,
    *,
    generation: str,
    logger,
) -> list[dict[str, Any]]:
    """Enumerate DF40 shared real-face crops (labelled 0)."""
    rows: list[dict[str, Any]] = []
    if not real_root.is_dir():
        logger.warning(
            "DF40 real root not found: %s (no real frames added).", real_root
        )
        return rows

    # Accept either `<real_root>/<source>/<video_id>/*.png`
    # or `<real_root>/<video_id>/*.png`. We glob every image and let
    # _group_by_video infer video_id from the parent.
    imgs = _iter_images(real_root)
    if not imgs:
        logger.warning("No real images under %s", real_root)
        return rows

    grouped = _group_by_video(imgs, real_root)
    for vid_id, frame_paths in grouped.items():
        for frame_idx, face_path in enumerate(frame_paths):
            rows.append(
                {
                    "face_path": str(face_path),
                    "frame_idx": frame_idx,
                    "video_id": f"real_{vid_id}",
                    "label": 0,
                    "dataset": "df40",
                    "generation": generation,
                    "technique": "real",
                }
            )
    logger.info(
        "Real frames: %d (videos=%d) from %s",
        len(rows),
        len(grouped),
        real_root,
    )
    return rows


def _parse_techniques(text: str | None) -> list[str] | None:
    if not text:
        return None
    techs = [item.strip() for item in text.split(",") if item.strip()]
    return techs or None


def _resolve_techniques(
    cli_techniques: list[str] | None,
    cfg: dict[str, Any],
    generation: str,
) -> list[str]:
    if cli_techniques:
        return cli_techniques
    gen_cfg = cfg["data"]["generations"].get(generation)
    if not gen_cfg:
        raise KeyError(
            f"Generation '{generation}' missing from config "
            f"(data.generations.{generation})."
        )
    techniques = gen_cfg.get("techniques")
    if not techniques:
        raise KeyError(
            f"data.generations.{generation}.techniques is empty; "
            "populate it in configs/default.yaml or pass --techniques."
        )
    return list(techniques)


def _run(args: argparse.Namespace) -> int:
    logger = get_logger(
        "catalog_df40",
        log_file=f"runs/catalog_df40_{args.generation}.log",
    )

    cfg = load_config(args.config)
    drive_root = Path(cfg["paths"]["drive_root"])

    df40_root = (
        Path(args.df40_root)
        if args.df40_root
        else drive_root / "datasets" / "raw" / "df40"
    )
    real_root = (
        Path(args.real_root)
        if args.real_root
        else drive_root / "datasets" / "raw" / "df40_real"
    )

    techniques = _resolve_techniques(
        _parse_techniques(args.techniques), cfg, args.generation
    )
    logger.info(
        "Cataloging DF40 generation=%s techniques=%s",
        args.generation,
        techniques,
    )

    fake_rows, inline_real_rows = catalog_df40_fakes(
        df40_root,
        techniques=techniques,
        generation=args.generation,
        logger=logger,
    )
    if not fake_rows:
        logger.error(
            "No DF40 fake images found under %s for techniques %s. "
            "Verify download completed and folder names match.",
            df40_root,
            techniques,
        )
        return 2
    logger.info(
        "DF40 fake rows total: %d  (inline-real rows: %d)",
        len(fake_rows),
        len(inline_real_rows),
    )

    pooled_real_rows: list[dict[str, Any]] = []
    if not args.no_real:
        pooled_real_rows = catalog_df40_reals(
            real_root, generation=args.generation, logger=logger
        )

    all_rows = inline_real_rows + pooled_real_rows + fake_rows
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
        description=(
            "Catalog DF40 preprocessed face images into a "
            "metadata_<generation>.csv for downstream splits/training."
        )
    )
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument(
        "--generation",
        required=True,
        choices=["gen1", "gen2", "gen3"],
        help="Generation bucket to build. Techniques come from "
        "data.generations.<gen>.techniques in the YAML config.",
    )
    p.add_argument(
        "--df40-root",
        default=None,
        help="Override {drive}/datasets/raw/df40 (fake crops).",
    )
    p.add_argument(
        "--real-root",
        default=None,
        help="Override {drive}/datasets/raw/df40_real (real crops).",
    )
    p.add_argument(
        "--techniques",
        default=None,
        help=(
            "Comma-separated technique folders (e.g. 'sd2.1,ddim,PixArt'). "
            "If omitted, read from config."
        ),
    )
    p.add_argument(
        "--no-real",
        action="store_true",
        help="Skip cataloging real-face crops (write only DF40 fakes).",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Override output metadata CSV path.",
    )
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(_run(_parse_args()))
