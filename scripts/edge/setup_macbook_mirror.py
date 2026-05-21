"""Setup local mirror on MacBook for edge evaluation.

CLI entrypoint. Mirrors the Colab ``00_setup_local_mirror.py`` flow but
designed for an offline MacBook workflow where:

  - DF40 zip archives are already downloaded manually (via Drive web UI)
    and sit in a local folder the user controls.
  - Student checkpoints + split CSVs are likewise pre-downloaded.
  - No Google Drive mount is assumed (no ``rclone`` / ``gdown`` calls).

Layout produced (under ``--output-root``, default ``~/ckd-edge``):

    ~/ckd-edge/
      df40/                       <- extracted technique frames
        sd2.1/...
        blendface/...
        ...
      df40_real/                  <- extracted real-class frames
        ff_real/...
        cdf_real/...
      mirror/
        datasets/splits/          <- path-rewritten CSVs
          gen1_test.csv
          gen2_test.csv
          gen3_test.csv
        checkpoints/
          students/
            gen1_seed0/best.pth
            gen1_seed1/best.pth
            ... (9 checkpoints)
      configs/
        macbook.yaml              <- drive_root override pointing at mirror/

After this finishes, ``notebooks/08_edge_evaluation_macbook.ipynb`` can
load ``configs/macbook.yaml`` and read images at native NVMe speed.

Usage:
    # Most common: ZIPs in ~/Downloads/df40_zips/, ckpts in ~/Downloads/ckd_ckpts/
    python scripts/edge/setup_macbook_mirror.py \
        --zip-dir   ~/Downloads/df40_zips \
        --ckpt-dir  ~/Downloads/ckd_ckpts \
        --csv-dir   ~/Downloads/ckd_splits \
        --output-root ~/ckd-edge

    # Resume mode (idempotent): skips already-extracted ZIPs and copied ckpts.
    python scripts/edge/setup_macbook_mirror.py ... --resume

Reads:
    {zip-dir}/<technique>.zip
    {zip-dir}/FaceForensics++_real_data_for_DF40.zip
    {zip-dir}/Celeb-DF-v2_real_data_for_DF40.zip
    {ckpt-dir}/gen{1,2,3}_*_seed{0,1,2}/best.pth (9 files)
    {csv-dir}/gen{1,2,3}_test.csv (3 files)

Writes:
    {output-root}/df40/<technique>/...
    {output-root}/df40_real/{ff_real,cdf_real}/...
    {output-root}/mirror/datasets/splits/gen{1,2,3}_test.csv (rewritten)
    {output-root}/mirror/checkpoints/students/gen{1,2,3}_*_seed{0,1,2}/best.pth
    configs/macbook.yaml
    Per-zip marker files (for --resume)
"""
from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.utils.logger import get_logger  # noqa: E402

# Match the gen-to-technique mapping from 00_setup_local_mirror.py exactly
# so that the same ZIPs can be reused across setups.
TECHNIQUES_PER_GEN: dict[str, list[str]] = {
    "gen1": [
        "blendface", "e4s", "facedancer", "faceswap", "fsgan",
        "inswap", "mobileswap", "simswap", "uniface",
    ],
    "gen2": [
        "fomm", "facevid2vid", "wav2lip", "MRAA", "one_shot_free",
        "pirender", "tpsm", "lia", "danet", "sadtalker", "mcnet",
    ],
    "gen3": [
        "sd2.1", "ddim", "pixart", "DiT", "SiT",
        "StyleGAN2", "StyleGAN3", "StyleGANXL", "VQGAN",
    ],
}

REAL_ZIPS: dict[str, str] = {
    "ff_real": "FaceForensics++_real_data_for_DF40.zip",
    "cdf_real": "Celeb-DF-v2_real_data_for_DF40.zip",
}

_EXTRACT_OVERRIDES: dict[str, str] = {
    "FaceForensics++_real_data_for_DF40.zip": "df40_real/ff_real/",
    "Celeb-DF-v2_real_data_for_DF40.zip": "df40_real/cdf_real/",
}

_EXTRACT_MARKER = ".extracted_ok"

CKPT_PATTERNS: list[str] = [
    "gen1_seed0/best.pth",
    "gen1_seed1/best.pth",
    "gen1_seed2/best.pth",
    "gen2_replay+ewc_seed0/best.pth",
    "gen2_replay+ewc_seed1/best.pth",
    "gen2_replay+ewc_seed2/best.pth",
    "gen3_replay+ewc_seed0/best.pth",
    "gen3_replay+ewc_seed1/best.pth",
    "gen3_replay+ewc_seed2/best.pth",
]

CSV_FILES: list[str] = [
    "gen1_test.csv",
    "gen2_test.csv",
    "gen3_test.csv",
]


def _required_zip_filenames(generations: list[str]) -> list[str]:
    needed: set[str] = set()
    for g in generations:
        if g not in TECHNIQUES_PER_GEN:
            raise ValueError(f"Unknown generation: {g!r}")
        for stem in TECHNIQUES_PER_GEN[g]:
            needed.add(f"{stem}.zip")
    needed.update(REAL_ZIPS.values())
    return sorted(needed)


def _extract_target(local_data: Path, zip_name: str) -> Path:
    if zip_name in _EXTRACT_OVERRIDES:
        return local_data / _EXTRACT_OVERRIDES[zip_name]
    return local_data / "df40"


def _per_zip_marker(local_data: Path, zip_name: str) -> Path:
    if zip_name in _EXTRACT_OVERRIDES:
        return local_data / _EXTRACT_OVERRIDES[zip_name] / _EXTRACT_MARKER
    stem = Path(zip_name).stem
    return local_data / "df40" / stem / _EXTRACT_MARKER


def step1_extract_zips(
    *,
    zip_dir: Path,
    local_data: Path,
    generations: list[str],
    resume: bool,
    logger,
) -> None:
    """Extract each requested ZIP from ``zip_dir`` into ``local_data``."""
    if not zip_dir.is_dir():
        raise FileNotFoundError(f"ZIP dir not found: {zip_dir}")
    local_data.mkdir(parents=True, exist_ok=True)
    needed = _required_zip_filenames(generations)
    logger.info(
        "Step 1/4: extracting %d ZIPs from %s -> %s", len(needed), zip_dir, local_data
    )

    for zname in needed:
        src = zip_dir / zname
        target = _extract_target(local_data, zname)
        marker = _per_zip_marker(local_data, zname)
        if resume and marker.is_file():
            logger.info("  [skip ] %s already extracted (marker at %s)", zname, marker)
            continue
        if not src.is_file():
            logger.warning("  [miss ] %s — not found in %s", zname, zip_dir)
            continue
        target.mkdir(parents=True, exist_ok=True)
        size_gb = src.stat().st_size / 1e9
        logger.info("  [ext  ] %s (%.2f GB) -> %s", zname, size_gb, target)
        with zipfile.ZipFile(src) as zf:
            zf.extractall(target)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(datetime.now().isoformat())


def step2_copy_checkpoints(
    *, ckpt_dir: Path, mirror_root: Path, resume: bool, logger
) -> None:
    """Copy 9 student checkpoints into the mirror layout."""
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint dir not found: {ckpt_dir}")
    dst_root = mirror_root / "checkpoints" / "students"
    dst_root.mkdir(parents=True, exist_ok=True)
    logger.info("Step 2/4: copying %d student checkpoints -> %s", len(CKPT_PATTERNS), dst_root)

    for rel in CKPT_PATTERNS:
        src = ckpt_dir / rel
        dst = dst_root / rel
        if not src.is_file():
            logger.warning("  [miss ] %s not in %s — skipping", rel, ckpt_dir)
            continue
        if resume and dst.is_file() and dst.stat().st_size == src.stat().st_size:
            logger.info("  [skip ] %s already copied (%.1f MB)", rel, dst.stat().st_size / 1e6)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        logger.info("  [copy ] %s -> %s (%.1f MB)", rel, dst, dst.stat().st_size / 1e6)


def step3_rewrite_csvs(
    *,
    csv_dir: Path,
    local_data: Path,
    mirror_root: Path,
    logger,
) -> None:
    """Rewrite test CSV face_path columns to point at the local MacBook data."""
    if not csv_dir.is_dir():
        raise FileNotFoundError(f"CSV dir not found: {csv_dir}")
    splits_dst = mirror_root / "datasets" / "splits"
    splits_dst.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Step 3/4: rewriting %d test CSVs -> %s (face_path -> %s)",
        len(CSV_FILES),
        splits_dst,
        local_data,
    )

    # Anything looking like the Colab path on Drive becomes the MacBook path.
    # We accept BOTH ``/content/drive/MyDrive/CKD_Thesis/datasets/raw/`` (the
    # original Drive prefix) and ``/content/df40_local/`` (the Colab NVMe
    # mirror prefix) — both should resolve to ``local_data``.
    rewrite_pairs = [
        ("/content/drive/MyDrive/CKD_Thesis/datasets/raw/", str(local_data) + "/"),
        ("/content/df40_local/", str(local_data) + "/"),
    ]

    for fname in CSV_FILES:
        src = csv_dir / fname
        dst = splits_dst / fname
        if not src.is_file():
            logger.warning("  [miss ] %s not in %s — skipping", fname, csv_dir)
            continue
        df = pd.read_csv(src, low_memory=False)
        if "face_path" not in df.columns:
            logger.error("  [bad  ] %s missing 'face_path' column", fname)
            continue
        for old, new in rewrite_pairs:
            df["face_path"] = df["face_path"].str.replace(old, new, regex=False)
        df.to_csv(dst, index=False)
        # Quick existence check on first 50 rows
        existing = sum(1 for p in df["face_path"].head(50) if Path(p).is_file())
        logger.info(
            "  [csv  ] %s -> %s (%d rows, sampled %d/50 exist on disk)",
            fname,
            dst,
            len(df),
            existing,
        )


def step4_write_config(
    *, mirror_root: Path, repo_root: Path, logger
) -> Path:
    """Generate configs/macbook.yaml derived from configs/default.yaml.

    Only the ``paths.drive_root`` field is rewritten to point at the
    MacBook mirror; everything else (student arch, training hparams,
    edge modes) is inherited unchanged.
    """
    src = repo_root / "configs" / "default.yaml"
    if not src.is_file():
        raise FileNotFoundError(f"default.yaml not found: {src}")
    dst = repo_root / "configs" / "macbook.yaml"

    text = src.read_text(encoding="utf-8")
    new_text = text.replace(
        'drive_root: "/content/drive/MyDrive/CKD_Thesis"',
        f'drive_root: "{mirror_root}"',
    )
    if new_text == text:
        logger.warning(
            "drive_root line not found verbatim in default.yaml — generated "
            "macbook.yaml may need manual review"
        )
    dst.write_text(new_text, encoding="utf-8")
    logger.info("Step 4/4: wrote %s (drive_root -> %s)", dst, mirror_root)
    return dst


def step5_verify(*, mirror_root: Path, logger) -> bool:
    """Sanity check: every test CSV resolves face_path correctly."""
    splits_dir = mirror_root / "datasets" / "splits"
    ok = True
    logger.info("Verifying face_path resolution on test CSVs...")
    for fname in CSV_FILES:
        csv = splits_dir / fname
        if not csv.is_file():
            logger.error("  [skip ] %s missing — verification incomplete", fname)
            ok = False
            continue
        df = pd.read_csv(csv, low_memory=False, usecols=["face_path"]).head(100)
        existing = sum(1 for p in df["face_path"] if Path(p).is_file())
        ratio = existing / max(1, len(df))
        symbol = "OK  " if ratio == 1.0 else "WARN" if ratio > 0.9 else "FAIL"
        logger.info(
            "  [%s] %s : %d/%d sampled face_paths exist",
            symbol,
            fname,
            existing,
            len(df),
        )
        if ratio < 0.95:
            ok = False
    return ok


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Setup local mirror on MacBook for CKD edge evaluation"
    )
    p.add_argument(
        "--zip-dir",
        required=True,
        type=Path,
        help="Folder containing DF40 *.zip files (download from Drive first)",
    )
    p.add_argument(
        "--ckpt-dir",
        required=True,
        type=Path,
        help="Folder containing 9 student checkpoint subdirs (gen{1,2,3}_*_seed{0,1,2})",
    )
    p.add_argument(
        "--csv-dir",
        required=True,
        type=Path,
        help="Folder containing gen{1,2,3}_test.csv split files",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path.home() / "ckd-edge",
        help="Root for extracted data + mirror layout (default: ~/ckd-edge)",
    )
    p.add_argument(
        "--generations",
        default="all",
        help="Comma-separated list of generations to extract, or 'all'",
    )
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def _run(args: argparse.Namespace) -> int:
    logger = get_logger(
        "edge.setup_macbook",
        log_file="runs/setup_macbook_mirror.log",
    )

    gens = (
        ["gen1", "gen2", "gen3"]
        if args.generations == "all"
        else [g.strip() for g in args.generations.split(",") if g.strip()]
    )

    zip_dir = args.zip_dir.expanduser().resolve()
    ckpt_dir = args.ckpt_dir.expanduser().resolve()
    csv_dir = args.csv_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    local_data = output_root
    mirror_root = output_root / "mirror"

    logger.info("Configuration:")
    logger.info("  zip_dir     = %s", zip_dir)
    logger.info("  ckpt_dir    = %s", ckpt_dir)
    logger.info("  csv_dir     = %s", csv_dir)
    logger.info("  output_root = %s", output_root)
    logger.info("  mirror_root = %s", mirror_root)
    logger.info("  generations = %s", gens)
    logger.info("  resume      = %s", args.resume)

    output_root.mkdir(parents=True, exist_ok=True)

    step1_extract_zips(
        zip_dir=zip_dir,
        local_data=local_data,
        generations=gens,
        resume=args.resume,
        logger=logger,
    )
    step2_copy_checkpoints(
        ckpt_dir=ckpt_dir, mirror_root=mirror_root, resume=args.resume, logger=logger
    )
    step3_rewrite_csvs(
        csv_dir=csv_dir,
        local_data=local_data,
        mirror_root=mirror_root,
        logger=logger,
    )
    config_path = step4_write_config(
        mirror_root=mirror_root, repo_root=_REPO_ROOT, logger=logger
    )
    ok = step5_verify(mirror_root=mirror_root, logger=logger)

    logger.info("=" * 60)
    logger.info("SETUP %s", "COMPLETE" if ok else "COMPLETED WITH WARNINGS")
    logger.info("=" * 60)
    logger.info("Use: --config %s in subsequent scripts/notebooks", config_path)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_run(_parse_args()))
