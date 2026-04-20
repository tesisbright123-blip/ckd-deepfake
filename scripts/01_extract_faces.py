"""Step 1: Face extraction from videos.

CLI entrypoint. Not imported by other code.

Usage:
    python scripts/01_extract_faces.py --config configs/default.yaml --generation gen1
    python scripts/01_extract_faces.py --config configs/default.yaml --generation gen2 --max-videos 50

Reads:
    YAML config (--config)
    .mp4 video files at {drive}/datasets/raw/<dataset>/...
Writes:
    face JPEGs at {drive}/datasets/faces/<dataset>/<video_id>/<frame:04d>.jpg
    metadata CSV at {drive}/datasets/faces/metadata_<generation>.csv
        columns: face_path, frame_idx, video_id, label, dataset, generation, technique
    log file at runs/extract_faces_<generation>.log
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

# Make ``src`` importable when invoked as ``python scripts/01_extract_faces.py``.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.face_extractor import ExtractorConfig, FaceExtractor  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402


# --------------------------------------------------------------------------- #
#  Per-dataset video catalogs
# --------------------------------------------------------------------------- #


VideoRecord = dict[str, Any]  # keys: video_path, video_id, label, technique


def _walk_mp4(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.mp4") if p.is_file())


def catalog_faceforensics_pp(dataset_root: Path) -> list[VideoRecord]:
    """FF++ layout:

    faceforensics_pp/
      original_sequences/youtube/c23/videos/*.mp4            -> real
      manipulated_sequences/<technique>/c23/videos/*.mp4     -> fake
    """
    records: list[VideoRecord] = []

    real_root = dataset_root / "original_sequences" / "youtube" / "c23" / "videos"
    if real_root.is_dir():
        for v in _walk_mp4(real_root):
            records.append(
                {
                    "video_path": v,
                    "video_id": f"real_{v.stem}",
                    "label": 0,
                    "technique": "real",
                }
            )

    manip_root = dataset_root / "manipulated_sequences"
    if manip_root.is_dir():
        for tech_dir in sorted(p for p in manip_root.iterdir() if p.is_dir()):
            vid_dir = tech_dir / "c23" / "videos"
            if not vid_dir.is_dir():
                continue
            technique = tech_dir.name
            for v in _walk_mp4(vid_dir):
                records.append(
                    {
                        "video_path": v,
                        "video_id": f"{technique}_{v.stem}",
                        "label": 1,
                        "technique": technique,
                    }
                )
    return records


def catalog_celeb_df_v2(dataset_root: Path) -> list[VideoRecord]:
    """Celeb-DF-v2 layout:

    celeb_df_v2/
      Celeb-real/*.mp4        -> real
      YouTube-real/*.mp4      -> real
      Celeb-synthesis/*.mp4   -> fake
    """
    records: list[VideoRecord] = []
    real_dirs = ["Celeb-real", "YouTube-real"]
    for name in real_dirs:
        d = dataset_root / name
        if d.is_dir():
            for v in _walk_mp4(d):
                records.append(
                    {
                        "video_path": v,
                        "video_id": f"{name}_{v.stem}",
                        "label": 0,
                        "technique": "real",
                    }
                )
    fake_dir = dataset_root / "Celeb-synthesis"
    if fake_dir.is_dir():
        for v in _walk_mp4(fake_dir):
            records.append(
                {
                    "video_path": v,
                    "video_id": f"synthesis_{v.stem}",
                    "label": 1,
                    "technique": "celeb_df_synthesis",
                }
            )
    return records


def catalog_df40(dataset_root: Path) -> list[VideoRecord]:
    """DF40 layout (subset of diffusion-based techniques):

    df40/<technique>/<real|fake>/*.mp4
    """
    records: list[VideoRecord] = []
    if not dataset_root.is_dir():
        return records
    for tech_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        technique = tech_dir.name
        for split_name, label in (("real", 0), ("fake", 1)):
            split_dir = tech_dir / split_name
            if not split_dir.is_dir():
                continue
            for v in _walk_mp4(split_dir):
                records.append(
                    {
                        "video_path": v,
                        "video_id": f"{technique}_{split_name}_{v.stem}",
                        "label": label,
                        "technique": technique,
                    }
                )
    return records


def catalog_deepfake_eval_2024(dataset_root: Path) -> list[VideoRecord]:
    """DeepFake-Eval-2024 layout (treat ``real`` / ``fake`` subdirs)."""
    records: list[VideoRecord] = []
    for split_name, label in (("real", 0), ("fake", 1)):
        d = dataset_root / split_name
        if d.is_dir():
            for v in _walk_mp4(d):
                records.append(
                    {
                        "video_path": v,
                        "video_id": f"{split_name}_{v.stem}",
                        "label": label,
                        "technique": f"dfeval2024_{split_name}",
                    }
                )
    return records


CATALOG_FNS = {
    "faceforensics_pp_c23": (catalog_faceforensics_pp, "faceforensics_pp"),
    "celeb_df_v2": (catalog_celeb_df_v2, "celeb_df_v2"),
    "df40_diffusion_subset": (catalog_df40, "df40"),
    "deepfake_eval_2024": (catalog_deepfake_eval_2024, "deepfake_eval_2024"),
}


# --------------------------------------------------------------------------- #
#  Orchestration
# --------------------------------------------------------------------------- #


def _collect_records(
    generation_cfg: dict[str, Any], drive_root: Path
) -> list[tuple[str, VideoRecord]]:
    """Walk all datasets for the requested generation."""
    out: list[tuple[str, VideoRecord]] = []
    for dataset_name in generation_cfg["datasets"]:
        if dataset_name not in CATALOG_FNS:
            raise ValueError(
                f"No catalog function registered for dataset '{dataset_name}'. "
                f"Known: {list(CATALOG_FNS)}"
            )
        catalog_fn, dir_name = CATALOG_FNS[dataset_name]
        dataset_root = drive_root / "datasets" / "raw" / dir_name
        records = catalog_fn(dataset_root)
        if not records:
            # Not fatal — maybe the dataset hasn't been downloaded yet.
            continue
        out.extend((dataset_name, r) for r in records)
    return out


def _run(args: argparse.Namespace) -> int:
    logger = get_logger(
        "extract_faces", log_file=f"runs/extract_faces_{args.generation}.log"
    )

    cfg = load_config(args.config)
    drive_root = Path(cfg["paths"]["drive_root"])
    generation_cfg = cfg["data"]["generations"][args.generation]

    records = _collect_records(generation_cfg, drive_root)
    if args.max_videos is not None:
        records = records[: args.max_videos]
    if not records:
        logger.error(
            "No videos found for generation %s under %s. "
            "Did you download the raw datasets?",
            args.generation,
            drive_root / "datasets" / "raw",
        )
        return 1

    logger.info(
        "Found %d videos across %d dataset(s) for %s",
        len(records),
        len(set(ds for ds, _ in records)),
        args.generation,
    )

    output_root = Path(args.output_dir) if args.output_dir else drive_root / "datasets" / "faces"
    output_root.mkdir(parents=True, exist_ok=True)

    extractor_cfg = ExtractorConfig(
        frames_per_video=cfg["data"]["face_extraction"]["frames_per_video"],
        output_size=cfg["data"]["face_extraction"]["output_size"],
        margin=cfg["data"]["face_extraction"]["margin"],
        jpeg_quality=95,
    )
    extractor = FaceExtractor(config=extractor_cfg)

    all_rows: list[dict[str, Any]] = []
    errors = 0
    for dataset_name, rec in tqdm(records, desc=f"Extract {args.generation}"):
        per_dataset_out = output_root / dataset_name
        try:
            rows = extractor.extract_from_video(
                video_path=rec["video_path"],
                output_dir=per_dataset_out,
                video_id=rec["video_id"],
                label=rec["label"],
                dataset=dataset_name,
                generation=args.generation,
                technique=rec["technique"],
            )
            all_rows.extend(rows)
        except Exception as exc:  # noqa: BLE001
            errors += 1
            logger.error(
                "Extraction failed for %s: %s", rec["video_path"], exc
            )

    if not all_rows:
        logger.error("No faces extracted. See log for details.")
        return 2

    metadata_path = output_root / f"metadata_{args.generation}.csv"
    _write_metadata(all_rows, metadata_path)
    detection_rate = _detection_rate(all_rows, records, extractor_cfg.frames_per_video)

    logger.info(
        "Done. videos=%d extracted_frames=%d errors=%d detection_rate=%.3f -> %s",
        len(records),
        len(all_rows),
        errors,
        detection_rate,
        metadata_path,
    )
    return 0


def _write_metadata(rows: Iterable[dict[str, Any]], path: Path) -> None:
    df = pd.DataFrame(rows)
    # Deterministic column order matches src/data/splits.REQUIRED_COLUMNS
    ordered = ["face_path", "frame_idx", "video_id", "label", "dataset", "generation", "technique"]
    df = df[[c for c in ordered if c in df.columns]]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _detection_rate(
    rows: list[dict[str, Any]],
    records: list[tuple[str, VideoRecord]],
    frames_per_video: int,
) -> float:
    expected = len(records) * frames_per_video
    return (len(rows) / expected) if expected > 0 else 0.0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract faces from deepfake videos")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument(
        "--generation",
        required=True,
        choices=["gen1", "gen2", "gen3"],
        help="Which generation block to process",
    )
    p.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Process at most N videos (useful for smoke tests)",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Override {drive}/datasets/faces. Optional.",
    )
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(_run(_parse_args()))
