"""Step 2: Create train/val/test splits (70/15/15 at video level).

CLI entrypoint. Not imported by other code.

Usage:
    python scripts/02_generate_splits.py --config configs/default.yaml --generation gen1
    python scripts/02_generate_splits.py --config configs/default.yaml --generation gen2 --seed 1

Reads:
    YAML config (--config)
    metadata CSV at {drive}/datasets/faces/metadata_<generation>.csv
        columns: face_path, frame_idx, video_id, label, dataset, generation, technique
Writes:
    split CSVs at {drive}/datasets/splits/<generation>_{train|val|test}.csv
        columns: face_path, frame_idx, video_id, label, dataset, generation, technique
    manifest at {drive}/datasets/splits/<generation>_manifest.json
        ``generated_at`` in ISO 8601 ``%Y-%m-%dT%H:%M:%S``
    log file at runs/generate_splits_<generation>.log
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make ``src`` importable when invoked as ``python scripts/02_generate_splits.py``.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.splits import SplitConfig, generate_splits, summarize_split  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402


def _build_split_config(
    generation_cfg: dict, seed: int
) -> SplitConfig:
    """Translate the YAML ``split_ratio`` list into a :class:`SplitConfig`."""
    ratios = generation_cfg.get("split_ratio", [0.70, 0.15, 0.15])
    if len(ratios) != 3:
        raise ValueError(
            f"split_ratio must have exactly 3 values (train, val, test); got {ratios!r}"
        )
    train, val, test = (float(r) for r in ratios)
    return SplitConfig(train=train, val=val, test=test, seed=seed)


def _run(args: argparse.Namespace) -> int:
    logger = get_logger(
        "generate_splits",
        log_file=f"runs/generate_splits_{args.generation}.log",
    )

    cfg = load_config(args.config)
    drive_root = Path(cfg["paths"]["drive_root"])
    generation_cfg = cfg["data"]["generations"][args.generation]

    metadata_csv = (
        Path(args.metadata_csv)
        if args.metadata_csv
        else drive_root / "datasets" / "faces" / f"metadata_{args.generation}.csv"
    )
    if not metadata_csv.is_file():
        logger.error(
            "Metadata CSV not found: %s. Run scripts/01_extract_faces.py first.",
            metadata_csv,
        )
        return 1

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else drive_root / "datasets" / "splits"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    split_cfg = _build_split_config(generation_cfg, seed=args.seed)
    logger.info(
        "Generating splits for %s (train=%.3f val=%.3f test=%.3f seed=%d)",
        args.generation,
        split_cfg.train,
        split_cfg.val,
        split_cfg.test,
        split_cfg.seed,
    )

    results = generate_splits(
        metadata_csv=metadata_csv,
        output_dir=output_dir,
        generation=args.generation,
        config=split_cfg,
    )

    for split_name, path in results.items():
        stats = summarize_split(path)
        logger.info(
            "Summary %s/%s: frames=%d videos=%d real=%d fake=%d",
            args.generation,
            split_name,
            stats["frames"],
            stats["videos"],
            stats["real_frames"],
            stats["fake_frames"],
        )

    logger.info("Done. Splits written to %s", output_dir)
    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate stratified video-level train/val/test splits"
    )
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument(
        "--generation",
        required=True,
        choices=["gen1", "gen2", "gen3"],
        help="Which generation block to split",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for the shuffle inside each (label, technique) stratum",
    )
    p.add_argument(
        "--metadata-csv",
        default=None,
        help="Override metadata CSV path. Default: "
        "{drive}/datasets/faces/metadata_<generation>.csv",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Override split output dir. Default: {drive}/datasets/splits",
    )
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(_run(_parse_args()))
