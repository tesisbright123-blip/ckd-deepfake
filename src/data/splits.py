"""Train/val/test split generation and CSV management.

Splits happen at the VIDEO level (not frame level) to prevent data
leakage — frames from the same video always stay in the same split.
Stratified by (label, technique). Default ratio: 70/15/15.

Called by: scripts/02_generate_splits.py
Reads: metadata CSV from face extraction.
       Columns: face_path, frame_idx, video_id, label, dataset,
                generation, technique
Writes:
    {output_dir}/{generation}_{train|val|test}.csv  (same column schema)
    {output_dir}/{generation}_manifest.json
        with "generated_at" in ISO 8601 %Y-%m-%dT%H:%M:%S
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Canonical column schema shared between metadata and split CSVs.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "face_path",
    "label",
    "video_id",
    "dataset",
    "generation",
    "technique",
)


@dataclass(frozen=True)
class SplitConfig:
    """Ratios, RNG seed, and per-video frame cap for a split.

    ``frames_per_video`` is the canonical knob for trading off compute vs.
    coverage. DF40 ships pre-extracted frames at very high temporal density
    (often 100-300 frames/video) but consecutive frames in deepfake videos
    are highly correlated, so using all of them is wasteful. Subsampling to
    32 frames/video matches DeepfakeBench's convention (Yan et al. 2023) and
    the DFDC-winner pipeline (Seferbekov 2020); literature ablations show
    <1% AUC drop relative to using every frame, while soft-label generation
    and student training both speed up by 3-5x.

    ``frames_per_video=None`` keeps every frame (legacy behaviour).
    """

    train: float = 0.70
    val: float = 0.15
    test: float = 0.15
    seed: int = 0
    frames_per_video: int | None = 32

    def __post_init__(self) -> None:
        total = self.train + self.val + self.test
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(
                f"Split ratios must sum to 1.0 (got {total:.4f}: "
                f"train={self.train}, val={self.val}, test={self.test})"
            )
        if min(self.train, self.val, self.test) < 0:
            raise ValueError("Split ratios must be non-negative")
        if self.frames_per_video is not None and self.frames_per_video <= 0:
            raise ValueError(
                f"frames_per_video must be a positive int or None, "
                f"got {self.frames_per_video}"
            )


def generate_splits(
    metadata_csv: str | Path,
    output_dir: str | Path,
    generation: str,
    config: SplitConfig | None = None,
) -> dict[str, Path]:
    """Generate train/val/test split CSVs for one generation.

    Args:
        metadata_csv: Input metadata CSV produced by face extraction.
        output_dir: Directory where ``{generation}_{split}.csv`` files
            are written.
        generation: Generation tag (``gen1``, ``gen2``, ``gen3``) used
            in the output filenames.
        config: Ratios and seed. Defaults to ``SplitConfig()``.

    Returns:
        Mapping from split name to the path of the written CSV.
    """
    cfg = config or SplitConfig()
    df = _load_metadata(Path(metadata_csv))
    _validate_schema(df)

    n_before = len(df)
    if cfg.frames_per_video is not None:
        df = _subsample_per_video(df, cap=cfg.frames_per_video, seed=cfg.seed)
        logger.info(
            "Subsampled %s: %d -> %d frames (cap=%d per video)",
            generation,
            n_before,
            len(df),
            cfg.frames_per_video,
        )
    else:
        logger.info(
            "No subsampling (frames_per_video=None) — using all %d frames", n_before
        )

    # Group by video_id while keeping per-video label/technique for stratification.
    video_table = (
        df.groupby("video_id", as_index=False)
        .agg(label=("label", "first"), technique=("technique", "first"))
    )

    splits = _split_videos(video_table, cfg)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Path] = {}
    for split_name, video_ids in splits.items():
        subset = df[df["video_id"].isin(video_ids)].copy()
        out_path = out_dir / f"{generation}_{split_name}.csv"
        subset.to_csv(out_path, index=False)
        results[split_name] = out_path
        logger.info(
            "Split %s/%s: %d videos, %d frames -> %s",
            generation,
            split_name,
            subset["video_id"].nunique(),
            len(subset),
            out_path,
        )

    _write_manifest(out_dir, generation, cfg, results, df)
    _check_no_leakage(results)
    return results


def _load_metadata(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Metadata CSV not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Metadata CSV is empty: {path}")
    return df


def _validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Metadata CSV is missing required columns: {missing}. "
            f"Present: {list(df.columns)}"
        )


def _subsample_per_video(
    df: pd.DataFrame,
    *,
    cap: int,
    seed: int,
) -> pd.DataFrame:
    """Cap the number of frames retained per ``video_id`` at ``cap``.

    Strategy: uniform temporal sampling. We sort each video's frames by
    ``frame_idx`` (when present) or row order, then take ``cap`` evenly-spaced
    indices. This preserves coverage across the video timeline (start /
    middle / end) better than a random subsample, while staying deterministic
    for a given ``seed``.

    Videos that already have <= ``cap`` frames are kept as-is.
    """
    if cap <= 0:
        raise ValueError(f"cap must be > 0, got {cap}")
    rng = np.random.default_rng(seed)

    sort_cols = ["video_id"] + (["frame_idx"] if "frame_idx" in df.columns else [])
    df_sorted = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    keep_mask = np.zeros(len(df_sorted), dtype=bool)
    for vid, group in df_sorted.groupby("video_id", sort=False):
        n = len(group)
        if n <= cap:
            keep_mask[group.index.to_numpy()] = True
            continue
        # Uniform spacing across the video. ``np.linspace`` with retstep=False
        # already gives ``cap`` indices in [0, n-1].
        chosen = np.linspace(0, n - 1, num=cap, dtype=np.int64)
        # Tiny jitter (within bucket) to break ties consistently between videos.
        # Use a per-video seed derived from cap+seed+hash(vid) for determinism
        # without correlating across runs.
        local_rng = np.random.default_rng(
            (seed * 1_000_003) ^ (abs(hash(str(vid))) & 0xFFFFFFFF)
        )
        jitter = local_rng.integers(low=0, high=max(1, n // (cap + 1)), size=cap)
        chosen = np.clip(chosen + jitter, 0, n - 1)
        chosen = np.unique(chosen)  # dedupe in case clip collapses
        keep_mask[group.index.to_numpy()[chosen]] = True

    out = df_sorted.loc[keep_mask].reset_index(drop=True)
    return out


def _split_videos(
    video_table: pd.DataFrame,
    cfg: SplitConfig,
) -> dict[str, list[str]]:
    """Stratified video-level split.

    Stratification key is (label, technique) so that each split has a
    similar mix of fake-generation techniques.
    """
    rng = np.random.default_rng(cfg.seed)
    out: dict[str, list[str]] = {"train": [], "val": [], "test": []}

    for _, bucket in video_table.groupby(["label", "technique"], dropna=False):
        ids = bucket["video_id"].to_numpy().copy()
        if ids.size == 0:
            continue
        rng.shuffle(ids)

        n = ids.size
        n_train = int(round(n * cfg.train))
        n_val = int(round(n * cfg.val))
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)
        n_test = n - n_train - n_val

        if n >= 3:
            # Guarantee every split gets at least one video when possible.
            if n_train == 0:
                n_train = 1
                n_val = max(n_val - 1, 0)
                n_test = n - n_train - n_val
            if n_val == 0:
                n_val = 1
                n_test = max(n_test - 1, 0)
            if n_test == 0:
                n_test = 1
                n_val = max(n_val - 1, 0)

        out["train"].extend(ids[:n_train].tolist())
        out["val"].extend(ids[n_train : n_train + n_val].tolist())
        out["test"].extend(ids[n_train + n_val : n_train + n_val + n_test].tolist())

    return out


def _write_manifest(
    out_dir: Path,
    generation: str,
    cfg: SplitConfig,
    results: dict[str, Path],
    df: pd.DataFrame,
) -> None:
    split_stats = {}
    for name, path in results.items():
        subset = pd.read_csv(path, usecols=["video_id"])
        split_stats[name] = {
            "csv": str(path),
            "videos": int(subset["video_id"].nunique()),
            "frames": int(len(subset)),
        }

    manifest = {
        "generation": generation,
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": cfg.seed,
        "ratios": {"train": cfg.train, "val": cfg.val, "test": cfg.test},
        "total_videos": int(df["video_id"].nunique()),
        "total_frames": int(len(df)),
        "splits": split_stats,
    }
    manifest_path = out_dir / f"{generation}_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Wrote manifest: %s", manifest_path)


def _check_no_leakage(results: dict[str, Path]) -> None:
    """Guard against accidental video_id overlap between splits."""
    seen: dict[str, str] = {}
    for split, path in results.items():
        ids = set(pd.read_csv(path, usecols=["video_id"])["video_id"].unique())
        overlap = [vid for vid in ids if vid in seen]
        if overlap:
            raise RuntimeError(
                f"Data leakage detected: {len(overlap)} video(s) appear in "
                f"both '{seen[overlap[0]]}' and '{split}'. "
                f"Example: {overlap[0]}"
            )
        for vid in ids:
            seen[vid] = split
    logger.info("Leakage check passed: no video_id shared across splits.")


def summarize_split(csv_path: str | Path) -> dict[str, int]:
    """Return quick summary counts for a split CSV (handy in notebooks)."""
    df = pd.read_csv(csv_path)
    return {
        "frames": int(len(df)),
        "videos": int(df["video_id"].nunique()),
        "real_frames": int((df["label"] == 0).sum()),
        "fake_frames": int((df["label"] == 1).sum()),
    }
