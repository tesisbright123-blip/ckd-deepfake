"""Setup local mirror for avoiding Drive FUSE bottleneck.

CLI entrypoint. Not imported by other code.

Background — Why this exists
----------------------------
Google Drive FUSE mount in Colab serves single large files at ~30-70 MB/s,
but each per-file open/read/close is an HTTP API call (~50-400ms each).
DF40 produces ~700K small face crops (50-100KB each) — at production scale
the per-file cost dominates: PyTorch DataLoader achieves only **2.5 fps** vs
the **368 fps** the same code gets when reading from local NVMe. That's a
**147x slowdown** purely from FUSE overhead.

Things that DO NOT work (already tested, do not repeat):
  - tar archive ON Drive: double FUSE penalty, hung after 56 minutes
  - rsync Drive -> local: 115 KB/s effective, 50GB takes 5+ days
  - Multi-threaded copy: 18 files/sec, ETA 4h (too slow at A100 cost)

The working approach codified here:
  1. Copy 19 large zip archives from ``df40_zip_backup/`` to local: ~9 min
     (zips already exist on Drive from a previous setup)
  2. Extract locally on NVMe: ~4 min for ~461K files
  3. Patch missing uniface frames directly from Drive (small count, FUSE OK)
  4. Rewrite split CSVs so ``face_path`` points to local NVMe paths
  5. Symlink soft labels & teacher checkpoints from Drive (small files)
  6. Generate ``configs/local.yaml`` with ``drive_root: /content/ckd_local``
  7. Verify a sample of paths actually exist locally

After running this once per Colab session, every script (03/04/05/06/07)
can be invoked with ``--config configs/local.yaml`` (or ``--splits-dir``)
and read images from local NVMe at full GPU-pipeline speed.

Usage:
    # Setup gen1 only (initial distillation)
    python scripts/00_setup_local_mirror.py --generations gen1

    # Setup gen1 + gen2 (continual gen2 needs both)
    python scripts/00_setup_local_mirror.py --generations gen1,gen2

    # Setup all 3 (continual gen3 needs all)
    python scripts/00_setup_local_mirror.py --generations all

    # Resume mode: skip steps already completed (idempotent)
    python scripts/00_setup_local_mirror.py --generations all --resume

Reads:
    {drive}/datasets/raw/df40_zip_backup/*.zip      (technique zips)
    {drive}/datasets/raw/df40_zip_backup/*real*.zip (real ZIPs)
    {drive}/datasets/raw/df40/uniface/...           (patch source for missing frames)
    {drive}/datasets/splits/{gen}_{split}.csv       (CSVs to rewrite)
    {drive}/soft_labels/{gen}/...                   (small files; symlinked)
    {drive}/checkpoints/teachers/*.pth              (small files; symlinked)
    configs/default.yaml                            (template for local.yaml)

Writes:
    /content/df40_local_zips/*.zip                  (intermediate, ephemeral)
    /content/df40_local/df40/<technique>/...        (extracted images)
    /content/df40_local/df40_real/{ff,cdf}_real/... (extracted real)
    /content/ckd_local/datasets/splits/*.csv        (path-rewritten)
    /content/ckd_local/soft_labels                  (symlink)
    /content/ckd_local/checkpoints/teachers         (symlink)
    /content/ckd_local/checkpoints/students/        (output dir, empty)
    /content/ckd_local/results/raw/                 (output dir, empty)
    configs/local.yaml                              (drive_root override)
    Per-step marker files                           (for ``--resume``)
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.utils.config import load_config  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

# ----------------------------------------------------------------------- #
#  Constants
# ----------------------------------------------------------------------- #

# Mapping technique → required zip stems per generation. Names must match
# the on-disk zip filenames in ``df40_zip_backup/``.
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

# Real ZIPs always needed (regardless of which gens we're setting up).
# Names match the original DF40 release filenames (don't get renamed).
REAL_ZIPS: dict[str, str] = {
    "ff_real": "FaceForensics++_real_data_for_DF40.zip",
    "cdf_real": "Celeb-DF-v2_real_data_for_DF40.zip",
}

# Zip name → target subdir under LOCAL_DATA. Default for technique zips:
# ``df40/<stem>/`` (technique stem becomes folder name, matching CSV paths).
# Special cases (real zips) go to ``df40_real/{label}/``.
_EXTRACT_OVERRIDES: dict[str, str] = {
    "FaceForensics++_real_data_for_DF40.zip": "df40_real/ff_real/",
    "Celeb-DF-v2_real_data_for_DF40.zip": "df40_real/cdf_real/",
}

# Local paths
LOCAL_ZIPS = Path("/content/df40_local_zips")
LOCAL_DATA = Path("/content/df40_local")
LOCAL_MIRROR = Path("/content/ckd_local")

# Marker filename written into target dirs after extraction succeeds — lets
# ``--resume`` skip already-done steps.
_EXTRACT_MARKER = ".extracted_ok"


# ----------------------------------------------------------------------- #
#  Helpers
# ----------------------------------------------------------------------- #

def _required_zip_filenames(generations: list[str]) -> list[str]:
    """Return list of zip filenames needed for the requested generations."""
    needed: set[str] = set()
    for g in generations:
        if g not in TECHNIQUES_PER_GEN:
            raise ValueError(f"Unknown generation: {g!r}")
        for stem in TECHNIQUES_PER_GEN[g]:
            needed.add(f"{stem}.zip")
    needed.update(REAL_ZIPS.values())
    return sorted(needed)


def _extract_target(zip_name: str) -> Path:
    """Where this zip should extract to under LOCAL_DATA.

    Technique zips (e.g. ``blendface.zip``) ship with a redundant root
    folder named after the technique — internal layout is
    ``blendface/frames/<vid>/000.png``. So we extract them to
    ``/content/df40_local/df40/`` (parent) and the zip's root becomes
    the technique folder, yielding ``df40/blendface/frames/<vid>/000.png``
    which matches the CSV ``face_path`` after rewrite.

    Real zips (FaceForensics++, Celeb-DF-v2) have their own override
    targets in ``_EXTRACT_OVERRIDES`` — ``df40_real/ff_real/`` and
    ``df40_real/cdf_real/`` — and their internal structure starts with
    ``FaceForensics++/...`` / ``Celeb-DF-v2/...`` so the resulting paths
    match CSV expectations.
    """
    if zip_name in _EXTRACT_OVERRIDES:
        return LOCAL_DATA / _EXTRACT_OVERRIDES[zip_name]
    return LOCAL_DATA / "df40"


# ----------------------------------------------------------------------- #
#  Step 1 — Copy zips from Drive to local NVMe (large file = fast)
# ----------------------------------------------------------------------- #

def step1_copy_zips(
    *, drive_root: Path, generations: list[str], resume: bool, logger
) -> None:
    src_dir = drive_root / "datasets" / "raw" / "df40_zip_backup"
    if not src_dir.is_dir():
        raise FileNotFoundError(
            f"Zip backup dir not found: {src_dir}. "
            "Expected df40_zip_backup/ on Drive (created in earlier sessions)."
        )

    LOCAL_ZIPS.mkdir(parents=True, exist_ok=True)
    needed = _required_zip_filenames(generations)
    logger.info(
        "Step 1/6: copying %d zips from %s to %s", len(needed), src_dir, LOCAL_ZIPS
    )

    for zname in needed:
        src = src_dir / zname
        dst = LOCAL_ZIPS / zname
        if not src.is_file():
            logger.error("Zip not found on Drive: %s", src)
            continue
        if resume and dst.is_file() and dst.stat().st_size == src.stat().st_size:
            logger.info("  [skip ] %s already copied (%.2f GB)",
                        zname, dst.stat().st_size / 1e9)
            continue
        shutil.copyfile(src, dst)
        logger.info("  [copy ] %s (%.2f GB)", zname, dst.stat().st_size / 1e9)


# ----------------------------------------------------------------------- #
#  Step 2 — Extract zips to LOCAL_DATA with correct subdir mapping
# ----------------------------------------------------------------------- #

def _per_zip_marker(zip_name: str) -> Path:
    """Per-zip marker path so resume can skip individual zips even when
    multiple zips share the same extract target dir (technique zips all
    extract into ``LOCAL_DATA/df40/``).

    Marker convention:
      - Technique zip ``<stem>.zip`` -> marker at
        ``LOCAL_DATA / df40 / <stem> / .extracted_ok`` (sits inside the
        technique folder produced by the zip's redundant root).
      - Real zip with override target -> marker inside the override dir
        (one zip per dir, so dir-level marker is fine).
    """
    if zip_name in _EXTRACT_OVERRIDES:
        return LOCAL_DATA / _EXTRACT_OVERRIDES[zip_name] / _EXTRACT_MARKER
    stem = Path(zip_name).stem
    return LOCAL_DATA / "df40" / stem / _EXTRACT_MARKER


def step2_extract_local(
    *, generations: list[str], resume: bool, logger,
    delete_zip_after: bool = True,
) -> None:
    """Extract each zip and write a per-zip marker.

    With ``delete_zip_after=True`` (default), each zip's local copy is
    removed after a successful extract. This frees ~50 GB on the Colab
    /content disk so we don't run out of space mid-pipeline.
    """
    LOCAL_DATA.mkdir(parents=True, exist_ok=True)
    needed = _required_zip_filenames(generations)
    logger.info("Step 2/6: extracting %d zips to %s", len(needed), LOCAL_DATA)

    for zname in needed:
        src = LOCAL_ZIPS / zname
        target = _extract_target(zname)
        marker = _per_zip_marker(zname)
        if resume and marker.is_file():
            logger.info("  [skip ] %s already extracted (marker at %s)", zname, marker)
            # Free local zip even if we skipped extract, to save disk.
            if delete_zip_after and src.is_file():
                src.unlink()
                logger.info("  [free ] removed local zip copy %s", src)
            continue
        if not src.is_file():
            logger.warning(
                "  [skip ] %s missing in local zips dir (no extracted marker either)",
                zname,
            )
            continue
        size_gb = src.stat().st_size / 1e9
        if Path(zname).stem == "uniface":
            # uniface.zip ships with a FLAT internal layout (``frames/...`` at
            # top level) instead of the ``<technique>/frames/...`` layout every
            # other zip uses. Extracting it to ``df40/`` (parent) would land the
            # frames at ``df40/frames/`` while the CSV expects
            # ``df40/uniface/frames/`` — making all uniface rows "missing" and
            # forcing step 3 to re-copy them from the Drive ``df40/uniface/``
            # folder. We normalize the layout here so the mirror is fully
            # self-contained from the zip backup alone (no Drive df40/ needed).
            logger.info("  [ext  ] %s (%.2f GB) -> df40/uniface (layout-normalized)",
                        zname, size_gb)
            _extract_uniface_normalized(src, logger=logger)
        else:
            target.mkdir(parents=True, exist_ok=True)
            logger.info("  [ext  ] %s (%.2f GB) -> %s", zname, size_gb, target)
            with zipfile.ZipFile(src) as zf:
                zf.extractall(target)
        # Write marker AFTER successful extract; relies on the zip's
        # redundant root folder existing for technique zips.
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(datetime.now().isoformat())
        if delete_zip_after:
            src.unlink()
            logger.info("  [free ] removed local zip %s after extract", src)


def _build_needed_df40_members(
    *, drive_root: Path, generations: list[str], logger
) -> set[str]:
    """Collect the set of ``df40/``-relative member paths the CSVs reference.

    A member path is the portion of ``face_path`` after ``/df40/`` — i.e.
    ``<technique>/frames/<vid>/<file>``. This is exactly the zip member name
    for a normally-rooted technique zip, so intersecting it with a zip's
    namelist tells us which members to extract.

    Real-data rows (``/df40_real/...``) are intentionally excluded — those
    zips are small and always fully extracted.

    Because the DF40 split CSVs already subsample to 32 frames/video, this
    set (~700K members) is far smaller than the full zip contents (~2-4M
    frames) — letting selective extraction fit a 112 GB free-tier disk.
    """
    needed: set[str] = set()
    splits = drive_root / "datasets" / "splits"
    for g in generations:
        for split in ("train", "val", "test"):
            csv = splits / f"{g}_{split}.csv"
            if not csv.is_file():
                logger.warning("  [members] CSV missing: %s", csv)
                continue
            df = pd.read_csv(csv, usecols=["face_path"])
            for fp in df["face_path"].astype(str):
                if "/df40/" in fp:
                    needed.add(fp.split("/df40/", 1)[1])
    logger.info("  [members] %d df40 frames referenced across %s CSVs",
                len(needed), generations)
    return needed


def _extract_zip_selective(
    zip_path: Path, target: Path, needed: set[str], *, logger
) -> int:
    """Extract only the members of ``zip_path`` present in ``needed``.

    Members are matched against the zip's namelist by exact name, so this
    relies on the same path consistency that full extraction already
    depends on (zip root folder == the directory in face_path). If NO member
    matches (a naming surprise), falls back to a full extract so correctness
    is never sacrificed for disk savings.

    Returns the number of members extracted (-1 if the full-extract fallback
    was used).
    """
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        to_extract = [n for n in names if n in needed]
        if not to_extract:
            logger.warning(
                "  [selective] %s: 0 of %d members matched the CSVs — "
                "falling back to FULL extract (correctness over disk).",
                zip_path.name, len(names),
            )
            zf.extractall(target)
            return -1
        for n in to_extract:
            zf.extract(n, target)
    logger.info("  [selective] %s: extracted %d / %d members",
                zip_path.name, len(to_extract), len(names))
    return len(to_extract)


def step12_copy_extract(
    *, drive_root: Path, generations: list[str], resume: bool, logger,
    selective: bool = True,
) -> None:
    """Interleaved copy -> extract -> delete, one zip at a time.

    Replaces the old "copy ALL zips, then extract ALL" flow, which peaked at
    ``base + sum(all zip sizes)`` (~112 GB) and overflowed a free-tier disk.
    Here only ONE zip is on disk at any moment, and (with ``selective=True``)
    only CSV-referenced frames are written — so peak stays well under the
    free-tier 112 GB budget.

    Per-zip handling:
      * uniface.zip  -> layout-normalized FULL extract (small, ~1 GB)
      * real zips    -> FULL extract to df40_real/ (small, ~5 GB total)
      * technique zips -> SELECTIVE extract (only members in the CSVs)
    """
    src_dir = drive_root / "datasets" / "raw" / "df40_zip_backup"
    if not src_dir.is_dir():
        raise FileNotFoundError(
            f"Zip backup dir not found: {src_dir}. "
            "Expected df40_zip_backup/ on Drive (created in earlier sessions)."
        )
    LOCAL_ZIPS.mkdir(parents=True, exist_ok=True)
    LOCAL_DATA.mkdir(parents=True, exist_ok=True)

    needed_members: set[str] = set()
    if selective:
        needed_members = _build_needed_df40_members(
            drive_root=drive_root, generations=generations, logger=logger,
        )
        # Fail LOUD instead of silently full-extracting (and overflowing the
        # disk) if the CSV scan came back empty — that means the split CSVs
        # weren't found or their face_path format is unexpected.
        if not needed_members:
            raise RuntimeError(
                "Selective extraction requested but 0 frames were found in the "
                f"split CSVs under {drive_root}/datasets/splits. Refusing to "
                "fall back to a full extract (it would overflow a free-tier "
                "disk). Check that the gen*_{train,val,test}.csv files exist "
                "and their face_path column contains '/df40/'. To force a full "
                "extract anyway, re-run with --full-extract on a large disk."
            )

    needed_zips = _required_zip_filenames(generations)
    logger.info("Steps 1-2/6: copy+extract %d zips (selective=%s) -> %s",
                len(needed_zips), selective, LOCAL_DATA)

    for zname in needed_zips:
        marker = _per_zip_marker(zname)
        if resume and marker.is_file():
            logger.info("  [skip ] %s already extracted (marker)", zname)
            continue
        src = src_dir / zname
        if not src.is_file():
            logger.error("  Zip not found on Drive: %s", src)
            continue

        local_zip = LOCAL_ZIPS / zname
        size_gb = src.stat().st_size / 1e9
        logger.info("  [copy ] %s (%.2f GB) -> local", zname, size_gb)
        shutil.copyfile(src, local_zip)

        if Path(zname).stem == "uniface":
            logger.info("  [ext  ] %s -> df40/uniface (normalized, full)", zname)
            _extract_uniface_normalized(local_zip, logger=logger)
        elif zname in _EXTRACT_OVERRIDES:
            target = _extract_target(zname)
            target.mkdir(parents=True, exist_ok=True)
            logger.info("  [ext  ] %s -> %s (real, full)", zname, target)
            with zipfile.ZipFile(local_zip) as zf:
                zf.extractall(target)
        else:
            target = LOCAL_DATA / "df40"
            if selective and needed_members:
                _extract_zip_selective(local_zip, target, needed_members, logger=logger)
            else:
                target.mkdir(parents=True, exist_ok=True)
                logger.info("  [ext  ] %s -> %s (full)", zname, target)
                with zipfile.ZipFile(local_zip) as zf:
                    zf.extractall(target)

        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(datetime.now().isoformat())
        local_zip.unlink()
        free_gb = shutil.disk_usage("/content").free / 1e9
        logger.info("  [free ] removed local %s (disk free: %.1f GB)", zname, free_gb)


def _extract_uniface_normalized(zip_path: Path, *, logger) -> None:
    """Extract uniface.zip so its content always lands at ``df40/uniface/``.

    Handles both possible internal layouts:
      * flat   — top-level ``frames/...``         -> move into df40/uniface/
      * nested — top-level ``uniface/frames/...`` -> move df40/uniface intact

    Extraction goes to a temp dir on the same filesystem first, so the final
    relocation is a fast rename, then the temp dir is removed.
    """
    # Start from a clean target. This function is only reached when the
    # per-zip marker is absent (fresh extract needed), so any pre-existing
    # df40/uniface/ content is leftover from a failed prior attempt — clearing
    # it prevents a partial-content + dst-exists "skip" from silently leaving
    # the buffer short some frames while the marker still gets written.
    target = LOCAL_DATA / "df40" / "uniface"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    tmp = LOCAL_DATA / "df40" / "_uniface_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        # Determine the root that actually holds the frames.
        src_root = tmp / "uniface" if (tmp / "uniface").is_dir() else tmp
        moved = 0
        for item in src_root.iterdir():
            dst = target / item.name
            if dst.exists():
                continue
            shutil.move(str(item), str(dst))
            moved += 1
        logger.info("  [uniface] normalized %d top-level entries -> %s", moved, target)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)


# ----------------------------------------------------------------------- #
#  Step 3 — Patch missing uniface frames from Drive
# ----------------------------------------------------------------------- #

def step3_patch_uniface(
    *, drive_root: Path, generations: list[str], logger
) -> None:
    """uniface.zip in df40_zip_backup is incomplete (only ~12K of the ~?? files
    in the original Drive folder). Identify missing frames by reading the gen1
    CSVs (all splits) and copy them directly via FUSE. Small count -> latency OK.
    """
    if "gen1" not in generations:
        return  # uniface only used in gen1
    logger.info("Step 3/6: patching missing uniface frames from Drive")

    drive_prefix = "/content/drive/MyDrive/CKD_Thesis/datasets/raw/"
    local_prefix = str(LOCAL_DATA) + "/"

    # Read all gen1 CSVs; collect uniface face_paths
    needed_drive_paths: set[str] = set()
    for split in ("train", "val", "test"):
        csv = drive_root / "datasets" / "splits" / f"gen1_{split}.csv"
        if not csv.is_file():
            logger.warning("  CSV missing: %s — skipping uniface patch lookup for split", csv)
            continue
        df = pd.read_csv(csv, usecols=["face_path", "technique"])
        df = df[df["technique"] == "uniface"]
        needed_drive_paths.update(df["face_path"].tolist())

    if not needed_drive_paths:
        logger.info("  [patch] no uniface paths found in gen1 CSVs — skipping")
        return

    # Identify which are missing locally
    missing: list[tuple[str, str]] = []
    for drive_path in needed_drive_paths:
        local_path = drive_path.replace(drive_prefix, local_prefix)
        if not Path(local_path).is_file():
            missing.append((drive_path, local_path))

    if not missing:
        logger.info("  [patch] uniface complete (%d files), no patching needed",
                    len(needed_drive_paths))
        return

    # If the local layout-normalization in step 2 worked, ``missing`` is empty
    # and we never reach here. If we DO reach here, the local mirror is short
    # some frames and we'd normally backfill from the Drive ``df40/uniface/``
    # folder. But that folder may have been deleted to free Drive storage —
    # detect that and fail loudly with a clear instruction instead of emitting
    # one OSError per missing file.
    drive_uniface_dir = drive_root / "datasets" / "raw" / "df40" / "uniface"
    if not drive_uniface_dir.is_dir():
        logger.error(
            "  [patch] %d uniface frames missing locally AND the Drive backfill "
            "folder %s does not exist. The step-2 layout normalization should "
            "have prevented this — check that uniface.zip is present in "
            "df40_zip_backup and re-run without --resume to force a clean "
            "extract. (Do NOT delete df40_zip_backup.)",
            len(missing),
            drive_uniface_dir,
        )
        return

    logger.info("  [patch] %d uniface files missing locally — copying from Drive",
                len(missing))

    def _copy_one(pair: tuple[str, str]) -> str:
        drive_path, local_path = pair
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(drive_path, local_path)
        return drive_path

    n_done = 0
    n_failed = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_copy_one, p) for p in missing]
        for fut in as_completed(futures):
            try:
                fut.result()
                n_done += 1
            except OSError as exc:
                logger.warning("  [patch] copy failed: %s", exc)
                n_failed += 1
    logger.info("  [patch] done: %d copied, %d failed (of %d total)",
                n_done, n_failed, len(missing))


# ----------------------------------------------------------------------- #
#  Step 4 — Create local mirror at LOCAL_MIRROR (rewritten CSVs + symlinks)
# ----------------------------------------------------------------------- #

def step4_create_mirror(
    *, drive_root: Path, generations: list[str], logger
) -> None:
    LOCAL_MIRROR.mkdir(parents=True, exist_ok=True)
    logger.info("Step 4/6: building local mirror at %s", LOCAL_MIRROR)

    # 4a. Rewrite split CSVs (Drive paths → local paths)
    splits_local = LOCAL_MIRROR / "datasets" / "splits"
    splits_local.mkdir(parents=True, exist_ok=True)

    drive_prefix = "/content/drive/MyDrive/CKD_Thesis/datasets/"
    local_prefix = "/content/df40_local/"

    for g in generations:
        for split in ("train", "val", "test"):
            src = drive_root / "datasets" / "splits" / f"{g}_{split}.csv"
            dst = splits_local / f"{g}_{split}.csv"
            if not src.is_file():
                logger.warning("  CSV missing: %s — skipping rewrite", src)
                continue
            df = pd.read_csv(src)
            df["face_path"] = df["face_path"].str.replace(
                drive_prefix + "raw/", local_prefix, regex=False,
            )
            df.to_csv(dst, index=False)
            logger.info("  [rewrite] %s (%d rows)", dst.name, len(df))

    # 4b. Symlink soft labels (small files, no FUSE bottleneck during read)
    sl_local = LOCAL_MIRROR / "soft_labels"
    sl_local.mkdir(parents=True, exist_ok=True)
    for g in generations:
        sl_drive = drive_root / "soft_labels" / g
        if not sl_drive.is_dir():
            logger.info("  [symlink] soft_labels/%s missing on Drive (run script 03 first)", g)
            continue
        sl_link = sl_local / g
        if sl_link.is_symlink() or sl_link.exists():
            sl_link.unlink()
        sl_link.symlink_to(sl_drive)
        logger.info("  [symlink] soft_labels/%s -> %s", g, sl_drive)

    # 4c. Symlink teacher checkpoints (small files)
    ck_local_teachers = LOCAL_MIRROR / "checkpoints" / "teachers"
    ck_local_teachers.parent.mkdir(parents=True, exist_ok=True)
    if ck_local_teachers.is_symlink() or ck_local_teachers.exists():
        ck_local_teachers.unlink() if ck_local_teachers.is_symlink() else shutil.rmtree(ck_local_teachers)
    ck_local_teachers.symlink_to(drive_root / "checkpoints" / "teachers")
    logger.info("  [symlink] checkpoints/teachers -> Drive")

    # 4d. Output dirs (will be filled by training scripts)
    (LOCAL_MIRROR / "checkpoints" / "students").mkdir(parents=True, exist_ok=True)
    (LOCAL_MIRROR / "results" / "raw").mkdir(parents=True, exist_ok=True)
    (LOCAL_MIRROR / "runs").mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------- #
#  Step 5 — Generate configs/local.yaml
# ----------------------------------------------------------------------- #

def step5_create_local_config(*, logger) -> None:
    src = _REPO_ROOT / "configs" / "default.yaml"
    dst = _REPO_ROOT / "configs" / "local.yaml"
    if not src.is_file():
        raise FileNotFoundError(f"Template config not found: {src}")
    cfg_text = src.read_text(encoding="utf-8")

    # Simple string substitution — works for the YAML field we're touching.
    drive_marker = 'drive_root: "/content/drive/MyDrive/CKD_Thesis"'
    local_marker = f'drive_root: "{LOCAL_MIRROR}"'
    if drive_marker not in cfg_text:
        logger.warning(
            "Could not find expected drive_root line in default.yaml — "
            "writing local.yaml as-is. Edit manually if drive_root needs override."
        )
    new_cfg = cfg_text.replace(drive_marker, local_marker)

    # Prepend a generated-file warning header
    header = (
        "# AUTO-GENERATED by scripts/00_setup_local_mirror.py\n"
        "# DO NOT EDIT BY HAND — re-run setup script to regenerate\n"
        "# Differs from default.yaml only in paths.drive_root\n\n"
    )
    dst.write_text(header + new_cfg, encoding="utf-8")
    logger.info("Step 5/6: wrote %s", dst)


# ----------------------------------------------------------------------- #
#  Step 6 — Verify a sample of rewritten paths actually exist locally
# ----------------------------------------------------------------------- #

def step6_verify(*, generations: list[str], logger) -> bool:
    splits_local = LOCAL_MIRROR / "datasets" / "splits"
    logger.info("Step 6/6: verifying sampled face_paths exist locally")
    all_ok = True
    for g in generations:
        csv = splits_local / f"{g}_train.csv"
        if not csv.is_file():
            logger.warning("  %s missing — cannot verify gen %s", csv, g)
            all_ok = False
            continue
        df = pd.read_csv(csv, usecols=["face_path"]).head(50)
        existing = sum(1 for p in df["face_path"] if Path(p).is_file())
        ratio = existing / max(len(df), 1)
        symbol = "OK" if ratio == 1.0 else "WARN" if ratio > 0.9 else "FAIL"
        logger.info(
            "  [%s] %s_train: %d/%d sampled face_paths exist (%.0f%%)",
            symbol, g, existing, len(df), ratio * 100,
        )
        if ratio < 0.95:
            all_ok = False
            logger.warning(
                "  ↳ likely missing files — check uniface patch step or "
                "extraction completeness for gen %s", g,
            )
    return all_ok


# ----------------------------------------------------------------------- #
#  Disk-space check
# ----------------------------------------------------------------------- #

def _check_disk_space(min_free_gb: float = 70.0) -> tuple[float, float]:
    """Return (free_gb, total_gb) for /content. Warn if below threshold."""
    stat = shutil.disk_usage("/content")
    free_gb = stat.free / 1e9
    total_gb = stat.total / 1e9
    return free_gb, total_gb


# ----------------------------------------------------------------------- #
#  Main
# ----------------------------------------------------------------------- #

def _run(args: argparse.Namespace) -> int:
    logger = get_logger(
        "setup_local_mirror",
        log_file="runs/setup_local_mirror.log",
    )

    # Resolve generations
    if args.generations.lower() == "all":
        gens = list(TECHNIQUES_PER_GEN.keys())
    else:
        gens = [g.strip() for g in args.generations.split(",") if g.strip()]
        for g in gens:
            if g not in TECHNIQUES_PER_GEN:
                logger.error("Unknown generation: %s", g)
                return 2

    # Disk-space pre-check
    free_gb, total_gb = _check_disk_space()
    logger.info(
        "Local /content disk: %.1f GB free of %.1f GB total", free_gb, total_gb
    )
    if free_gb < 70.0:
        logger.warning(
            "Low disk space — setup needs ~60-70 GB. If you hit ENOSPC, "
            "run with fewer generations (e.g. --generations gen1) and "
            "re-setup later for gen2/gen3."
        )

    # Load config to discover Drive root
    cfg = load_config(args.config)
    drive_root = Path(cfg["paths"]["drive_root"])
    if not drive_root.is_dir():
        logger.error(
            "Drive root not found: %s — is Google Drive mounted?", drive_root,
        )
        return 1

    logger.info(
        "=== Setup local mirror for generations %s (drive_root=%s) ===",
        gens, drive_root,
    )

    try:
        step12_copy_extract(
            drive_root=drive_root, generations=gens,
            resume=args.resume, logger=logger,
            selective=not args.full_extract,
        )
        step3_patch_uniface(drive_root=drive_root, generations=gens,
                             logger=logger)
        step4_create_mirror(drive_root=drive_root, generations=gens,
                             logger=logger)
        step5_create_local_config(logger=logger)
        all_ok = step6_verify(generations=gens, logger=logger)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Setup failed: %s", exc)
        return 1

    logger.info("=== SETUP COMPLETE ===")
    logger.info("  Local data:        %s", LOCAL_DATA)
    logger.info("  Local mirror:      %s", LOCAL_MIRROR)
    logger.info("  Generated config:  configs/local.yaml")
    logger.info(
        "  Run subsequent scripts with --config configs/local.yaml "
        "(or pass --splits-dir %s/datasets/splits)",
        LOCAL_MIRROR,
    )
    return 0 if all_ok else 1


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Setup local NVMe mirror of DF40 dataset to bypass Google Drive "
            "FUSE bottleneck. Run once per Colab session before training."
        ),
    )
    p.add_argument(
        "--generations",
        default="gen1",
        help=(
            "Comma-separated list of generations to set up "
            "(e.g. 'gen1,gen2'), or 'all' for gen1+gen2+gen3."
        ),
    )
    p.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Base config to read drive_root from (we never write to this file).",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip steps already completed. Marker files (.extracted_ok) and "
            "matching file sizes are checked. Always safe to pass — re-running "
            "without --resume forces full re-do."
        ),
    )
    p.add_argument(
        "--full-extract",
        action="store_true",
        help=(
            "Extract EVERY frame from each zip instead of only the "
            "CSV-referenced ones. Default is selective extraction (only frames "
            "the split CSVs use), which fits a 112 GB free-tier disk. Pass this "
            "only on a large disk (Colab Pro+/local) when you want the full "
            "DF40 frames on disk."
        ),
    )
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(_run(_parse_args()))
