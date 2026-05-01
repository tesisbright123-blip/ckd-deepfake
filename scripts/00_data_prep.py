"""Robust DF40 download + extract — survives laptop crash, runtime disconnect,
gdrive quota errors. Idempotent: re-running picks up exactly where it left off.

Usage from Colab (or shell):
    # See what's done / missing — quick, no side effects.
    python scripts/00_data_prep.py --action status

    # Download missing ZIPs (resume from last state). Quota errors per-file
    # are logged and skipped, not fatal.
    python scripts/00_data_prep.py --action download

    # Extract every ZIP that hasn't been extracted (atomic; uses a marker
    # file inside each technique folder so partial extracts are detectable
    # and re-tried automatically).
    python scripts/00_data_prep.py --action extract

    # Do all three end-to-end.
    python scripts/00_data_prep.py --action all

State machine per technique
---------------------------
For each ``<technique>``, the on-disk state is one of:

* ``EXTRACTED``      — ``<technique>/.extracted_ok`` marker present.
* ``EXTRACTED_LEGACY`` — folder non-empty, no ZIP, no marker. Almost certainly
  extracted by an earlier (pre-marker) version of this code. We back-fill the
  marker so future runs treat it as ``EXTRACTED``.
* ``PARTIAL``        — folder exists *and* ZIP exists. Means a previous
  extract was interrupted. Wipe folder and re-extract from ZIP.
* ``NEEDS_EXTRACT``  — ZIP exists, no folder. Standard extract path.
* ``NEEDS_DOWNLOAD`` — neither folder nor ZIP. Need to gdown.

Real ZIPs (``ff_real.zip`` / ``cdf_real.zip``) follow the same logic but live
under ``datasets/raw/df40_real/{ff_real,cdf_real}/``.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# --------------------------------------------------------------------------- #
#  Configuration
# --------------------------------------------------------------------------- #
DRIVE_ROOT = Path("/content/drive/MyDrive/CKD_Thesis")
DF40_TRAIN_FOLDER_ID = "1U8meBbqVvmUkc5GD0jxct6xe6Gwk9wKD"
FF_REAL_FILE_ID = "1dHJdS0NZ6wpewbGA5B0PdIBS9gz28pdb"
CDF_REAL_FILE_ID = "1FGZ3aYsF-Yru50rPLoT5ef8-2Nkt4uBw"

EXTRACTED_MARKER = ".extracted_ok"
UNZIP_TIMEOUT_SECONDS = 4 * 60 * 60  # 4h cap per ZIP — pixart needs ~3h on Drive.

# Techniques required by configs/default.yaml -> data.generations.<gen>.techniques.
NEEDED_TECHNIQUES: dict[str, list[str]] = {
    "gen1": [
        "faceswap", "fsgan", "simswap", "inswap", "blendface",
        "uniface", "mobileswap", "e4s", "facedancer",
    ],
    "gen2": [
        "fomm", "facevid2vid", "wav2lip", "MRAA", "one_shot_free",
        "pirender", "tpsm", "lia", "danet", "sadtalker", "mcnet", "heygen",
    ],
    "gen3": [
        "sd2.1", "ddim", "PixArt", "DiT", "SiT", "MidJourney",
        "StyleGAN2", "StyleGAN3", "StyleGANXL", "VQGAN",
        "CollabDiff", "whichfaceisreal",
    ],
}
ALL_NEEDED = sorted(
    {t for ts in NEEDED_TECHNIQUES.values() for t in ts},
    key=str.lower,
)


# --------------------------------------------------------------------------- #
#  Path helpers
# --------------------------------------------------------------------------- #
def df40_root() -> Path:
    return DRIVE_ROOT / "datasets" / "raw" / "df40"


def df40_real_root() -> Path:
    return DRIVE_ROOT / "datasets" / "raw" / "df40_real"


def find_existing_dir(root: Path, name: str) -> Path | None:
    """Case-insensitive lookup of ``name`` directly under ``root``."""
    if not root.is_dir():
        return None
    name_lower = name.lower()
    for p in root.iterdir():
        if p.is_dir() and p.name.lower() == name_lower:
            return p
    return None


def find_existing_zip(root: Path, technique: str) -> Path | None:
    """Case-insensitive lookup of ``<technique>.zip`` directly under ``root``."""
    if not root.is_dir():
        return None
    name_lower = f"{technique.lower()}.zip"
    for p in root.glob("*.zip"):
        if p.name.lower() == name_lower:
            return p
    return None


# --------------------------------------------------------------------------- #
#  Per-technique state machine
# --------------------------------------------------------------------------- #
def technique_state(technique: str, *, base: Path) -> tuple[str, Path | None, Path | None]:
    """Return ``(state, folder, zip)`` for one technique.

    ``state`` is one of the labels documented in the module docstring.
    """
    folder = find_existing_dir(base, technique)
    zp = find_existing_zip(base, technique)

    folder_has_marker = folder is not None and (folder / EXTRACTED_MARKER).is_file()
    folder_non_empty = folder is not None and any(folder.iterdir())

    if folder_has_marker:
        return "EXTRACTED", folder, zp
    if folder is not None and zp is not None:
        return "PARTIAL", folder, zp
    if folder is not None and zp is None and folder_non_empty:
        # Legacy completion (old extract cell, before we added markers).
        # Mark it now so we don't re-extract every time.
        write_marker(folder, source_zip="<legacy>", source_size_bytes=None, legacy=True)
        return "EXTRACTED_LEGACY", folder, None
    if folder is not None and zp is None and not folder_non_empty:
        # Empty leftover folder. Treat as needing download.
        try:
            folder.rmdir()
        except OSError:
            pass
        return "NEEDS_DOWNLOAD", None, None
    if zp is not None:
        return "NEEDS_EXTRACT", None, zp
    return "NEEDS_DOWNLOAD", None, None


def write_marker(
    folder: Path,
    *,
    source_zip: str,
    source_size_bytes: int | None,
    legacy: bool = False,
) -> None:
    payload = {
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "source_zip": source_zip,
        "source_size_bytes": source_size_bytes,
        "legacy_backfill": legacy,
    }
    (folder / EXTRACTED_MARKER).write_text(json.dumps(payload, indent=2))


# --------------------------------------------------------------------------- #
#  Status reporting
# --------------------------------------------------------------------------- #
def status_report() -> dict:
    """Print + return a structured snapshot of current data state."""
    base = df40_root()
    base.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print(f"Status snapshot at {datetime.now(timezone.utc).isoformat()}")
    print(f"Drive root: {DRIVE_ROOT}")
    print("=" * 64)

    summary: dict = {"generations": {}, "real": {}}

    for gen, techs in NEEDED_TECHNIQUES.items():
        per_state: dict[str, list[str]] = {
            "EXTRACTED": [], "EXTRACTED_LEGACY": [], "PARTIAL": [],
            "NEEDS_EXTRACT": [], "NEEDS_DOWNLOAD": [],
        }
        for t in techs:
            st, _, _ = technique_state(t, base=base)
            per_state[st].append(t)
        done = len(per_state["EXTRACTED"]) + len(per_state["EXTRACTED_LEGACY"])
        print(f"\n[{gen}]  {done}/{len(techs)} ready")
        for st, names in per_state.items():
            if names:
                print(f"  {st:18s} ({len(names)}): {names}")
        summary["generations"][gen] = per_state

    # Bonus techniques (downloaded but not in needed list — usually harmless)
    on_disk = {p.name.lower() for p in base.iterdir() if p.is_dir()}
    needed_lower = {t.lower() for t in ALL_NEEDED}
    bonus = sorted(on_disk - needed_lower)
    if bonus:
        print(f"\n[bonus] not in NEEDED but present: {bonus}")
        summary["bonus"] = bonus

    # Real
    print("\n[real]")
    for label, target in [("ff_real", df40_real_root() / "ff_real"),
                           ("cdf_real", df40_real_root() / "cdf_real")]:
        marker_ok = target.is_dir() and (target / EXTRACTED_MARKER).is_file()
        non_empty = target.is_dir() and any(target.iterdir())
        zp_path = Path(f"/content/{label}.zip")
        if marker_ok:
            state = "EXTRACTED"
        elif non_empty:
            # Legacy: backfill marker
            write_marker(target, source_zip="<legacy>", source_size_bytes=None, legacy=True)
            state = "EXTRACTED_LEGACY"
        elif zp_path.is_file():
            state = "NEEDS_EXTRACT"
        else:
            state = "NEEDS_DOWNLOAD"
        print(f"  {label:10s}  {state}")
        summary["real"][label] = state

    # Disk usage
    try:
        out = subprocess.run(
            ["du", "-sh", str(base)], capture_output=True, text=True, timeout=120,
        )
        print(f"\nDrive usage in {base}: {out.stdout.strip()}")
    except Exception:  # noqa: BLE001
        pass

    print("=" * 64)
    return summary


# --------------------------------------------------------------------------- #
#  Download
# --------------------------------------------------------------------------- #
def run_download() -> None:
    """Resume download of DF40_train folder + real ZIPs.

    Uses ``gdown --folder ... --remaining-ok`` so that quota errors on
    individual files do not abort the whole batch. Real ZIPs are tried
    individually with try/except.
    """
    base = df40_root()
    base.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("Phase 1/2: Downloading DF40_train folder (resume mode)")
    print("Quota errors per-file will be logged and skipped, not fatal.")
    print("=" * 64)

    # gdown --folder skips files that already exist at destination by name.
    subprocess.run(
        ["gdown", "--folder",
         f"https://drive.google.com/drive/folders/{DF40_TRAIN_FOLDER_ID}",
         "-O", str(base), "--remaining-ok"],
        check=False,
    )

    print("\n" + "=" * 64)
    print("Phase 2/2: Real ZIPs (FF++ + Celeb-DF)")
    print("=" * 64)

    real_root = df40_real_root()
    real_root.mkdir(parents=True, exist_ok=True)

    for fid, label in [(FF_REAL_FILE_ID, "ff_real"),
                        (CDF_REAL_FILE_ID, "cdf_real")]:
        target = real_root / label
        target.mkdir(parents=True, exist_ok=True)
        marker_ok = (target / EXTRACTED_MARKER).is_file()
        if marker_ok:
            print(f"[skip ] {label} already extracted")
            continue
        zp = Path(f"/content/{label}.zip")
        if zp.is_file() and zp.stat().st_size > 1_000_000:
            print(f"[skip ] {label}.zip already downloaded ({zp.stat().st_size/1e9:.1f}GB)")
            continue
        print(f"[try  ] {label}.zip ...")
        try:
            subprocess.run(
                ["gdown", "--id", fid, "-O", str(zp)], check=True,
            )
            print(f"[done ] {label}.zip downloaded")
        except subprocess.CalledProcessError as exc:
            print(f"[FAIL ] {label}: {exc}")
            print(f"        -> retry later, or use 'Add Shortcut to My Drive' workaround")


# --------------------------------------------------------------------------- #
#  Extract
# --------------------------------------------------------------------------- #
def extract_zip_to_folder(zp: Path, target: Path) -> tuple[bool, str]:
    """Atomically extract a ZIP, with redundant-root unwrapping + marker."""
    if target.is_dir():
        # Wipe partial state
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=False)

    try:
        result = subprocess.run(
            ["unzip", "-q", str(zp), "-d", str(target)],
            capture_output=True, text=True, timeout=UNZIP_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return False, f"timeout after {UNZIP_TIMEOUT_SECONDS}s"

    if result.returncode != 0:
        return False, f"unzip rc={result.returncode}: {result.stderr[:200]}"

    # Handle redundant root folder (some ZIPs nest contents under <name>/)
    entries = list(target.iterdir())
    tech_name = target.name
    if (
        len(entries) == 1
        and entries[0].is_dir()
        and entries[0].name.lower() == tech_name.lower()
    ):
        nested = entries[0]
        for item in nested.iterdir():
            shutil.move(str(item), str(target))
        nested.rmdir()

    # Write marker BEFORE deleting source ZIP, so a crash mid-cleanup
    # leaves us in EXTRACTED state, not PARTIAL.
    write_marker(
        target,
        source_zip=zp.name,
        source_size_bytes=zp.stat().st_size,
        legacy=False,
    )
    zp.unlink()
    return True, "ok"


def run_extract() -> None:
    """Extract every ZIP under df40/ + the two real ZIPs at /content/."""
    base = df40_root()
    base.mkdir(parents=True, exist_ok=True)

    zips = sorted(base.glob("*.zip"))
    print("=" * 64)
    print(f"Extracting {len(zips)} ZIPs in {base}")
    print("=" * 64)

    for zp in zips:
        tech_name = zp.stem
        target = base / tech_name
        size_gb = zp.stat().st_size / 1e9

        # State check: skip if already extracted with marker
        marker_existing = find_existing_dir(base, tech_name)
        if marker_existing is not None and (marker_existing / EXTRACTED_MARKER).is_file():
            print(f"[skip   ] {tech_name} (marker present, ZIP shouldn't be here — deleting)")
            zp.unlink()
            continue

        print(f"[extract] {zp.name} ({size_gb:.1f}GB) ... ", end="", flush=True)
        ok, msg = extract_zip_to_folder(zp, target)
        print("done" if ok else f"FAIL: {msg}")

    # Real ZIPs at /content/
    print("\n" + "=" * 64)
    print("Extracting real ZIPs (FF++, Celeb-DF)")
    print("=" * 64)
    for label in ["ff_real", "cdf_real"]:
        zp = Path(f"/content/{label}.zip")
        target = df40_real_root() / label
        if not zp.is_file():
            if not target.is_dir() or not (target / EXTRACTED_MARKER).is_file():
                print(f"[skip   ] {label}: no ZIP and not extracted (need to download first)")
            else:
                print(f"[skip   ] {label}: already extracted")
            continue
        if (target / EXTRACTED_MARKER).is_file():
            print(f"[skip   ] {label}: already extracted, deleting stray ZIP")
            zp.unlink()
            continue
        print(f"[extract] {label}.zip ({zp.stat().st_size/1e9:.1f}GB) ... ", end="", flush=True)
        ok, msg = extract_zip_to_folder(zp, target)
        print("done" if ok else f"FAIL: {msg}")


# --------------------------------------------------------------------------- #
#  Entrypoint
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Robust DF40 download + extract pipeline.",
    )
    parser.add_argument(
        "--action",
        choices=["status", "download", "extract", "all"],
        default="all",
        help="What to do. 'all' = status -> download -> extract -> status.",
    )
    args = parser.parse_args()

    if args.action == "status":
        status_report()
        return 0

    if args.action in ("download", "all"):
        run_download()

    if args.action in ("extract", "all"):
        run_extract()

    if args.action == "all":
        print("\n\n")
        print("#" * 64)
        print("# FINAL STATUS")
        print("#" * 64)
        status_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
