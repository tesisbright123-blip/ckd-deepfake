"""Step 6: Hyperparameter / architectural sensitivity analysis.

Entrypoint script - invoked via CLI, not imported by other code.

Usage:
    python scripts/06_ablation_study.py --ablation A1 --seed 0
    python scripts/06_ablation_study.py --ablation A2 --variants t_1,t_4
    python scripts/06_ablation_study.py --ablation A3 --seeds 0,1,2
    python scripts/06_ablation_study.py --list

Ablations (see ``_ABLATION_REGISTRY`` below for the concrete grids):
    A1: KD vs Retention weight (alpha / beta / gamma) — continual, gen2 on replay
    A2: KD temperature — initial distillation on gen1
    A3: Replay buffer size — continual, gen2 on replay
    A4: Student head capacity — initial distillation on gen1
    A5: Number of teachers (ensemble subsets) — requires pre-generated soft labels
        per subset under {drive}/soft_labels/{gen1}_{subset}/... (documented below)
    A6: Anti-forgetting method comparison — continual gen2 with method in
        {ewc, lwf, replay, replay+ewc, der++}. Buffer 10% baseline; same
        learning rate / num_epochs / EWC lambda / LwF temperature across
        all five variants. Pre-requisite: gen1 best.pth must exist.

Each variant is orchestrated by writing a *temporary* YAML config with the
override applied and invoking ``scripts/04_initial_distillation.py`` or
``scripts/05_continual_distillation.py`` as a subprocess. The child scripts
write their usual metrics JSON; this orchestrator reads those back and writes
an aggregate summary.

Calls: scripts/04_initial_distillation.py, scripts/05_continual_distillation.py
Reads:
    Base YAML config (--config)
    Existing splits, soft labels, and prerequisite checkpoints
    (e.g. A1/A3 require a gen1 checkpoint — run scripts/04 first)
Writes:
    Per-variant run artefacts under
        {drive}/checkpoints/students/ablation/{ablation}/{variant}/seed{N}/
        {drive}/results/raw/ablation/{ablation}/{variant}_seed{N}.json
    Aggregate summary at
        {drive}/results/raw/ablation/{ablation}_summary.json
        (ISO 8601 ``generated_at``)
    Log file at runs/ablation_{ablation}.log
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.utils.config import load_config  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

_INITIAL_SCRIPT = _REPO_ROOT / "scripts" / "04_initial_distillation.py"
_CONTINUAL_SCRIPT = _REPO_ROOT / "scripts" / "05_continual_distillation.py"
_TIMESTAMP_FMT = "%Y-%m-%dT%H:%M:%S"


# --------------------------------------------------------------------------- #
#  Variant definition
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Variant:
    """A single ablation cell.

    Attributes:
        name: Short id used in filenames, e.g. ``"alpha07_beta01"``.
        description: Human-readable summary for logs + summary JSON.
        script: Which child script to run (``"initial"`` → scripts/04,
            ``"continual"`` → scripts/05).
        config_overrides: Deep-merged into the loaded YAML before it is
            re-dumped into the temp config consumed by the child script.
        script_args: Extra CLI args appended to the child invocation
            (e.g. ``{"--generation": "gen2", "--method": "replay"}``).
    """

    name: str
    description: str
    script: str  # "initial" | "continual"
    config_overrides: dict[str, Any] = field(default_factory=dict)
    script_args: dict[str, str] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
#  Ablation registry
# --------------------------------------------------------------------------- #
def _a1_alpha_beta() -> list[Variant]:
    """A1 — KD vs Retention loss weight.

    Runs continual distillation on gen2 with replay, sweeping the
    (alpha, beta, gamma) triple in ``training.continual_distillation``.
    """
    base = {"generation": "gen2", "method": "replay"}
    grid = [
        ("kd_heavy", 0.7, 0.1, 0.2),
        ("default", 0.5, 0.3, 0.2),
        ("retention_heavy", 0.3, 0.5, 0.2),
        ("aggressive_retention", 0.2, 0.6, 0.2),
    ]
    out: list[Variant] = []
    for label, alpha, beta, gamma in grid:
        out.append(
            Variant(
                name=f"a{alpha:.1f}_b{beta:.1f}_g{gamma:.1f}".replace(".", ""),
                description=f"{label} (alpha={alpha}, beta={beta}, gamma={gamma})",
                script="continual",
                config_overrides={
                    "training": {
                        "continual_distillation": {
                            "alpha": alpha,
                            "beta": beta,
                            "gamma": gamma,
                        }
                    }
                },
                script_args={"--generation": base["generation"], "--method": base["method"]},
            )
        )
    return out


def _a2_temperature() -> list[Variant]:
    """A2 — KD softmax temperature.

    Runs initial distillation on gen1 with T ∈ {1, 2, 4, 8}.
    """
    return [
        Variant(
            name=f"t_{int(t)}",
            description=f"initial KD temperature T={t}",
            script="initial",
            config_overrides={
                "training": {"initial_distillation": {"temperature": t}}
            },
            script_args={"--generation": "gen1"},
        )
        for t in (1.0, 2.0, 4.0, 8.0)
    ]


def _a3_buffer_size() -> list[Variant]:
    """A3 — Replay buffer size sensitivity (percentage of previous train set).

    Grid {5%, 10%, 15%, 20%} is centred so the main-pipeline value (10%) is
    one of the points — letting the ablation be read as a sensitivity curve
    around the chosen config rather than an unrelated sweep. Uses the SAME
    method as the main pipeline (replay+ewc) so the buffer effect is isolated
    within the deployed configuration, not a different (replay-only) one.
    """
    return [
        Variant(
            name=f"buf_{int(pct * 100):02d}pct",
            description=f"replay buffer = {pct * 100:.0f}% of previous train set (replay+ewc)",
            script="continual",
            config_overrides={
                "training": {
                    "anti_forgetting": {"replay": {"buffer_percentage": pct}}
                }
            },
            script_args={"--generation": "gen2", "--method": "replay+ewc"},
        )
        for pct in (0.05, 0.10, 0.15, 0.20)
    ]


def _a4_head_capacity() -> list[Variant]:
    """A4 — Student head capacity.

    Sweeps hidden_dim × dropout for the MobileNetV3 student head. Adding a
    whole new backbone requires extending ``src/models/students/`` and
    adjusting ``build_student`` — out of scope for this grid.
    """
    grid = [(128, 0.1), (256, 0.2), (256, 0.4), (512, 0.2)]
    out: list[Variant] = []
    for hidden, dropout in grid:
        name = f"h{hidden}_d{int(dropout * 10):02d}"
        out.append(
            Variant(
                name=name,
                description=f"student head hidden_dim={hidden}, dropout={dropout}",
                script="initial",
                config_overrides={
                    "student": {
                        "head": {"hidden_dim": hidden, "dropout": dropout}
                    }
                },
                script_args={"--generation": "gen1"},
            )
        )
    return out


def _a5_teacher_count() -> list[Variant]:
    """A5 — Ensemble subset size.

    Pre-requisite: soft labels must already exist for each subset under
    ``{drive}/soft_labels/gen1_{subset}/{split}/ensemble.npy``. Generate them
    via ``python scripts/03_generate_soft_labels.py --teachers <subset>``
    and move/symlink the output dir to the matching name. The child script
    is then pointed at the alternate path via --results/checkpoint dirs, but
    soft-label path itself is hard-coded to ``{drive}/soft_labels/{gen}/...``
    in scripts/04; so for A5 we instead override ``paths.drive_root`` to a
    shadow tree that contains the subset soft labels under the canonical
    gen1 directory. See ``_prepare_a5_drive`` (TODO, out of scope here) for
    an automation sketch.
    """
    return [
        Variant(
            name="t1_effb4",
            description="ensemble of 1: efficientnet_b4",
            script="initial",
            config_overrides={
                "teacher": {
                    "models": [
                        {
                            "name": "efficientnet_b4",
                            "source": "deepfakebench",
                            "weight_path": "{drive}/checkpoints/teachers/efficientnet_b4.pth",
                            "input_size": 224,
                            "ensemble_weight": "auto",
                        }
                    ]
                }
            },
            script_args={"--generation": "gen1"},
        ),
        Variant(
            name="t2_effb4_recce",
            description="ensemble of 2: efficientnet_b4 + recce",
            script="initial",
            config_overrides={
                "teacher": {
                    "models": [
                        {
                            "name": "efficientnet_b4",
                            "source": "deepfakebench",
                            "weight_path": "{drive}/checkpoints/teachers/efficientnet_b4.pth",
                            "input_size": 224,
                            "ensemble_weight": "auto",
                        },
                        {
                            "name": "recce",
                            "source": "deepfakebench",
                            "weight_path": "{drive}/checkpoints/teachers/recce.pth",
                            "input_size": 224,
                            "ensemble_weight": "auto",
                        },
                    ]
                }
            },
            script_args={"--generation": "gen1"},
        ),
        Variant(
            name="t3_full",
            description="ensemble of 3 (default)",
            script="initial",
            config_overrides={},
            script_args={"--generation": "gen1"},
        ),
    ]


def _a6_anti_forgetting_methods() -> list[Variant]:
    """A6 — Anti-forgetting method comparison (gen2 continual distillation).

    Compares the five anti-forgetting strategies side-by-side, all with
    the same buffer percentage (10%) and same gen2 task. Held constant:
    learning rate, batch size, num epochs, replay selection (herding),
    EWC lambda, LwF temperature.

    Variants:
        ewc           — Fisher penalty only
        lwf           — Learning-without-Forgetting (KD on frozen prev student)
        replay        — herding-selected exemplar buffer only
        replay+ewc    — combine data + weight protection
        der++         — Dark Experience Replay (logits MSE on stored logits)

    Pre-requisite: gen1 initial-distillation checkpoint must exist at
    ``{drive}/checkpoints/students/gen1/best.pth``. Each variant runs
    scripts/05 with the matching --method flag and writes its own metrics
    JSON under ``{drive}/results/raw/ablation/A6/<variant>_seed{N}.json``.
    """
    methods = ["ewc", "lwf", "replay", "replay+ewc", "der++"]
    out: list[Variant] = []
    for m in methods:
        # File-system-safe variant name: replace + and ++ with English words.
        safe_name = m.replace("++", "_pp").replace("+", "_plus")
        out.append(
            Variant(
                name=safe_name,
                description=f"Anti-forgetting method = {m} (gen2 continual)",
                script="continual",
                config_overrides={},
                script_args={"--generation": "gen2", "--method": m},
            )
        )
    return out


_ABLATION_REGISTRY: dict[str, tuple[str, list[Variant]]] = {
    "A1": ("KD vs retention weight", _a1_alpha_beta()),
    "A2": ("KD temperature", _a2_temperature()),
    "A3": ("Replay buffer size", _a3_buffer_size()),
    "A4": ("Student head capacity", _a4_head_capacity()),
    "A5": ("Number of teachers", _a5_teacher_count()),
    "A6": (
        "Anti-forgetting method (ewc/lwf/replay/replay+ewc/der++)",
        _a6_anti_forgetting_methods(),
    ),
}


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _write_patched_config(
    base_config_path: Path,
    overrides: dict[str, Any],
    tmp_dir: Path,
    variant_name: str,
) -> Path:
    """Write ``base_config_path`` with ``overrides`` deep-merged to a temp file.

    We load the **raw** YAML (no path resolution) so the placeholders survive
    into the child process, which will resolve them itself via load_config.
    """
    with open(base_config_path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}
    patched = _deep_merge(raw, overrides)

    target = tmp_dir / f"config_{variant_name}.yaml"
    with open(target, "w", encoding="utf-8") as f:
        yaml.safe_dump(patched, f, sort_keys=False)
    return target


def _variant_output_dirs(
    drive_root: Path, ablation: str, variant: Variant, seed: int
) -> tuple[Path, Path]:
    """Checkpoint + results dirs for a specific (variant, seed) run."""
    checkpoint_dir = (
        drive_root
        / "checkpoints"
        / "students"
        / "ablation"
        / ablation
        / variant.name
        / f"seed{seed}"
    )
    results_dir = drive_root / "results" / "raw" / "ablation" / ablation / variant.name
    return checkpoint_dir, results_dir


def _expected_metrics_path(
    results_dir: Path, variant: Variant, generation: str
) -> Path:
    """The JSON path the child script is guaranteed to produce."""
    if variant.script == "initial":
        return results_dir / f"{generation}_initial_metrics.json"
    method = variant.script_args.get("--method", "")
    return results_dir / f"{generation}_{method}_continual_metrics.json"


def _invoke_child(
    script_path: Path,
    config_path: Path,
    checkpoint_dir: Path,
    results_dir: Path,
    seed: int,
    extra_args: dict[str, str],
) -> int:
    """Run a child script synchronously. Returns its exit code."""
    cmd: list[str] = [
        sys.executable,
        str(script_path),
        "--config",
        str(config_path),
        "--seed",
        str(seed),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--results-dir",
        str(results_dir),
    ]
    for key, value in extra_args.items():
        cmd.extend([key, str(value)])
    return subprocess.run(cmd, check=False).returncode


def _load_metrics(metrics_path: Path) -> dict[str, Any] | None:
    if not metrics_path.is_file():
        return None
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _summarize_variant_run(
    variant: Variant, seed: int, metrics: dict[str, Any] | None
) -> dict[str, Any]:
    """Trim the per-run JSON to the fields we aggregate in the summary."""
    if metrics is None:
        return {
            "variant": variant.name,
            "description": variant.description,
            "seed": seed,
            "status": "failed",
        }
    splits = metrics.get("splits") or {}
    per_split = {
        split: {
            "auc": float(m.get("auc", 0.0)),
            "log_loss": float(m.get("log_loss", 0.0)),
            "accuracy": float(m.get("accuracy", 0.0)),
            "num_samples": int(m.get("num_samples", 0)),
        }
        for split, m in splits.items()
    }
    return {
        "variant": variant.name,
        "description": variant.description,
        "seed": seed,
        "status": "ok",
        "splits": per_split,
        "cgrs": metrics.get("cgrs"),
        "auc_peak": metrics.get("auc_peak"),
        "auc_after": metrics.get("auc_after"),
        "best_val_auc": metrics.get("best_val_auc"),
    }


def _write_summary(
    summary_path: Path,
    *,
    ablation: str,
    label: str,
    variants: list[Variant],
    runs: list[dict[str, Any]],
    base_config_path: Path,
) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "ablation": ablation,
        "label": label,
        "base_config": str(base_config_path),
        "generated_at": datetime.now(timezone.utc).strftime(_TIMESTAMP_FMT),
        "variants": [
            {"name": v.name, "description": v.description, "script": v.script}
            for v in variants
        ],
        "runs": runs,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# --------------------------------------------------------------------------- #
#  Drive sync + resume helpers (defensive against Colab session resets)
# --------------------------------------------------------------------------- #
def _drive_sync_paths(
    drive_sync_root: Path, ablation: str, variant: Variant, seed: int
) -> tuple[Path, Path, Path]:
    """Return (drive_results_dir, drive_checkpoint_dir, completion_marker).

    The completion marker lives next to the metrics JSON so a simple
    ``Path.is_file()`` check tells us whether the variant has finished
    syncing to Drive previously.
    """
    drive_results_dir = (
        drive_sync_root / "results" / "raw" / "ablation" / ablation / variant.name
    )
    drive_checkpoint_dir = (
        drive_sync_root
        / "checkpoints"
        / "students"
        / "ablation"
        / ablation
        / variant.name
        / f"seed{seed}"
    )
    marker = drive_results_dir / f".completed_seed{seed}"
    return drive_results_dir, drive_checkpoint_dir, marker


def _is_variant_completed_on_drive(
    drive_sync_root: Path | None,
    ablation: str,
    variant: Variant,
    seed: int,
) -> bool:
    """Resume support: was this (variant, seed) successfully synced before?"""
    if drive_sync_root is None:
        return False
    _, _, marker = _drive_sync_paths(drive_sync_root, ablation, variant, seed)
    return marker.is_file()


def _load_completed_metrics(
    drive_sync_root: Path,
    ablation: str,
    variant: Variant,
    seed: int,
    generation: str,
) -> dict[str, Any] | None:
    """Read previously-synced metrics JSON from Drive for resume."""
    drive_results_dir, _, _ = _drive_sync_paths(
        drive_sync_root, ablation, variant, seed
    )
    # Look for both single-seed-named and multi-seed-named files.
    candidates: list[Path] = []
    if variant.script == "initial":
        candidates.append(drive_results_dir / f"{generation}_initial_metrics.json")
        candidates.append(drive_results_dir / f"{generation}_initial_metrics_seed{seed}.json")
    else:
        method = variant.script_args.get("--method", "")
        candidates.append(
            drive_results_dir / f"{generation}_{method}_continual_metrics.json"
        )
        candidates.append(
            drive_results_dir
            / f"{generation}_{method}_continual_metrics_seed{seed}.json"
        )
    for c in candidates:
        if c.is_file():
            return _load_metrics(c)
    return None


def _copy_file_safe(src: Path, dst: Path, *, logger) -> bool:
    """Copy with mkdir parent, robust to FUSE quirks. Returns success."""
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        return True
    except OSError as exc:
        logger.error("Copy failed %s -> %s: %s", src, dst, exc)
        return False


def _sync_variant_to_drive(
    *,
    drive_sync_root: Path,
    local_results_dir: Path,
    local_checkpoint_dir: Path,
    ablation: str,
    variant: Variant,
    seed: int,
    metrics_filename: str,
    logger,
) -> bool:
    """Copy variant outputs from local NVMe to Drive immediately after run.

    This is the defense against Colab session resets — even if the runtime
    dies before the orchestrator can finish, every previously-completed
    variant is already on Drive.

    Writes the completion marker LAST (atomic-ish — if the marker exists,
    callers know the metrics+ckpt are both there).

    Returns True on full success, False if any required artifact was missing.
    """
    drive_results_dir, drive_checkpoint_dir, marker = _drive_sync_paths(
        drive_sync_root, ablation, variant, seed
    )

    success = True

    # 1. Sync metrics JSON (required)
    local_metrics = local_results_dir / metrics_filename
    if local_metrics.is_file():
        drive_metrics = drive_results_dir / metrics_filename
        if not _copy_file_safe(local_metrics, drive_metrics, logger=logger):
            success = False
        else:
            logger.info("  [sync ] metrics -> %s", drive_metrics)
    else:
        logger.warning("  [sync ] no metrics JSON at %s -> skipping marker", local_metrics)
        return False

    # 2. Sync best.pth (optional but desirable for reproducibility)
    local_best = local_checkpoint_dir / "best.pth"
    if local_best.is_file():
        drive_best = drive_checkpoint_dir / "best.pth"
        if _copy_file_safe(local_best, drive_best, logger=logger):
            size_mb = drive_best.stat().st_size / 1e6
            logger.info("  [sync ] checkpoint -> %s (%.1f MB)", drive_best, size_mb)

    # 3. Sync child script log if present (helpful for debugging variant-specific issues)
    method = variant.script_args.get("--method")
    gen = variant.script_args.get("--generation", "gen1")
    child_log_candidates = []
    if variant.script == "initial":
        child_log_candidates.append(Path("runs") / f"initial_distillation_{gen}.log")
    else:
        child_log_candidates.append(
            Path("runs") / f"continual_distillation_{gen}_{method}.log"
        )
    for log_src in child_log_candidates:
        if log_src.is_file():
            log_dst = drive_results_dir / log_src.name
            _copy_file_safe(log_src, log_dst, logger=logger)

    # 4. Write completion marker LAST so partial syncs are not misread as done.
    if success:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps(
                {
                    "completed_at": datetime.now(timezone.utc).strftime(_TIMESTAMP_FMT),
                    "variant": variant.name,
                    "seed": seed,
                    "metrics_filename": metrics_filename,
                },
                indent=2,
            )
        )
        logger.info("  [sync ] marker -> %s", marker)
    return success


def _update_progress_summary(
    *,
    drive_sync_root: Path,
    ablation: str,
    label: str,
    variants: list[Variant],
    runs: list[dict[str, Any]],
    base_config_path: Path,
    is_final: bool,
) -> None:
    """Write/update the cross-variant summary JSON on Drive.

    Called after each variant so that even a partial run leaves a
    structured record. The ``is_final`` flag flips a top-level boolean
    so consumers can detect whether the summary represents a finished
    grid or an in-progress one.
    """
    summary_dir = drive_sync_root / "results" / "raw" / "ablation"
    summary_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_summary.json" if is_final else "_progress.json"
    summary_path = summary_dir / f"{ablation}{suffix}"
    _write_summary(
        summary_path,
        ablation=ablation,
        label=label,
        variants=variants,
        runs=runs,
        base_config_path=base_config_path,
    )


# --------------------------------------------------------------------------- #
#  Run loop
# --------------------------------------------------------------------------- #
def _list_ablations() -> None:
    for ablation, (label, variants) in _ABLATION_REGISTRY.items():
        print(f"{ablation}: {label} ({len(variants)} variants)")
        for v in variants:
            print(f"  - {v.name}: {v.description}")


def _filter_variants(
    variants: list[Variant], wanted: list[str] | None
) -> list[Variant]:
    if not wanted:
        return variants
    want = set(wanted)
    filtered = [v for v in variants if v.name in want]
    if not filtered:
        known = ", ".join(v.name for v in variants)
        raise SystemExit(
            f"No variants matched {sorted(want)}. Known variants: {known}"
        )
    return filtered


def _parse_csv_list(text: str | None) -> list[str] | None:
    if not text:
        return None
    return [item.strip() for item in text.split(",") if item.strip()]


def _run(args: argparse.Namespace) -> int:
    if args.list:
        _list_ablations()
        return 0

    if args.ablation not in _ABLATION_REGISTRY:
        known = ", ".join(sorted(_ABLATION_REGISTRY))
        raise SystemExit(f"Unknown ablation {args.ablation!r}. Known: {known}")

    label, all_variants = _ABLATION_REGISTRY[args.ablation]
    variants = _filter_variants(all_variants, _parse_csv_list(args.variants))
    seeds = [int(s) for s in (_parse_csv_list(args.seeds) or [str(args.seed)])]

    logger = get_logger(
        f"ablation.{args.ablation}",
        log_file=f"runs/ablation_{args.ablation}.log",
    )
    logger.info(
        "Ablation %s (%s) - %d variants x %d seeds",
        args.ablation,
        label,
        len(variants),
        len(seeds),
    )

    # Resolve the drive root via the resolved config (matches child script view).
    resolved_cfg = load_config(args.config)
    drive_root = Path(resolved_cfg["paths"]["drive_root"])

    drive_sync_root = (
        Path(args.drive_sync_root).expanduser().resolve()
        if args.drive_sync_root
        else None
    )
    if drive_sync_root is not None:
        logger.info("Drive sync target: %s", drive_sync_root)
        if args.resume:
            logger.info("Resume mode: variants with completion markers on Drive will be skipped")
    elif args.resume:
        logger.warning(
            "--resume passed without --drive-sync-root; resume will rely on the "
            "local drive_root (%s) which is NVMe-ephemeral on Colab.",
            drive_root,
        )

    base_config_path = Path(args.config).resolve()
    runs: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix=f"ablation_{args.ablation}_") as tmp:
        tmp_dir = Path(tmp)
        for variant in variants:
            patched_cfg = _write_patched_config(
                base_config_path, variant.config_overrides, tmp_dir, variant.name
            )
            script_path = (
                _INITIAL_SCRIPT if variant.script == "initial" else _CONTINUAL_SCRIPT
            )
            generation = variant.script_args.get("--generation", "gen1")

            for seed in seeds:
                checkpoint_dir, results_dir = _variant_output_dirs(
                    drive_root, args.ablation, variant, seed
                )
                results_dir.mkdir(parents=True, exist_ok=True)

                # Resume: skip variants already synced to Drive.
                if args.resume and drive_sync_root is not None and _is_variant_completed_on_drive(
                    drive_sync_root, args.ablation, variant, seed
                ):
                    metrics = _load_completed_metrics(
                        drive_sync_root, args.ablation, variant, seed, generation
                    )
                    summary_entry = _summarize_variant_run(variant, seed, metrics)
                    summary_entry["status"] = "skipped (resume — already on Drive)"
                    runs.append(summary_entry)
                    logger.info(
                        "[skip ] variant=%s seed=%d (resume — completion marker on Drive)",
                        variant.name,
                        seed,
                    )
                    # Keep progress summary in sync as we skip pre-completed cells.
                    if drive_sync_root is not None:
                        _update_progress_summary(
                            drive_sync_root=drive_sync_root,
                            ablation=args.ablation,
                            label=label,
                            variants=variants,
                            runs=runs,
                            base_config_path=base_config_path,
                            is_final=False,
                        )
                    continue

                logger.info(
                    "-> variant=%s seed=%d (%s)",
                    variant.name,
                    seed,
                    variant.description,
                )
                if args.dry_run:
                    runs.append(
                        {
                            "variant": variant.name,
                            "description": variant.description,
                            "seed": seed,
                            "status": "skipped (dry-run)",
                            "config_path": str(patched_cfg),
                            "checkpoint_dir": str(checkpoint_dir),
                            "results_dir": str(results_dir),
                        }
                    )
                    continue

                rc = _invoke_child(
                    script_path=script_path,
                    config_path=patched_cfg,
                    checkpoint_dir=checkpoint_dir,
                    results_dir=results_dir,
                    seed=seed,
                    extra_args=variant.script_args,
                )
                if rc != 0:
                    logger.error(
                        "Child script exited with code %d (variant=%s seed=%d)",
                        rc,
                        variant.name,
                        seed,
                    )
                    runs.append(
                        {
                            "variant": variant.name,
                            "description": variant.description,
                            "seed": seed,
                            "status": f"failed (rc={rc})",
                        }
                    )
                    # Update progress summary even on failures, so we can resume past them.
                    if drive_sync_root is not None:
                        _update_progress_summary(
                            drive_sync_root=drive_sync_root,
                            ablation=args.ablation,
                            label=label,
                            variants=variants,
                            runs=runs,
                            base_config_path=base_config_path,
                            is_final=False,
                        )
                    if args.fail_fast:
                        return rc
                    continue

                metrics_path = _expected_metrics_path(
                    results_dir, variant, generation
                )
                metrics = _load_metrics(metrics_path)
                summary_entry = _summarize_variant_run(variant, seed, metrics)
                summary_entry["metrics_path"] = str(metrics_path)
                runs.append(summary_entry)
                logger.info(
                    "   ok: %s (metrics=%s)", variant.name, metrics_path
                )

                # Sync this variant to Drive immediately so a later session
                # reset cannot wipe its results.
                if drive_sync_root is not None and metrics_path.is_file():
                    sync_ok = _sync_variant_to_drive(
                        drive_sync_root=drive_sync_root,
                        local_results_dir=results_dir,
                        local_checkpoint_dir=checkpoint_dir,
                        ablation=args.ablation,
                        variant=variant,
                        seed=seed,
                        metrics_filename=metrics_path.name,
                        logger=logger,
                    )
                    if not sync_ok:
                        logger.warning(
                            "Drive sync incomplete for variant=%s seed=%d — "
                            "result file may not be on Drive yet.",
                            variant.name,
                            seed,
                        )
                    # Always refresh the progress summary so a partial run is recoverable.
                    _update_progress_summary(
                        drive_sync_root=drive_sync_root,
                        ablation=args.ablation,
                        label=label,
                        variants=variants,
                        runs=runs,
                        base_config_path=base_config_path,
                        is_final=False,
                    )

    summary_path = (
        drive_root / "results" / "raw" / "ablation" / f"{args.ablation}_summary.json"
    )
    _write_summary(
        summary_path,
        ablation=args.ablation,
        label=label,
        variants=variants,
        runs=runs,
        base_config_path=base_config_path,
    )
    logger.info("Wrote local ablation summary: %s", summary_path)

    if drive_sync_root is not None:
        # Final summary on Drive (replaces the progress file's role).
        drive_summary_path = (
            drive_sync_root
            / "results"
            / "raw"
            / "ablation"
            / f"{args.ablation}_summary.json"
        )
        _write_summary(
            drive_summary_path,
            ablation=args.ablation,
            label=label,
            variants=variants,
            runs=runs,
            base_config_path=base_config_path,
        )
        logger.info("Wrote final Drive ablation summary: %s", drive_summary_path)
    return 0


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Orchestrate hyperparameter / architectural ablations"
    )
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument(
        "--ablation",
        default=None,
        help=f"Ablation id, one of {sorted(_ABLATION_REGISTRY)}",
    )
    p.add_argument(
        "--variants",
        default=None,
        help="Comma-separated subset of variant names (default: all)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seeds, overrides --seed (e.g. '0,1,2')",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only resolve variants, don't invoke child scripts",
    )
    p.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort the whole grid on the first failing child",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="Print the ablation registry and exit",
    )
    p.add_argument(
        "--drive-sync-root",
        default=None,
        help=(
            "Persistent Drive path to copy per-variant artefacts into AFTER "
            "each variant completes (defensive against Colab session resets). "
            "Typical Colab value: "
            "'/content/drive/MyDrive/CKD_Thesis' (regardless of where the "
            "config's drive_root points). When set, also writes a "
            "<ablation>_progress.json next to each completion, plus a final "
            "<ablation>_summary.json."
        ),
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip variants that already have a completion marker on "
            "--drive-sync-root. Without --drive-sync-root, this flag has no "
            "effect on Colab because the NVMe state is ephemeral."
        ),
    )
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(_run(_parse_args()))
