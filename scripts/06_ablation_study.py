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
    """A3 — Replay buffer size (fraction of previous train set)."""
    return [
        Variant(
            name=f"buf_{int(pct * 100):02d}pct",
            description=f"replay buffer = {pct * 100:.0f}% of previous train set",
            script="continual",
            config_overrides={
                "training": {
                    "anti_forgetting": {"replay": {"buffer_percentage": pct}}
                }
            },
            script_args={"--generation": "gen2", "--method": "replay"},
        )
        for pct in (0.01, 0.05, 0.10, 0.20)
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


_ABLATION_REGISTRY: dict[str, tuple[str, list[Variant]]] = {
    "A1": ("KD vs retention weight", _a1_alpha_beta()),
    "A2": ("KD temperature", _a2_temperature()),
    "A3": ("Replay buffer size", _a3_buffer_size()),
    "A4": ("Student head capacity", _a4_head_capacity()),
    "A5": ("Number of teachers", _a5_teacher_count()),
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
    logger.info("Wrote ablation summary: %s", summary_path)
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
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(_run(_parse_args()))
