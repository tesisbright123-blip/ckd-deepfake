"""Step 8: Generate every figure the thesis needs.

Entrypoint script — invoked via CLI, not imported by other code.

Usage:
    python scripts/08_generate_figures.py                   # generate everything
    python scripts/08_generate_figures.py --figures cgrs    # only CGRS + method comparison
    python scripts/08_generate_figures.py --figures ablation,edge
    python scripts/08_generate_figures.py --results-dir /other/path/to/results/raw

Figures produced (skipped when their inputs are missing):

* ``method_comparison``   — per-gen AUC × anti-forgetting method (bar chart).
* ``cgrs_trajectory``     — CGRS over training rounds (one line per method).
* ``ablation_{A1..A5}``   — metric vs swept parameter for each ablation summary.
* ``edge_pareto``         — latency vs AUC per TFLite quantization mode.
* ``training_{tag}``      — train loss + val AUC over epochs, one per
                            checkpoint whose ``metrics.history`` was saved.

Calls:
    src/utils/visualization.py (plot_* helpers + save_figure)
Reads:
    {results_dir}/{gen}_initial_metrics.json
    {results_dir}/{gen}_{method}_continual_metrics.json
    {results_dir}/ablation/{A}_summary.json
    {results_dir}/{run_tag}_edge_metrics.json
Writes:
    {figures_dir}/{name}.png + .pdf
    Log file at runs/generate_figures.log
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.utils.config import load_config  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.utils.visualization import (  # noqa: E402
    method_run_from_continual_json,
    method_run_from_initial_json,
    plot_ablation_sweep,
    plot_cgrs_trajectory,
    plot_edge_pareto,
    plot_method_comparison,
    plot_training_history,
    save_figure,
    setup_thesis_style,
)

_FIGURE_KINDS = ("method", "cgrs", "ablation", "edge", "training")
_ABLATION_X_LABELS = {
    "A1": "Variant (alpha-beta-gamma triple)",
    "A2": "KD temperature T",
    "A3": "Replay buffer size (%)",
    "A4": "Head config",
    "A5": "Number of teachers",
}


# --------------------------------------------------------------------------- #
#  Discovery
# --------------------------------------------------------------------------- #
def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return None


def _discover_continual_runs(results_dir: Path) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*_continual_metrics.json")):
        payload = _load_json(path)
        if payload is not None:
            payload["__source"] = str(path)
            payloads.append(payload)
    return payloads


def _discover_initial_runs(results_dir: Path) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*_initial_metrics.json")):
        payload = _load_json(path)
        if payload is not None:
            payload["__source"] = str(path)
            payloads.append(payload)
    return payloads


def _discover_ablation_summaries(results_dir: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    ablation_dir = results_dir / "ablation"
    if not ablation_dir.is_dir():
        return out
    for path in sorted(ablation_dir.glob("*_summary.json")):
        payload = _load_json(path)
        if payload is None:
            continue
        key = payload.get("ablation") or path.stem.split("_")[0]
        out[key] = payload
    return out


def _discover_edge_summaries(results_dir: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(results_dir.glob("*_edge_metrics.json")):
        payload = _load_json(path)
        if payload is None:
            continue
        tag = path.stem.replace("_edge_metrics", "")
        out[tag] = payload
    return out


# --------------------------------------------------------------------------- #
#  Figure builders
# --------------------------------------------------------------------------- #
def _figure_method_comparison(
    continual_payloads: list[dict[str, Any]],
    initial_payloads: list[dict[str, Any]],
    figures_dir: Path,
    logger,
) -> None:
    runs = [
        r
        for r in (method_run_from_continual_json(p) for p in continual_payloads)
        if r is not None
    ]
    if not runs:
        logger.info("method_comparison: no continual runs found — skipping")
        return
    baseline_auc: dict[str, float] = {}
    for payload in initial_payloads:
        baseline = method_run_from_initial_json(payload)
        if baseline is not None:
            baseline_auc.update(baseline.per_gen_auc)
    fig = plot_method_comparison(runs, baseline_auc=baseline_auc or None)
    save_figure(fig, figures_dir / "method_comparison")


def _figure_cgrs_trajectory(
    continual_payloads: list[dict[str, Any]], figures_dir: Path, logger
) -> None:
    grouped: dict[str, list] = {}
    for payload in continual_payloads:
        run = method_run_from_continual_json(payload)
        if run is None or run.cgrs is None:
            continue
        grouped.setdefault(run.method, []).append(run)
    if not grouped:
        logger.info("cgrs_trajectory: no runs with CGRS — skipping")
        return
    # Sort each method's runs by generation order so the line is monotonic.
    gen_order = ("gen1", "gen2", "gen3")
    for method in grouped:
        grouped[method].sort(key=lambda r: gen_order.index(r.generation))
    fig = plot_cgrs_trajectory(grouped)
    save_figure(fig, figures_dir / "cgrs_trajectory")


def _figure_ablations(
    summaries: dict[str, dict[str, Any]], figures_dir: Path, logger
) -> None:
    if not summaries:
        logger.info("ablation: no summaries found — skipping")
        return
    for ablation, summary in summaries.items():
        # Pick the dominant split for this ablation.
        split = _dominant_split(summary)
        x_label = _ABLATION_X_LABELS.get(ablation, ablation)
        fig = plot_ablation_sweep(
            summary,
            x_key="name",
            x_label=x_label,
            metric="auc",
            split_name=split,
            title=f"{ablation}: {summary.get('label', '')}",
        )
        save_figure(fig, figures_dir / f"ablation_{ablation}")


def _dominant_split(summary: dict[str, Any]) -> str | None:
    """Pick the split (e.g. gen1_test) that appears most often across runs."""
    counts: dict[str, int] = {}
    for row in summary.get("runs", []):
        for split in (row.get("splits") or {}):
            counts[split] = counts.get(split, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.__getitem__)


def _figure_edge(
    edge_summaries: dict[str, dict[str, Any]], figures_dir: Path, logger
) -> None:
    if not edge_summaries:
        logger.info("edge_pareto: no edge summaries found — skipping")
        return
    fig = plot_edge_pareto(edge_summaries)
    save_figure(fig, figures_dir / "edge_pareto")


def _figure_training(
    continual_payloads: list[dict[str, Any]],
    initial_payloads: list[dict[str, Any]],
    figures_dir: Path,
    logger,
) -> None:
    payloads = list(continual_payloads) + list(initial_payloads)
    emitted = 0
    for payload in payloads:
        history = payload.get("history")
        if not history:
            continue
        tag = _training_tag(payload)
        fig = plot_training_history(history, title=f"Training history — {tag}")
        save_figure(fig, figures_dir / f"training_{tag}")
        emitted += 1
    if emitted == 0:
        logger.info(
            "training: no checkpoints with a ``history`` block — skipping"
        )


def _training_tag(payload: dict[str, Any]) -> str:
    gen = payload.get("generation", "gen?")
    method = payload.get("method")
    return f"{gen}_{method}" if method else str(gen)


# --------------------------------------------------------------------------- #
#  Run loop
# --------------------------------------------------------------------------- #
def _parse_kinds(text: str | None) -> set[str]:
    if not text or text.lower() == "all":
        return set(_FIGURE_KINDS)
    wanted = {item.strip().lower() for item in text.split(",") if item.strip()}
    unknown = wanted - set(_FIGURE_KINDS)
    if unknown:
        raise SystemExit(
            f"Unknown figure kinds: {sorted(unknown)}. "
            f"Expected subset of {_FIGURE_KINDS}."
        )
    return wanted


def _run(args: argparse.Namespace) -> int:
    logger = get_logger("generate_figures", log_file="runs/generate_figures.log")
    setup_thesis_style()

    cfg = load_config(args.config)
    drive_root = Path(cfg["paths"]["drive_root"])

    results_dir = (
        Path(args.results_dir) if args.results_dir else drive_root / "results" / "raw"
    )
    figures_dir = (
        Path(args.figures_dir) if args.figures_dir else drive_root / "results" / "figures"
    )
    figures_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Results dir: %s", results_dir)
    logger.info("Figures dir: %s", figures_dir)

    kinds = _parse_kinds(args.figures)
    logger.info("Generating figure kinds: %s", sorted(kinds))

    continual_payloads = _discover_continual_runs(results_dir) if (
        {"method", "cgrs", "training"} & kinds
    ) else []
    initial_payloads = _discover_initial_runs(results_dir) if (
        {"method", "training"} & kinds
    ) else []
    ablation_summaries = _discover_ablation_summaries(results_dir) if (
        "ablation" in kinds
    ) else {}
    edge_summaries = _discover_edge_summaries(results_dir) if (
        "edge" in kinds
    ) else {}

    logger.info(
        "Inputs: continual=%d initial=%d ablations=%d edge=%d",
        len(continual_payloads),
        len(initial_payloads),
        len(ablation_summaries),
        len(edge_summaries),
    )

    if "method" in kinds:
        _figure_method_comparison(
            continual_payloads, initial_payloads, figures_dir, logger
        )
    if "cgrs" in kinds:
        _figure_cgrs_trajectory(continual_payloads, figures_dir, logger)
    if "ablation" in kinds:
        _figure_ablations(ablation_summaries, figures_dir, logger)
    if "edge" in kinds:
        _figure_edge(edge_summaries, figures_dir, logger)
    if "training" in kinds:
        _figure_training(
            continual_payloads, initial_payloads, figures_dir, logger
        )
    logger.info("Done. Figures written to %s", figures_dir)
    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate all thesis figures from existing result JSONs"
    )
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument(
        "--figures",
        default="all",
        help=(
            "Comma-separated subset of "
            f"{_FIGURE_KINDS}, or 'all' (default)."
        ),
    )
    p.add_argument(
        "--results-dir",
        default=None,
        help="Override {drive}/results/raw lookup path",
    )
    p.add_argument(
        "--figures-dir",
        default=None,
        help="Override {drive}/results/figures output path",
    )
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(_run(_parse_args()))
