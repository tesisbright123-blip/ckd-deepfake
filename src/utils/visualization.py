"""Publication-quality matplotlib helpers for CKD thesis figures.

All plotting functions take already-aggregated dicts / dataclasses and return
the :class:`matplotlib.figure.Figure` so callers can chain further tweaks
before saving. Use :func:`save_figure` to emit both PNG (for slide decks)
and PDF (for LaTeX) versions from a single figure.

Exposed figure builders:

* :func:`plot_method_comparison` — Bar chart: per-gen test AUC per method.
* :func:`plot_cgrs_trajectory`   — Line plot: CGRS across generations per method.
* :func:`plot_ablation_sweep`    — Metric vs swept parameter (A1/A2/A3/A4).
* :func:`plot_edge_pareto`       — Scatter: latency vs AUC per TFLite mode.
* :func:`plot_training_history`  — Train loss + val AUC over epochs.

Design notes:

* No seaborn dep. We rely on plain matplotlib with a small :data:`PALETTE`.
* ``setup_thesis_style`` applies rcParams once; callers can override per figure.
* Missing data keys are logged as warnings — the plotting function still
  returns a figure with whatever rows/points it could build, so a partial
  run doesn't bring the whole figure script down.

Called by:
    scripts/08_generate_figures.py
    notebooks/04_results_analysis.ipynb (optional)
Reads / Writes: none directly — all IO happens in the caller.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Small ColorBrewer-inspired palette with decent contrast on paper + screen.
PALETTE: dict[str, str] = {
    "baseline": "#4c72b0",
    "ewc": "#dd8452",
    "lwf": "#55a467",
    "replay": "#c44e52",
    "fp32": "#4c72b0",
    "fp16": "#dd8452",
    "int8": "#c44e52",
    "accent": "#8172b2",
    "muted": "#7f7f7f",
}

_METHOD_ORDER = ("ewc", "lwf", "replay")
_GENERATION_ORDER = ("gen1", "gen2", "gen3")
_MODE_ORDER = ("fp32", "fp16", "int8")


# --------------------------------------------------------------------------- #
#  Style setup
# --------------------------------------------------------------------------- #
def setup_thesis_style() -> None:
    """Apply the matplotlib rcParams used across all thesis figures."""
    plt.rcParams.update(
        {
            "figure.figsize": (7.0, 4.5),
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "legend.frameon": False,
            "legend.fontsize": 10,
            "lines.linewidth": 1.75,
            "lines.markersize": 6,
        }
    )


def save_figure(fig: Figure, path: str | Path, *, formats: Sequence[str] = ("png", "pdf")) -> list[Path]:
    """Save ``fig`` in every requested format alongside the given stem.

    ``path`` may carry a suffix (ignored) or not. Each format writes a
    sibling file with the matching extension.
    """
    base = Path(path).with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fmt in formats:
        out = base.with_suffix(f".{fmt}")
        fig.savefig(out, format=fmt)
        written.append(out)
    plt.close(fig)
    logger.info("Saved figure: %s (%s)", base, list(formats))
    return written


# --------------------------------------------------------------------------- #
#  Data containers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MethodRun:
    """Compact view of a single continual-distillation result JSON."""

    method: str
    generation: str
    per_gen_auc: dict[str, float]  # ``gen_id -> test AUC after this round``
    cgrs: float | None


def _short_method(name: str) -> str:
    return name.upper() if len(name) <= 3 else name.capitalize()


# --------------------------------------------------------------------------- #
#  Figure 1: Per-gen AUC × method comparison bar chart
# --------------------------------------------------------------------------- #
def plot_method_comparison(
    runs: Iterable[MethodRun],
    *,
    title: str = "Per-generation test AUC by anti-forgetting method",
    generations: Sequence[str] = _GENERATION_ORDER,
    baseline_auc: Mapping[str, float] | None = None,
) -> Figure:
    """Grouped bar chart of test AUC per generation, one group per method.

    Args:
        runs: Iterable of :class:`MethodRun` — one per (method, final-gen)
            evaluation. Only the *latest* run per method is used (the one
            that reports on the most generations).
        title: Figure title.
        generations: Which generation columns to include, in order.
        baseline_auc: Optional ``gen_id -> AUC`` for a "no anti-forgetting"
            control to draw behind the bars. Skipped when ``None``.
    """
    runs_list = list(runs)
    latest_by_method: dict[str, MethodRun] = {}
    for run in runs_list:
        incumbent = latest_by_method.get(run.method)
        if incumbent is None or len(run.per_gen_auc) >= len(incumbent.per_gen_auc):
            latest_by_method[run.method] = run

    methods = [m for m in _METHOD_ORDER if m in latest_by_method]
    methods += [m for m in latest_by_method if m not in _METHOD_ORDER]

    fig, ax = plt.subplots()
    x = np.arange(len(generations))
    total = max(len(methods), 1)
    bar_width = 0.8 / total
    for i, method in enumerate(methods):
        run = latest_by_method[method]
        heights = [float(run.per_gen_auc.get(gen, 0.0)) for gen in generations]
        offset = (i - (total - 1) / 2.0) * bar_width
        ax.bar(
            x + offset,
            heights,
            width=bar_width * 0.92,
            color=PALETTE.get(method, PALETTE["accent"]),
            label=_short_method(method),
            edgecolor="white",
            linewidth=0.6,
        )

    if baseline_auc:
        for idx, gen in enumerate(generations):
            if gen not in baseline_auc:
                continue
            ax.hlines(
                y=baseline_auc[gen],
                xmin=idx - 0.45,
                xmax=idx + 0.45,
                colors=PALETTE["muted"],
                linestyles=":",
                linewidth=1.2,
                label="no-AF" if idx == 0 else None,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(generations)
    ax.set_ylabel("Test AUC")
    ax.set_ylim(0.5, 1.0)
    ax.set_title(title)
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
#  Figure 2: CGRS trajectory line plot
# --------------------------------------------------------------------------- #
def plot_cgrs_trajectory(
    per_method_runs: Mapping[str, Sequence[MethodRun]],
    *,
    title: str = "Cross-Generational Retention Score trajectory",
) -> Figure:
    """Line plot: CGRS per training round, one line per method.

    Each method's list of runs should be ordered by training generation
    (gen2, then gen3, etc.). Rounds without a reported CGRS are skipped.
    """
    fig, ax = plt.subplots()
    for method, runs in per_method_runs.items():
        xs: list[str] = []
        ys: list[float] = []
        for run in runs:
            if run.cgrs is None:
                continue
            xs.append(run.generation)
            ys.append(float(run.cgrs))
        if not xs:
            logger.warning("No CGRS values to plot for method=%s", method)
            continue
        ax.plot(
            xs,
            ys,
            marker="o",
            color=PALETTE.get(method, PALETTE["accent"]),
            label=_short_method(method),
        )
    ax.axhline(y=1.0, color=PALETTE["muted"], linestyle=":", linewidth=1.0)
    ax.set_ylabel("CGRS")
    ax.set_xlabel("Training round (generation just trained on)")
    ax.set_ylim(0.5, 1.05)
    ax.set_title(title)
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
#  Figure 3: Ablation sweep line plot
# --------------------------------------------------------------------------- #
def plot_ablation_sweep(
    summary: Mapping[str, Any],
    *,
    x_key: str,
    x_label: str,
    metric: str = "auc",
    split_name: str | None = None,
    title: str | None = None,
) -> Figure:
    """Plot a metric vs a swept parameter from an ablation summary JSON.

    Args:
        summary: The ``{ablation}_summary.json`` payload (dict).
        x_key: Field inside each variant to use as the x axis — either a
            top-level variant key (e.g. ``"name"``) or a dotted path into
            ``config_overrides`` (e.g. ``"training.initial_distillation.temperature"``).
            If ``x_key == "name"`` we sort by variant name.
        x_label: Axis label.
        metric: Which ``splits[split]`` metric to plot. Defaults to ``auc``.
        split_name: Which split to read (e.g. ``gen1_test``, ``gen2_test``).
            When ``None``, falls back to the single split present.
        title: Optional figure title; defaults to the ablation's label.
    """
    fig, ax = plt.subplots()
    runs = summary.get("runs") or []
    if not runs:
        logger.warning("Ablation summary has no runs — empty plot")
        return fig

    # Aggregate per variant: mean ± std across seeds.
    grouped: dict[str, list[float]] = {}
    x_values: dict[str, float | str] = {}
    for row in runs:
        if row.get("status") != "ok":
            continue
        variant = row["variant"]
        splits = row.get("splits") or {}
        split_key = split_name or next(iter(splits), None)
        if split_key is None or split_key not in splits:
            continue
        value = splits[split_key].get(metric)
        if value is None:
            continue
        grouped.setdefault(variant, []).append(float(value))
        x_values.setdefault(variant, _extract_x(row, variant, x_key, summary))

    variants = sorted(
        grouped,
        key=lambda v: x_values[v] if isinstance(x_values[v], (int, float)) else v,
    )
    xs = [x_values[v] for v in variants]
    means = [float(np.mean(grouped[v])) for v in variants]
    stds = [float(np.std(grouped[v])) if len(grouped[v]) > 1 else 0.0 for v in variants]

    ax.errorbar(
        xs,
        means,
        yerr=stds,
        marker="o",
        capsize=3,
        color=PALETTE["baseline"],
        ecolor=PALETTE["muted"],
    )
    if xs and all(isinstance(v, str) for v in xs):
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(list(xs), rotation=30, ha="right")
    ax.set_xlabel(x_label)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title or summary.get("label", summary.get("ablation", "Ablation")))
    fig.tight_layout()
    return fig


def _extract_x(
    row: Mapping[str, Any],
    variant: str,
    x_key: str,
    summary: Mapping[str, Any],
) -> float | str:
    """Best-effort extraction of the numeric x for a variant.

    For ``x_key == "name"`` we just return the variant name. Otherwise we
    look the value up in ``summary["variants"][*].description``-carried
    overrides — but since the summary doesn't echo overrides directly, we
    fall back to parsing trailing numbers out of the variant name (works
    for ``t_4``, ``buf_05pct``, ``h256_d02`` style names).
    """
    if x_key == "name":
        return variant
    # Parse any float at the tail of the variant name, e.g. "t_4" -> 4.0.
    for chunk in reversed(variant.replace("pct", "").split("_")):
        cleaned = chunk.lstrip("abcdefghijklmnopqrstuvwxyz")
        try:
            return float(cleaned)
        except ValueError:
            continue
    return variant


# --------------------------------------------------------------------------- #
#  Figure 4: Edge Pareto (latency vs AUC per TFLite mode)
# --------------------------------------------------------------------------- #
def plot_edge_pareto(
    edge_summaries: Mapping[str, Mapping[str, Any]],
    *,
    title: str = "Edge Pareto: latency vs AUC by quantization mode",
) -> Figure:
    """Scatter latency (x) vs AUC (y), one marker per (generation, mode).

    Args:
        edge_summaries: ``run_tag -> edge_metrics.json payload``. ``run_tag``
            is the key in the filename, e.g. ``gen1`` or ``gen3_replay``.
    """
    fig, ax = plt.subplots()
    for run_tag, summary in edge_summaries.items():
        modes = summary.get("modes", [])
        for entry in modes:
            mode = entry.get("mode", "")
            color = PALETTE.get(mode, PALETTE["accent"])
            ax.scatter(
                entry.get("latency_ms_mean"),
                entry.get("auc"),
                color=color,
                edgecolor="white",
                linewidth=0.6,
                s=70,
                label=f"{run_tag}/{mode}",
            )
            ax.annotate(
                f"{run_tag}\n{mode}",
                xy=(entry.get("latency_ms_mean", 0.0), entry.get("auc", 0.0)),
                xytext=(6, 4),
                textcoords="offset points",
                fontsize=8,
                color=PALETTE["muted"],
            )
    ax.set_xlabel("Mean inference latency (ms, 1-thread CPU)")
    ax.set_ylabel("Test AUC")
    ax.set_title(title)
    # Deduplicate legend labels (same mode seen multiple times).
    handles, labels = ax.get_legend_handles_labels()
    seen: dict[str, Any] = {}
    for h, lbl in zip(handles, labels):
        mode = lbl.split("/")[-1]
        seen.setdefault(mode, h)
    if seen:
        ax.legend(seen.values(), seen.keys(), loc="lower left", title="mode")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
#  Figure 5: Training history (train loss + val AUC)
# --------------------------------------------------------------------------- #
def plot_training_history(
    history: Sequence[Mapping[str, Any]],
    *,
    title: str = "Training history",
    loss_key: str = "train_loss",
    auc_key: str = "val_auc",
) -> Figure:
    """Twin-axis plot of train loss (left) and val AUC (right) vs epoch."""
    if not history:
        logger.warning("Empty history passed to plot_training_history")
        return plt.figure()

    epochs = [int(h.get("epoch", i)) for i, h in enumerate(history)]
    losses = [float(h.get(loss_key, float("nan"))) for h in history]
    aucs = [float(h.get(auc_key, float("nan"))) for h in history]

    fig, ax_loss = plt.subplots()
    ax_loss.plot(
        epochs, losses, color=PALETTE["baseline"], marker="o", label=loss_key
    )
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel(loss_key.replace("_", " ").title(), color=PALETTE["baseline"])
    ax_loss.tick_params(axis="y", labelcolor=PALETTE["baseline"])

    ax_auc = ax_loss.twinx()
    ax_auc.plot(
        epochs,
        aucs,
        color=PALETTE["replay"],
        marker="s",
        label=auc_key,
        linestyle="--",
    )
    ax_auc.set_ylabel(auc_key.replace("_", " ").title(), color=PALETTE["replay"])
    ax_auc.tick_params(axis="y", labelcolor=PALETTE["replay"])
    ax_auc.grid(False)
    ax_auc.set_ylim(0.5, 1.0)

    ax_loss.set_title(title)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
#  Construction helpers
# --------------------------------------------------------------------------- #
def method_run_from_continual_json(payload: Mapping[str, Any]) -> MethodRun | None:
    """Build a :class:`MethodRun` from a ``{gen}_{method}_continual_metrics.json``."""
    method = payload.get("method")
    generation = payload.get("generation")
    if not method or not generation:
        logger.warning("Missing method/generation in payload — skipping")
        return None
    auc_after = payload.get("auc_after") or {}
    if not auc_after:
        splits = payload.get("splits") or {}
        auc_after = {
            key.replace("_test", ""): float(val.get("auc", 0.0))
            for key, val in splits.items()
            if key.endswith("_test")
        }
    cgrs = payload.get("cgrs")
    return MethodRun(
        method=str(method),
        generation=str(generation),
        per_gen_auc={str(k): float(v) for k, v in auc_after.items()},
        cgrs=float(cgrs) if cgrs is not None else None,
    )


def method_run_from_initial_json(payload: Mapping[str, Any]) -> MethodRun | None:
    """Build a :class:`MethodRun` for the gen1 initial-distillation baseline."""
    generation = payload.get("generation") or "gen1"
    splits = payload.get("splits") or {}
    per_gen = {
        key.replace("_test", ""): float(val.get("auc", 0.0))
        for key, val in splits.items()
        if key.endswith("_test")
    }
    if not per_gen:
        return None
    return MethodRun(
        method="baseline",
        generation=str(generation),
        per_gen_auc=per_gen,
        cgrs=None,
    )
