"""Aggregate per-seed metrics JSONs into mean +/- std for thesis reporting.

CLI entrypoint. Not imported by other code.

When ``scripts/04_initial_distillation.py`` or
``scripts/05_continual_distillation.py`` is run with ``--num-seeds N``
(N > 1), each seed writes its own metrics JSON with a ``_seed{N}`` suffix.
This aggregator scans those per-seed files, groups them by
``(generation, method)``, and writes a single ``*_aggregated.json`` per
group with mean and std for headline metrics:

  - ``best_val_auc``
  - ``cgrs``                          (continual only)
  - ``avg_forgetting_all``            (continual only)
  - ``avg_forgetting_prev``           (continual only)
  - ``auc_after.<gen>``               (per-generation post-update AUC)
  - test split AUC / accuracy / log_loss

Per-seed values are also kept in the output for full transparency, so
downstream figure scripts can compute alternative aggregations.

Usage:
    # Aggregate every multi-seed run found under {drive}/results/raw/
    python scripts/aggregate_seeds.py --config configs/local.yaml

    # Limit to a specific generation+method combo
    python scripts/aggregate_seeds.py --filter gen2_replay+ewc

    # Override results dir
    python scripts/aggregate_seeds.py --results-dir /content/ckd_local/results/raw

Reads:
    {results_dir}/{gen}_initial_metrics_seed*.json
    {results_dir}/{gen}_{method}_continual_metrics_seed*.json
Writes:
    {results_dir}/{gen}_initial_metrics_aggregated.json
    {results_dir}/{gen}_{method}_continual_metrics_aggregated.json
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.utils.config import load_config  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

# Regex captures (group_key, seed). Examples:
#   gen1_initial_metrics_seed0.json     → group="gen1_initial_metrics",     seed=0
#   gen2_replay+ewc_continual_metrics_seed2.json
#                                       → group="gen2_replay+ewc_continual_metrics", seed=2
_SEED_FILE_RE = re.compile(r"^(.+)_seed(\d+)\.json$")


def _group_seed_files(results_dir: Path, *, name_filter: str | None) -> dict[str, list[tuple[int, Path]]]:
    """Walk results_dir, group per-seed JSONs by their base name."""
    groups: dict[str, list[tuple[int, Path]]] = {}
    for path in sorted(results_dir.glob("*_seed*.json")):
        match = _SEED_FILE_RE.match(path.name)
        if not match:
            continue
        group_key = match.group(1)
        seed = int(match.group(2))
        if name_filter and name_filter not in group_key:
            continue
        groups.setdefault(group_key, []).append((seed, path))
    # Sort each group's files by seed for deterministic output ordering
    for key in groups:
        groups[key].sort(key=lambda t: t[0])
    return groups


def _mean_std(values: list[float]) -> dict[str, float]:
    """Return {mean, std, n}. std uses sample stdev (Bessel-corrected) when n>1."""
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    mean = statistics.fmean(values)
    if len(values) >= 2:
        std = statistics.stdev(values)
    else:
        std = 0.0
    return {"mean": float(mean), "std": float(std), "n": len(values)}


def _collect_scalar(per_seed: list[dict[str, Any]], path: list[str]) -> list[float] | None:
    """Pull a scalar at a nested key path (e.g. ['cgrs']) from each per-seed dict.

    Returns None if the key path is missing in *any* seed (so we can skip
    metrics that don't exist for this run type — e.g. cgrs only exists in
    continual metrics).
    """
    out: list[float] = []
    for d in per_seed:
        cur: Any = d
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                return None
            cur = cur[key]
        if cur is None:
            return None
        try:
            out.append(float(cur))
        except (TypeError, ValueError):
            return None
    return out


def _collect_split_metric(
    per_seed: list[dict[str, Any]], split_name: str, metric_name: str,
) -> list[float] | None:
    """Pull `splits[split_name][metric_name]` from each seed."""
    out: list[float] = []
    for d in per_seed:
        splits = d.get("splits") or {}
        sp = splits.get(split_name) or {}
        if metric_name not in sp:
            return None
        try:
            out.append(float(sp[metric_name]))
        except (TypeError, ValueError):
            return None
    return out


def _peak_test_auc_for_gen(
    results_dir: Path, gen: str, *, exclude_paths: set[str] | None = None,
) -> tuple[float | None, str | None]:
    """Find peak test AUC for a generation across ALL metrics JSONs.

    Used to recompute CGRS post-hoc when scripts/05's
    ``_peak_auc_from_previous_runs`` couldn't find peak (e.g. multi-seed
    glob mismatch in older script versions, leading to cgrs=1.0 in
    every per-seed continual JSON).

    Looks at both single-seed and multi-seed naming patterns. For a
    given generation ``gen``, the peak is the max ``splits.{gen}_test.auc``
    across all matched files — that's the best the model ever achieved
    on this generation's held-out test set.

    Args:
        exclude_paths: Optional set of file paths to ignore. Useful when
            we want the peak-from-PRIOR-rounds (excluding files written
            during the current generation's training, where retention has
            already started degrading).

    Returns: (peak_auc or None, source_file or None) for traceability.
    """
    patterns = [
        f"{gen}_initial_metrics.json",
        f"{gen}_initial_metrics_seed*.json",
        f"{gen}_*_continual_metrics.json",
        f"{gen}_*_continual_metrics_seed*.json",
    ]
    best: float | None = None
    best_source: str | None = None
    exclude_paths = exclude_paths or set()
    for pat in patterns:
        for path in results_dir.glob(pat):
            if str(path) in exclude_paths:
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            splits = payload.get("splits") or {}
            sp = splits.get(f"{gen}_test") or {}
            auc = sp.get("auc")
            if auc is None:
                continue
            try:
                auc_f = float(auc)
            except (TypeError, ValueError):
                continue
            if best is None or auc_f > best:
                best = auc_f
                best_source = str(path)
    return best, best_source


def _recompute_cgrs_from_peaks(
    per_seed_payloads: list[dict[str, Any]],
    *,
    results_dir: Path,
    current_generation: str | None,
    logger,
) -> dict[str, Any]:
    """Post-hoc CGRS recomputation: for each seed in a continual run,
    re-derive CGRS / avg_forgetting from ``auc_after`` + freshly-looked-up
    per-gen peaks.

    Why: legacy scripts/05 writes wrong CGRS when peak lookup glob misses
    multi-seed files. We don't trust the cgrs/avg_forgetting fields in the
    per-seed JSONs. We DO trust ``auc_after`` (which is computed correctly
    from the trainer's eval pass) and the test AUCs in initial/earlier
    continual metrics files.

    Returns a partial dict to merge into the aggregated payload, with keys:
      - peaks_used_for_correction: {gen: {auc, source}}
      - cgrs_corrected: {mean, std, n}
      - cgrs_corrected_per_seed: list[float]
      - avg_forgetting_all_corrected: {mean, std, n}
      - avg_forgetting_prev_corrected: {mean, std, n}
    """
    # Collect every gen mentioned in any seed's auc_after dict.
    gen_keys: set[str] = set()
    for p in per_seed_payloads:
        aa = p.get("auc_after") or {}
        gen_keys.update(aa.keys())
    if not gen_keys:
        return {}

    # Look up peak for each gen across all metrics files. We exclude the
    # current group's own JSONs from peak lookup so a continual run's own
    # post-update AUC doesn't get used as the "peak" for the gen we just
    # trained on (that would just reproduce auc_after = peak = 1.0 ratios).
    own_paths = {p.get("_source_file") for p in per_seed_payloads if p.get("_source_file")}
    peaks: dict[str, dict[str, Any]] = {}
    for gen in sorted(gen_keys):
        peak, source = _peak_test_auc_for_gen(
            results_dir, gen, exclude_paths=own_paths,
        )
        if peak is not None:
            peaks[gen] = {"auc": peak, "source": source}

    if not peaks:
        return {}

    # Per-seed correct CGRS / avg_forgetting
    cgrs_per_seed: list[float] = []
    forg_all_per_seed: list[float] = []
    forg_prev_per_seed: list[float] = []
    for p in per_seed_payloads:
        aa = p.get("auc_after") or {}
        ratios: list[float] = []
        forg_all: list[float] = []
        forg_prev: list[float] = []
        for gen, auc_after_g in aa.items():
            peak_info = peaks.get(gen)
            if peak_info is None or peak_info["auc"] <= 0:
                continue
            try:
                a = float(auc_after_g)
            except (TypeError, ValueError):
                continue
            peak = peak_info["auc"]
            ratios.append(a / peak)
            drop = peak - a
            forg_all.append(drop)
            if current_generation is None or gen != current_generation:
                forg_prev.append(drop)
        if ratios:
            cgrs_per_seed.append(sum(ratios) / len(ratios))
        if forg_all:
            forg_all_per_seed.append(sum(forg_all) / len(forg_all))
        if forg_prev:
            forg_prev_per_seed.append(sum(forg_prev) / len(forg_prev))

    out: dict[str, Any] = {
        "peaks_used_for_correction": peaks,
    }
    if cgrs_per_seed:
        out["cgrs_corrected"] = _mean_std(cgrs_per_seed)
        out["cgrs_corrected_per_seed"] = cgrs_per_seed
    if forg_all_per_seed:
        out["avg_forgetting_all_corrected"] = _mean_std(forg_all_per_seed)
        out["avg_forgetting_all_corrected_per_seed"] = forg_all_per_seed
    if forg_prev_per_seed:
        out["avg_forgetting_prev_corrected"] = _mean_std(forg_prev_per_seed)
        out["avg_forgetting_prev_corrected_per_seed"] = forg_prev_per_seed
    return out


def _aggregate_one_group(
    group_key: str, files: list[tuple[int, Path]], logger,
) -> dict[str, Any]:
    """Build aggregate dict for one (gen, method) group."""
    per_seed_payloads: list[dict[str, Any]] = []
    for seed, fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to read %s: %s — skipping", fp, exc)
            continue
        payload["_seed"] = seed
        payload["_source_file"] = str(fp)
        per_seed_payloads.append(payload)

    if not per_seed_payloads:
        return {}

    aggregated: dict[str, Any] = {
        "group_key": group_key,
        "num_seeds": len(per_seed_payloads),
        "seeds": [p["_seed"] for p in per_seed_payloads],
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # Inherit a few static fields from seed-0 (assumed identical across seeds)
    first = per_seed_payloads[0]
    for key in ("generation", "method", "previous_generation"):
        if key in first:
            aggregated[key] = first[key]

    # Headline scalars (None-safe — skip if missing in any seed)
    for label, json_path in [
        ("best_val_auc", ["best_val_auc"]),
        ("cgrs", ["cgrs"]),
        ("avg_forgetting_all", ["avg_forgetting_all"]),
        ("avg_forgetting_prev", ["avg_forgetting_prev"]),
        ("elapsed_seconds", ["elapsed_seconds"]),
        ("gpu_hours", ["gpu_hours"]),
    ]:
        values = _collect_scalar(per_seed_payloads, json_path)
        if values is not None:
            aggregated[label] = _mean_std(values)
            aggregated[f"{label}_per_seed"] = values

    # Per-generation post-update AUC (continual only)
    auc_after_keys: set[str] = set()
    for p in per_seed_payloads:
        if isinstance(p.get("auc_after"), dict):
            auc_after_keys.update(p["auc_after"].keys())
    if auc_after_keys:
        agg_auc_after: dict[str, dict[str, float]] = {}
        for gen in sorted(auc_after_keys):
            values = []
            for p in per_seed_payloads:
                aa = p.get("auc_after") or {}
                if gen in aa:
                    try:
                        values.append(float(aa[gen]))
                    except (TypeError, ValueError):
                        pass
            if values:
                agg_auc_after[gen] = _mean_std(values)
        aggregated["auc_after"] = agg_auc_after

    # Per-split test metrics (works for both initial and continual)
    splits_aggregate: dict[str, dict[str, dict[str, float]]] = {}
    test_split_names: set[str] = set()
    for p in per_seed_payloads:
        splits = p.get("splits") or {}
        for sname in splits:
            if sname.endswith("_test"):
                test_split_names.add(sname)
    for sname in sorted(test_split_names):
        per_metric: dict[str, dict[str, float]] = {}
        for metric in ("auc", "log_loss", "accuracy"):
            values = _collect_split_metric(per_seed_payloads, sname, metric)
            if values is not None:
                per_metric[metric] = _mean_std(values)
        if per_metric:
            splits_aggregate[sname] = per_metric
    if splits_aggregate:
        aggregated["splits"] = splits_aggregate

    # Post-hoc corrected CGRS / forgetting — works around the legacy
    # scripts/05 _peak_auc_from_previous_runs glob mismatch that wrote
    # cgrs=1.0 / avg_forgetting=0.0 in every multi-seed continual JSON.
    # We re-derive these from auc_after + per-gen peaks looked up across
    # ALL per-seed metrics files. Original (broken) cgrs and
    # avg_forgetting fields are still aggregated above for reference.
    if auc_after_keys:
        # Need the directory containing the per-seed JSONs
        first_path = first.get("_source_file")
        if first_path:
            _results_dir = Path(first_path).parent
            current_gen = aggregated.get("generation")
            corrected = _recompute_cgrs_from_peaks(
                per_seed_payloads,
                results_dir=_results_dir,
                current_generation=current_gen,
                logger=None,
            )
            aggregated.update(corrected)

    return aggregated


def _run(args: argparse.Namespace) -> int:
    logger = get_logger(
        "aggregate_seeds",
        log_file="runs/aggregate_seeds.log",
    )

    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        cfg = load_config(args.config)
        drive_root = Path(cfg["paths"]["drive_root"])
        results_dir = drive_root / "results" / "raw"

    if not results_dir.is_dir():
        logger.error("Results dir not found: %s", results_dir)
        return 1

    groups = _group_seed_files(results_dir, name_filter=args.filter)
    if not groups:
        logger.warning(
            "No per-seed metrics found under %s (looking for *_seed*.json). "
            "Did any script run with --num-seeds > 1?",
            results_dir,
        )
        return 0

    logger.info("Aggregating %d group(s) from %s", len(groups), results_dir)
    written = 0
    for group_key, files in groups.items():
        logger.info("  %s: %d seeds", group_key, len(files))
        agg = _aggregate_one_group(group_key, files, logger)
        if not agg:
            continue
        out_path = results_dir / f"{group_key}_aggregated.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(agg, f, indent=2)
        written += 1
        # Print a one-line summary
        if "best_val_auc" in agg:
            bva = agg["best_val_auc"]
            logger.info(
                "    -> %s  (best_val_auc mean=%.4f std=%.4f n=%d)",
                out_path.name, bva["mean"], bva["std"], bva["n"],
            )
        else:
            logger.info("    -> %s", out_path.name)
        if "cgrs" in agg:
            c = agg["cgrs"]
            logger.info(
                "       CGRS (raw, may be 1.0 due to legacy bug) mean=%.4f std=%.4f",
                c["mean"], c["std"],
            )
        if "cgrs_corrected" in agg:
            cc = agg["cgrs_corrected"]
            logger.info(
                "       CGRS (CORRECTED post-hoc) mean=%.4f std=%.4f n=%d",
                cc["mean"], cc["std"], cc["n"],
            )
        if "avg_forgetting_prev_corrected" in agg:
            af = agg["avg_forgetting_prev_corrected"]
            logger.info(
                "       AvgF_prev (CORRECTED) mean=%.4f std=%.4f",
                af["mean"], af["std"],
            )
        if "peaks_used_for_correction" in agg:
            peaks_str = ", ".join(
                f"{g}={info['auc']:.4f}"
                for g, info in agg["peaks_used_for_correction"].items()
            )
            logger.info("       peaks_used: %s", peaks_str)

    logger.info("Done. %d aggregated JSON(s) written.", written)
    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate per-seed metrics into mean +/- std summaries."
    )
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument(
        "--results-dir",
        default=None,
        help=(
            "Override the results directory (default: "
            "{drive}/results/raw)."
        ),
    )
    p.add_argument(
        "--filter",
        default=None,
        help=(
            "Only aggregate groups whose key contains this substring "
            "(e.g. 'gen2_replay+ewc' to skip other generations/methods)."
        ),
    )
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(_run(_parse_args()))
