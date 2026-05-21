"""CoreML conversion, latency benchmarking, and accuracy evaluation.

Companion module to ``edge_eval.py`` — provides the Apple-Silicon-native
deployment path that targets Neural Engine (ANE) + GPU via CoreML.

The pipeline mirrors the TFLite flow but uses ``coremltools`` directly
from a JIT-traced PyTorch model (cleaner than going via ONNX for
MobileNetV3 — coremltools 7+ supports hard-swish natively):

    PyTorch (.pth)  --torch.jit.trace-->  Traced TorchScript
                                            │
                                            ▼
                                       coremltools.convert
                                            │
                                            ▼
                                   {fp32, int8}.mlpackage
                                            │
                                            ▼
                                  ct.models.MLModel.predict
                                       (CPU + ANE + GPU)

All heavy deps (``torch``, ``coremltools``) are lazy-imported so the
module can be inspected without macOS-specific wheels installed.

Called by:
    scripts/edge/run_edge_eval_macbook.py
    notebooks/08_edge_evaluation_macbook.ipynb (via direct import)
Reads:
    A student ``nn.Module`` already loaded with weights by the caller.
    Representative dataset for INT8 calibration (iterable of NCHW batches).
Writes:
    ``{output_dir}/{mode}.mlpackage`` — CoreML model bundles
"""
from __future__ import annotations

import platform
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import BinaryMetrics, compute_binary_metrics
from src.utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_COREML_MODES: tuple[str, ...] = ("fp32", "int8")
COREML_COMPUTE_UNITS: dict[str, str] = {
    "all": "ALL",          # CPU + GPU + ANE (default, fastest)
    "cpu_only": "CPU_ONLY",
    "cpu_gpu": "CPU_AND_GPU",
    "cpu_ane": "CPU_AND_NE",  # CPU + Neural Engine only
}


@dataclass(frozen=True)
class CoreMLBenchmarkResult:
    """One (mode, model) CoreML benchmark row."""

    mode: str
    compute_unit: str
    mlpackage_path: Path
    size_mb: float
    latency_ms_mean: float
    latency_ms_p50: float
    latency_ms_p95: float
    latency_ms_p99: float
    num_runs: int
    num_warmup: int
    auc: float
    log_loss: float
    accuracy: float
    num_samples: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "compute_unit": self.compute_unit,
            "mlpackage_path": str(self.mlpackage_path),
            "size_mb": float(self.size_mb),
            "latency_ms_mean": float(self.latency_ms_mean),
            "latency_ms_p50": float(self.latency_ms_p50),
            "latency_ms_p95": float(self.latency_ms_p95),
            "latency_ms_p99": float(self.latency_ms_p99),
            "num_runs": int(self.num_runs),
            "num_warmup": int(self.num_warmup),
            "auc": float(self.auc),
            "log_loss": float(self.log_loss),
            "accuracy": float(self.accuracy),
            "num_samples": int(self.num_samples),
        }


def _assert_macos() -> None:
    """CoreML predict path only works on macOS (the Foundation framework).

    Conversion (``coremltools.convert``) works cross-platform; runtime
    inference does not. The notebook is expected to run on a MacBook so
    this is a friendly upfront error.
    """
    if platform.system() != "Darwin":
        raise RuntimeError(
            "CoreML inference requires macOS. Running on "
            f"{platform.system()!r}. Conversion works cross-platform but "
            "predict() does not."
        )


def _trace_for_coreml(
    model: nn.Module, input_size: int = 224
) -> tuple[Any, np.ndarray]:
    """JIT-trace the model and return (traced_module, example_input_numpy).

    coremltools expects either a TorchScript model or an ONNX model. JIT
    trace is the cleanest path for MobileNetV3-Large + a simple linear head.
    """
    model = model.eval().cpu()
    example = torch.zeros((1, 3, input_size, input_size), dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(model, example)
    return traced, example.numpy()


def export_coreml(
    model: nn.Module,
    output_path: str | Path,
    *,
    input_size: int = 224,
    mode: str = "fp32",
    representative_loader: DataLoader | None = None,
    calibration_batches: int = 16,
    compute_precision: str = "FLOAT32",
    minimum_deployment_target: str | None = None,
) -> Path:
    """Convert ``model`` to a CoreML ``.mlpackage`` bundle.

    Args:
        model: Already-loaded student ``nn.Module``.
        output_path: Where to write the ``.mlpackage`` (will be replaced
            if it exists).
        input_size: Input H/W (square crops).
        mode: One of ``SUPPORTED_COREML_MODES``. ``"int8"`` triggers
            post-training int8 weight quantization (activations stay fp16
            on ANE — that's coremltools' default for int8 mode).
        representative_loader: Required for ``mode="int8"``. Used for
            activation calibration via coremltools' OptimizationConfig.
        calibration_batches: Cap on calibration batches consumed.
        compute_precision: ``"FLOAT32"`` or ``"FLOAT16"`` — affects
            intermediate activations on ANE.
        minimum_deployment_target: e.g. ``"iOS16"`` or ``"macOS13"``.
            Set when targeting a specific iOS version; ``None`` lets
            coremltools pick a sensible default.

    Returns:
        Path to the written ``.mlpackage``.
    """
    import coremltools as ct  # lazy: heavy macOS-specific dep

    mode = mode.lower()
    if mode not in SUPPORTED_COREML_MODES:
        raise ValueError(
            f"Unsupported CoreML mode {mode!r}. Expected one of "
            f"{SUPPORTED_COREML_MODES}."
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    traced, example = _trace_for_coreml(model, input_size=input_size)

    # Image input spec — explicit ImageType makes ANE acceleration
    # available for typical RGB inputs. We use generic NCHW float tensor
    # spec to match our PyTorch preprocessing exactly (already normalized).
    inputs = [
        ct.TensorType(
            name="input",
            shape=(1, 3, input_size, input_size),
            dtype=np.float32,
        )
    ]
    outputs = [ct.TensorType(name="logits", dtype=np.float32)]

    precision_map = {
        "FLOAT32": ct.precision.FLOAT32,
        "FLOAT16": ct.precision.FLOAT16,
    }
    convert_kwargs: dict[str, Any] = {
        "inputs": inputs,
        "outputs": outputs,
        "compute_precision": precision_map.get(
            compute_precision.upper(), ct.precision.FLOAT32
        ),
        "convert_to": "mlprogram",  # modern .mlpackage format
    }
    if minimum_deployment_target is not None:
        target_map = {
            "iOS15": ct.target.iOS15,
            "iOS16": ct.target.iOS16,
            "iOS17": ct.target.iOS17,
            "macOS13": ct.target.macOS13,
            "macOS14": ct.target.macOS14,
        }
        if minimum_deployment_target in target_map:
            convert_kwargs["minimum_deployment_target"] = target_map[
                minimum_deployment_target
            ]

    logger.info(
        "CoreML convert: mode=%s precision=%s target=%s -> %s",
        mode,
        compute_precision,
        minimum_deployment_target or "auto",
        output_path,
    )
    mlmodel = ct.convert(traced, **convert_kwargs)

    if mode == "int8":
        try:
            # coremltools 7+ post-training int8 weight quantization.
            from coremltools.optimize.coreml import (  # type: ignore[import-not-found]
                OpLinearQuantizerConfig,
                OptimizationConfig,
                linear_quantize_weights,
            )

            op_config = OpLinearQuantizerConfig(
                mode="linear_symmetric",
                dtype="int8",
                granularity="per_channel",
            )
            opt_config = OptimizationConfig(global_config=op_config)
            mlmodel = linear_quantize_weights(mlmodel, config=opt_config)
            logger.info(
                "CoreML int8 weight quantization applied (per_channel "
                "linear_symmetric)"
            )
        except ImportError as exc:
            raise RuntimeError(
                "coremltools.optimize.coreml not available — upgrade "
                "coremltools to >= 7.0 for int8 quantization support."
            ) from exc

    # Remove existing if present (coremltools writes a directory bundle).
    if output_path.exists():
        import shutil

        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()
    mlmodel.save(str(output_path))
    logger.info(
        "CoreML saved: %s (%.2f MB)", output_path, _mlpackage_size_mb(output_path)
    )
    return output_path


def _mlpackage_size_mb(path: str | Path) -> float:
    """``.mlpackage`` is a directory bundle — sum all file sizes."""
    p = Path(path)
    if p.is_file():
        return float(p.stat().st_size) / (1024.0 * 1024.0)
    total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return float(total) / (1024.0 * 1024.0)


def _load_coreml_model(
    mlpackage_path: str | Path,
    *,
    compute_unit: str = "all",
) -> Any:
    """Load a ``.mlpackage`` with the requested compute unit selection."""
    _assert_macos()
    import coremltools as ct  # lazy

    cu_map = {
        "all": ct.ComputeUnit.ALL,
        "cpu_only": ct.ComputeUnit.CPU_ONLY,
        "cpu_gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu_ane": ct.ComputeUnit.CPU_AND_NE,
    }
    if compute_unit not in cu_map:
        raise ValueError(
            f"Unknown compute_unit {compute_unit!r}. Expected one of "
            f"{sorted(cu_map.keys())}."
        )
    return ct.models.MLModel(str(mlpackage_path), compute_units=cu_map[compute_unit])


def _predict_coreml(model: Any, image_nchw: np.ndarray) -> np.ndarray:
    """Run one prediction call. Input shape: (1, 3, H, W) float32."""
    out = model.predict({"input": image_nchw.astype(np.float32)})
    # Output keys may be 'logits' (as we set) or 'var_<n>' depending on
    # coremltools version — pick the first ndarray-valued key.
    for value in out.values():
        if isinstance(value, np.ndarray):
            return value
    raise RuntimeError(f"CoreML output had no ndarray value (keys={list(out.keys())})")


def _softmax_last_dim(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def benchmark_coreml(
    mlpackage_path: str | Path,
    *,
    input_size: int = 224,
    num_runs: int = 100,
    num_warmup: int = 10,
    compute_unit: str = "all",
) -> dict[str, float]:
    """Single-sample latency benchmark on the requested compute unit."""
    model = _load_coreml_model(mlpackage_path, compute_unit=compute_unit)
    dummy = np.zeros((1, 3, input_size, input_size), dtype=np.float32)

    for _ in range(num_warmup):
        _predict_coreml(model, dummy)

    latencies_ms: list[float] = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _predict_coreml(model, dummy)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    sorted_ms = sorted(latencies_ms)
    n = len(sorted_ms)
    return {
        "latency_ms_mean": float(statistics.fmean(sorted_ms)),
        "latency_ms_p50": float(statistics.median(sorted_ms)),
        "latency_ms_p95": float(sorted_ms[int(0.95 * (n - 1))]),
        "latency_ms_p99": float(sorted_ms[int(0.99 * (n - 1))]),
        "num_runs": int(num_runs),
        "num_warmup": int(num_warmup),
        "compute_unit": str(compute_unit),
    }


def evaluate_coreml(
    mlpackage_path: str | Path,
    loader: DataLoader,
    *,
    compute_unit: str = "all",
    fake_class_index: int = 1,
) -> BinaryMetrics:
    """Run the CoreML model over ``loader`` and compute AUC / log-loss / acc.

    Args:
        mlpackage_path: Path to the ``.mlpackage`` bundle.
        loader: Yields ``(images, hard_labels, soft_labels, meta)``.
        compute_unit: One of ``COREML_COMPUTE_UNITS`` keys.
        fake_class_index: Column treated as ``P(fake)``.
    """
    model = _load_coreml_model(mlpackage_path, compute_unit=compute_unit)

    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for batch in loader:
        images, hard_labels, *_ = batch
        # Pass through one sample at a time (coremltools predict() expects
        # batch=1 by default; batching would need a different model spec).
        batch_np = (
            images.detach().cpu().numpy().astype(np.float32, copy=False)
            if isinstance(images, torch.Tensor)
            else np.asarray(images, dtype=np.float32)
        )
        batch_probs = np.empty(batch_np.shape[0], dtype=np.float32)
        for i in range(batch_np.shape[0]):
            logits = _predict_coreml(model, batch_np[i : i + 1])
            probs = _softmax_last_dim(logits)
            batch_probs[i] = float(probs.reshape(-1)[fake_class_index])
        all_probs.append(batch_probs)

        labels_np = (
            hard_labels.detach().cpu().numpy()
            if isinstance(hard_labels, torch.Tensor)
            else np.asarray(hard_labels)
        )
        all_labels.append(labels_np.astype(np.int64))

    if not all_probs:
        raise RuntimeError("CoreML eval loader yielded zero batches")

    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    return compute_binary_metrics(y_true, y_prob)


def run_coreml_benchmark(
    model: nn.Module,
    *,
    eval_loader: DataLoader,
    output_dir: str | Path,
    representative_loader: DataLoader | None = None,
    modes: Iterable[str] = SUPPORTED_COREML_MODES,
    compute_unit: str = "all",
    input_size: int = 224,
    num_runs: int = 100,
    num_warmup: int = 10,
    calibration_batches: int = 16,
    minimum_deployment_target: str | None = None,
) -> list[CoreMLBenchmarkResult]:
    """End-to-end: convert to CoreML(s), benchmark + evaluate each."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[CoreMLBenchmarkResult] = []
    for mode in modes:
        mode_norm = mode.lower()
        mlpackage_path = output_dir / f"{mode_norm}.mlpackage"
        try:
            export_coreml(
                model,
                mlpackage_path,
                input_size=input_size,
                mode=mode_norm,
                representative_loader=representative_loader,
                calibration_batches=calibration_batches,
                compute_precision="FLOAT32" if mode_norm == "fp32" else "FLOAT16",
                minimum_deployment_target=minimum_deployment_target,
            )
        except Exception as exc:  # noqa: BLE001 — log + skip rather than abort all modes
            logger.error(
                "CoreML conversion failed for mode=%s: %s: %s",
                mode_norm,
                type(exc).__name__,
                exc,
            )
            continue

        try:
            latency = benchmark_coreml(
                mlpackage_path,
                input_size=input_size,
                num_runs=num_runs,
                num_warmup=num_warmup,
                compute_unit=compute_unit,
            )
            metrics = evaluate_coreml(
                mlpackage_path, eval_loader, compute_unit=compute_unit
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "CoreML runtime eval failed for mode=%s: %s: %s",
                mode_norm,
                type(exc).__name__,
                exc,
            )
            continue

        results.append(
            CoreMLBenchmarkResult(
                mode=mode_norm,
                compute_unit=compute_unit,
                mlpackage_path=mlpackage_path,
                size_mb=_mlpackage_size_mb(mlpackage_path),
                latency_ms_mean=latency["latency_ms_mean"],
                latency_ms_p50=latency["latency_ms_p50"],
                latency_ms_p95=latency["latency_ms_p95"],
                latency_ms_p99=latency["latency_ms_p99"],
                num_runs=latency["num_runs"],
                num_warmup=latency["num_warmup"],
                auc=metrics.auc,
                log_loss=metrics.log_loss,
                accuracy=metrics.accuracy,
                num_samples=metrics.num_samples,
            )
        )
        logger.info(
            "CoreML mode=%s cu=%s size=%.2fMB lat_mean=%.2fms p95=%.2fms "
            "auc=%.4f",
            mode_norm,
            compute_unit,
            results[-1].size_mb,
            results[-1].latency_ms_mean,
            results[-1].latency_ms_p95,
            results[-1].auc,
        )
    return results
