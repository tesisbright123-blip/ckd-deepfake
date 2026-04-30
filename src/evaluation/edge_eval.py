"""TFLite conversion, latency benchmarking, and accuracy evaluation for edge.

The pipeline for the CKD student is:

    PyTorch (.pth)  --torch.onnx-->  ONNX
                                     │
                                     ▼
                                 onnx2tf      (ONNX → TF SavedModel → TFLite)
                                     │
                                     ▼
                         {fp32, fp16, int8}.tflite
                                     │
                                     ▼
                       tflite.Interpreter (CPU latency + AUC)

All heavy deps (torch, onnx, onnx2tf, tensorflow / ai-edge-litert) are
lazy-imported so that unit tests / lightweight CLI paths can exercise the
module without pulling in the full Colab stack.

Called by:
    scripts/07_edge_evaluation.py
Reads:
    A student checkpoint (loaded by the caller); this module only sees the
    already-instantiated ``nn.Module``. Representative dataset for INT8
    calibration is supplied as an iterable of ``torch.Tensor`` batches.
Writes:
    ``{output_dir}/model.onnx``
    ``{output_dir}/tflite/{fp32,fp16,int8}.tflite`` (naming normalized
    from onnx2tf's conventional outputs)
"""
from __future__ import annotations

import shutil
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

# onnx2tf default filenames for the TFLite outputs.
_ONNX2TF_FILENAMES: dict[str, tuple[str, ...]] = {
    "fp32": ("model_float32.tflite",),
    "fp16": ("model_float16.tflite",),
    "int8": ("model_full_integer_quant.tflite", "model_integer_quant.tflite"),
}

# Supported quantization modes, ordered by expected fidelity.
SUPPORTED_MODES: tuple[str, ...] = ("fp32", "fp16", "int8")


# --------------------------------------------------------------------------- #
#  Benchmark result type
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EdgeBenchmarkResult:
    """One (mode, model) benchmark row."""

    mode: str
    tflite_path: Path
    size_mb: float
    latency_ms_mean: float
    latency_ms_p50: float
    latency_ms_p95: float
    num_runs: int
    num_warmup: int
    auc: float
    log_loss: float
    accuracy: float
    num_samples: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "tflite_path": str(self.tflite_path),
            "size_mb": float(self.size_mb),
            "latency_ms_mean": float(self.latency_ms_mean),
            "latency_ms_p50": float(self.latency_ms_p50),
            "latency_ms_p95": float(self.latency_ms_p95),
            "num_runs": int(self.num_runs),
            "num_warmup": int(self.num_warmup),
            "auc": float(self.auc),
            "log_loss": float(self.log_loss),
            "accuracy": float(self.accuracy),
            "num_samples": int(self.num_samples),
        }


# --------------------------------------------------------------------------- #
#  ONNX export
# --------------------------------------------------------------------------- #
def export_onnx(
    model: nn.Module,
    output_path: str | Path,
    *,
    input_size: int = 224,
    opset: int = 17,
    batch_size: int = 1,
) -> Path:
    """Export ``model`` to ONNX with a single NCHW image input.

    Uses a static batch dim by default (matches typical edge inference) but
    can be overridden. The exported graph has input name ``"input"`` and
    output name ``"logits"`` so downstream tools can reference them.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = model.eval().cpu()
    dummy = torch.zeros((batch_size, 3, input_size, input_size), dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        opset_version=opset,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        do_constant_folding=True,
    )
    logger.info(
        "Exported ONNX (opset=%d, input=%dx%d) to %s",
        opset,
        input_size,
        input_size,
        output_path,
    )
    return output_path


# --------------------------------------------------------------------------- #
#  TFLite conversion via onnx2tf
# --------------------------------------------------------------------------- #
def _collect_calibration(
    loader: Iterable[Any],
    *,
    max_batches: int,
) -> np.ndarray:
    """Concatenate a few batches of images into an NHWC float32 calibration tensor.

    The dataloader yields ``(images, hard_labels, soft_labels, meta)`` tuples
    where ``images`` is ``(B, 3, H, W)`` float32. TFLite expects NHWC so we
    permute on the fly.
    """
    chunks: list[np.ndarray] = []
    seen = 0
    for batch in loader:
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        if isinstance(images, torch.Tensor):
            array = images.detach().cpu().numpy().astype(np.float32, copy=False)
        else:
            array = np.asarray(images, dtype=np.float32)
        # Convert NCHW -> NHWC.
        if array.ndim == 4 and array.shape[1] == 3:
            array = np.transpose(array, (0, 2, 3, 1))
        chunks.append(array)
        seen += 1
        if seen >= max_batches:
            break
    if not chunks:
        raise RuntimeError("No calibration batches collected — loader is empty")
    return np.concatenate(chunks, axis=0)


def _discover_tflite_output(
    onnx2tf_output_dir: Path, candidate_names: tuple[str, ...]
) -> Path | None:
    for name in candidate_names:
        candidate = onnx2tf_output_dir / name
        if candidate.is_file():
            return candidate
    return None


def _normalize_tflite_outputs(
    onnx2tf_output_dir: Path,
    tflite_dir: Path,
    modes: Iterable[str],
) -> dict[str, Path]:
    """Move onnx2tf's default-named TFLite files to ``tflite_dir/{mode}.tflite``."""
    tflite_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, Path] = {}
    for mode in modes:
        src = _discover_tflite_output(onnx2tf_output_dir, _ONNX2TF_FILENAMES[mode])
        if src is None:
            logger.warning(
                "onnx2tf did not produce a TFLite file for mode=%s in %s",
                mode,
                onnx2tf_output_dir,
            )
            continue
        dst = tflite_dir / f"{mode}.tflite"
        shutil.copyfile(src, dst)
        mapping[mode] = dst
        logger.info("TFLite %s: %s (%.2f MB)", mode, dst, file_size_mb(dst))
    return mapping


def convert_to_tflite(
    onnx_path: str | Path,
    output_dir: str | Path,
    *,
    modes: Iterable[str] = SUPPORTED_MODES,
    representative_loader: DataLoader | None = None,
    calibration_batches: int = 16,
    input_name: str = "input",
) -> dict[str, Path]:
    """Convert an ONNX model to TFLite in one or more quantization modes.

    Uses ``onnx2tf`` which produces a TensorFlow SavedModel plus a set of
    TFLite files named by onnx2tf's own convention. We copy/rename the
    relevant ones to ``{output_dir}/{mode}.tflite``.

    Args:
        onnx_path: Path to the ONNX graph produced by :func:`export_onnx`.
        output_dir: Destination dir — an ``onnx2tf_out/`` temp subtree will
            hold onnx2tf's raw output; the final tflite files live directly
            under this directory.
        modes: Subset of ``SUPPORTED_MODES`` to produce.
        representative_loader: Needed for INT8; ignored for fp32/fp16. Each
            yielded batch contributes up to its full minibatch to the
            calibration set.
        calibration_batches: Cap on how many batches are consumed from the
            loader during INT8 calibration.
        input_name: ONNX input name, must match :func:`export_onnx` output.

    Returns:
        Mapping ``mode -> tflite_path`` for every requested mode that was
        successfully produced.
    """
    import onnx2tf  # lazy import: heavy dep, only loaded when we actually convert.

    onnx_path = Path(onnx_path)
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx2tf_out = output_dir / "onnx2tf_out"
    onnx2tf_out.mkdir(parents=True, exist_ok=True)

    modes_set = {m.lower() for m in modes}
    unknown = modes_set - set(SUPPORTED_MODES)
    if unknown:
        raise ValueError(
            f"Unknown quantization modes: {sorted(unknown)}. "
            f"Expected subset of {SUPPORTED_MODES}."
        )

    # Build the kwargs onnx2tf honors. It emits fp32 & fp16 by default; we
    # only opt-in to INT8 when calibration data is provided.
    convert_kwargs: dict[str, Any] = {
        "input_onnx_file_path": str(onnx_path),
        "output_folder_path": str(onnx2tf_out),
        "non_verbose": True,
        "copy_onnx_input_output_names_to_tflite": True,
    }

    if "int8" in modes_set:
        if representative_loader is None:
            logger.warning(
                "INT8 mode requested but no representative_loader supplied — "
                "skipping INT8 quantization."
            )
            modes_set.discard("int8")
        else:
            calib = _collect_calibration(
                representative_loader, max_batches=calibration_batches
            )
            calib_path = onnx2tf_out / "calibration.npy"
            np.save(calib_path, calib)
            logger.info(
                "INT8 calibration: %d samples saved to %s", calib.shape[0], calib_path
            )
            convert_kwargs["custom_input_op_name_np_data_path"] = [
                [input_name, str(calib_path), "none", "none"]
            ]
            convert_kwargs["output_integer_quantized_tflite"] = True
            # Prefer per-channel quantization: lower accuracy loss on
            # MobileNetV3 (whose Squeeze-and-Excitation modules have very
            # different per-channel dynamic ranges). Fall back to per-tensor
            # if the runtime onnx2tf rejects per-channel for this graph.
            convert_kwargs["quant_type"] = "per-channel"

    logger.info(
        "Running onnx2tf (modes=%s, onnx=%s, out=%s)",
        sorted(modes_set),
        onnx_path,
        onnx2tf_out,
    )
    try:
        onnx2tf.convert(**convert_kwargs)
    except Exception as exc:  # noqa: BLE001 — log and retry per-tensor
        if convert_kwargs.get("quant_type") == "per-channel":
            logger.warning(
                "onnx2tf per-channel INT8 failed (%s: %s); retrying with per-tensor.",
                type(exc).__name__,
                exc,
            )
            convert_kwargs["quant_type"] = "per-tensor"
            onnx2tf.convert(**convert_kwargs)
        else:
            raise

    return _normalize_tflite_outputs(onnx2tf_out, output_dir, modes_set)


# --------------------------------------------------------------------------- #
#  TFLite runtime helpers
# --------------------------------------------------------------------------- #
def _load_tflite_interpreter(tflite_path: str | Path):
    """Instantiate a TFLite Interpreter, preferring the lightweight runtime."""
    try:
        from ai_edge_litert.interpreter import Interpreter  # type: ignore[import-not-found]

        return Interpreter(model_path=str(tflite_path))
    except ImportError:
        pass
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore[import-not-found]

        return Interpreter(model_path=str(tflite_path))
    except ImportError:
        pass
    import tensorflow as tf  # type: ignore[import-not-found]

    return tf.lite.Interpreter(model_path=str(tflite_path))


def _prepare_input(
    images: torch.Tensor | np.ndarray,
    input_detail: dict[str, Any],
) -> np.ndarray:
    """Convert an NCHW torch / numpy batch to the interpreter's expected input.

    Handles dtype coercion (INT8 quantized models expect scale/zero_point
    preprocessing) and NCHW → NHWC layout.
    """
    if isinstance(images, torch.Tensor):
        array = images.detach().cpu().numpy().astype(np.float32, copy=False)
    else:
        array = np.asarray(images, dtype=np.float32)
    if array.ndim == 4 and array.shape[1] == 3:
        array = np.transpose(array, (0, 2, 3, 1))

    dtype = input_detail["dtype"]
    if dtype in (np.int8, np.uint8):
        quant = input_detail.get("quantization", (0.0, 0))
        scale, zero_point = quant if len(quant) >= 2 else (1.0, 0)
        if scale == 0:
            scale = 1.0
        quantized = np.round(array / scale + zero_point)
        info = np.iinfo(dtype)
        quantized = np.clip(quantized, info.min, info.max)
        return quantized.astype(dtype)
    return array.astype(dtype)


def _dequantize_output(
    raw: np.ndarray, output_detail: dict[str, Any]
) -> np.ndarray:
    """Map INT8 outputs back to float32 using the interpreter's quant params."""
    if raw.dtype in (np.int8, np.uint8):
        quant = output_detail.get("quantization", (0.0, 0))
        scale, zero_point = quant if len(quant) >= 2 else (1.0, 0)
        if scale == 0:
            scale = 1.0
        return (raw.astype(np.float32) - float(zero_point)) * float(scale)
    return raw.astype(np.float32, copy=False)


def _softmax_last_dim(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


# --------------------------------------------------------------------------- #
#  Latency benchmark
# --------------------------------------------------------------------------- #
def benchmark_tflite(
    tflite_path: str | Path,
    *,
    input_size: int = 224,
    num_runs: int = 100,
    num_warmup: int = 10,
    num_threads: int = 1,
) -> dict[str, float]:
    """Run a synthetic single-sample latency benchmark on CPU.

    Uses ``perf_counter`` around ``interpreter.invoke()`` to time the inner
    inference call only (pre/post-processing excluded).

    Returns:
        Dict with ``latency_ms_mean / _p50 / _p95 / num_runs / num_warmup``.
    """
    interpreter = _load_tflite_interpreter(tflite_path)
    try:
        interpreter.set_num_threads(int(num_threads))
    except (AttributeError, ValueError):
        pass
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    input_detail = input_details[0]

    # Resize input to batch=1, (H, W, 3) then re-allocate.
    interpreter.resize_tensor_input(input_detail["index"], [1, input_size, input_size, 3])
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]

    dummy_input = np.zeros((1, 3, input_size, input_size), dtype=np.float32)
    prepared = _prepare_input(dummy_input, input_detail)
    interpreter.set_tensor(input_detail["index"], prepared)

    for _ in range(num_warmup):
        interpreter.invoke()

    latencies_ms: list[float] = []
    for _ in range(num_runs):
        start = time.perf_counter()
        interpreter.invoke()
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    latencies_ms.sort()
    return {
        "latency_ms_mean": float(statistics.fmean(latencies_ms)),
        "latency_ms_p50": float(statistics.median(latencies_ms)),
        "latency_ms_p95": float(latencies_ms[int(0.95 * (len(latencies_ms) - 1))]),
        "num_runs": int(num_runs),
        "num_warmup": int(num_warmup),
    }


# --------------------------------------------------------------------------- #
#  Accuracy evaluation
# --------------------------------------------------------------------------- #
def evaluate_tflite(
    tflite_path: str | Path,
    loader: DataLoader,
    *,
    num_threads: int = 1,
    fake_class_index: int = 1,
) -> BinaryMetrics:
    """Run the TFLite model over ``loader`` and compute AUC / log-loss / acc.

    Args:
        tflite_path: Path to the ``.tflite`` file.
        loader: Yields ``(images, hard_labels, soft_labels, meta)`` tuples.
            Only ``images`` and ``hard_labels`` are consumed here.
        num_threads: CPU threads for the interpreter. ``1`` matches typical
            edge hardware.
        fake_class_index: Column of the softmaxed output to treat as
            ``P(fake)``. Matches the training-time convention.
    """
    interpreter = _load_tflite_interpreter(tflite_path)
    try:
        interpreter.set_num_threads(int(num_threads))
    except (AttributeError, ValueError):
        pass
    interpreter.allocate_tensors()

    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    current_batch_size = None
    for batch in loader:
        images, hard_labels, *_ = batch
        batch_size = images.shape[0]
        if batch_size != current_batch_size:
            interpreter.resize_tensor_input(
                input_detail["index"],
                [batch_size, images.shape[2], images.shape[3], images.shape[1]],
            )
            interpreter.allocate_tensors()
            input_detail = interpreter.get_input_details()[0]
            output_detail = interpreter.get_output_details()[0]
            current_batch_size = batch_size

        prepared = _prepare_input(images, input_detail)
        interpreter.set_tensor(input_detail["index"], prepared)
        interpreter.invoke()
        raw_output = interpreter.get_tensor(output_detail["index"])
        logits = _dequantize_output(raw_output, output_detail)
        probs = _softmax_last_dim(logits)[:, fake_class_index]
        all_probs.append(probs.astype(np.float32))

        labels_np = (
            hard_labels.detach().cpu().numpy()
            if isinstance(hard_labels, torch.Tensor)
            else np.asarray(hard_labels)
        )
        all_labels.append(labels_np.astype(np.int64))

    if not all_probs:
        raise RuntimeError("TFLite eval loader yielded zero batches")

    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    return compute_binary_metrics(y_true, y_prob)


# --------------------------------------------------------------------------- #
#  File-size helpers + top-level driver
# --------------------------------------------------------------------------- #
def file_size_mb(path: str | Path) -> float:
    return float(Path(path).stat().st_size) / (1024.0 * 1024.0)


def run_edge_benchmark(
    model: nn.Module,
    *,
    eval_loader: DataLoader,
    output_dir: str | Path,
    representative_loader: DataLoader | None = None,
    modes: Iterable[str] = SUPPORTED_MODES,
    input_size: int = 224,
    num_runs: int = 100,
    num_warmup: int = 10,
    num_threads: int = 1,
    calibration_batches: int = 16,
    opset: int = 17,
) -> list[EdgeBenchmarkResult]:
    """End-to-end: export ONNX, convert TFLite(s), benchmark + evaluate each.

    Missing quantization outputs (e.g. INT8 when no representative loader is
    supplied) are logged and skipped instead of aborting the benchmark.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = export_onnx(
        model, output_dir / "model.onnx", input_size=input_size, opset=opset
    )

    tflite_dir = output_dir / "tflite"
    tflite_paths = convert_to_tflite(
        onnx_path,
        tflite_dir,
        modes=modes,
        representative_loader=representative_loader,
        calibration_batches=calibration_batches,
    )

    results: list[EdgeBenchmarkResult] = []
    for mode in modes:
        path = tflite_paths.get(mode)
        if path is None:
            logger.warning("Skipping benchmark for mode=%s (no TFLite artefact)", mode)
            continue
        latency = benchmark_tflite(
            path,
            input_size=input_size,
            num_runs=num_runs,
            num_warmup=num_warmup,
            num_threads=num_threads,
        )
        metrics = evaluate_tflite(path, eval_loader, num_threads=num_threads)
        results.append(
            EdgeBenchmarkResult(
                mode=mode,
                tflite_path=path,
                size_mb=file_size_mb(path),
                latency_ms_mean=latency["latency_ms_mean"],
                latency_ms_p50=latency["latency_ms_p50"],
                latency_ms_p95=latency["latency_ms_p95"],
                num_runs=latency["num_runs"],
                num_warmup=latency["num_warmup"],
                auc=metrics.auc,
                log_loss=metrics.log_loss,
                accuracy=metrics.accuracy,
                num_samples=metrics.num_samples,
            )
        )
        logger.info(
            "mode=%s size=%.2fMB lat_mean=%.2fms p95=%.2fms auc=%.4f",
            mode,
            results[-1].size_mb,
            results[-1].latency_ms_mean,
            results[-1].latency_ms_p95,
            results[-1].auc,
        )
    return results
