"""Numerical sanity checks for PyTorch <-> TFLite <-> CoreML conversion.

After conversion, we run the SAME input through each runtime and compare
outputs. Drift larger than the documented thresholds indicates a layer
mismatch — typically caused by HardSwish or Squeeze-Excite quantization
issues on MobileNetV3-Large.

Documented expected drift (architecture: MobileNetV3-Large + 2-class head,
input 224x224 float32 normalized ImageNet):

    runtime       | mode  | max_abs_diff vs PyTorch  | comments
    ------------- | ----- | ------------------------ | ---------------------
    TFLite        | fp32  | < 5e-4                   | float roundoff
    TFLite        | fp16  | < 5e-3                   | half-precision math
    TFLite        | int8  | < 2e-1                   | quantization drift
    CoreML mlprog | fp32  | < 5e-4                   | tracing-only
    CoreML mlprog | int8  | < 2e-1                   | weight quant + fp16 act

If you see drift outside these bands, inspect the conversion logs for
HardSwish / SE-block warnings and fall back to ReLU6 approximation
(see ``EDGE_EVAL_MACBOOK_GUIDE.md``).

Called by:
    notebooks/08_edge_evaluation_macbook.ipynb (Cell 4: pilot conversion)
    scripts/edge/run_edge_eval_macbook.py
Reads:
    Already-converted TFLite / CoreML files + a PyTorch ``nn.Module``.
Writes:
    None — returns a structured ``SanityReport`` for the caller to log.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.utils.logger import get_logger

logger = get_logger(__name__)


# Thresholds per (runtime, mode) — keep in sync with the docstring above.
_DRIFT_THRESHOLDS: dict[tuple[str, str], float] = {
    ("tflite", "fp32"): 5e-4,
    ("tflite", "fp16"): 5e-3,
    ("tflite", "int8"): 2e-1,
    ("coreml", "fp32"): 5e-4,
    ("coreml", "int8"): 2e-1,
}


@dataclass(frozen=True)
class SanityRow:
    runtime: str  # "tflite" | "coreml"
    mode: str     # "fp32" | "fp16" | "int8"
    artifact_path: Path
    max_abs_diff: float
    mean_abs_diff: float
    cosine_similarity: float
    threshold: float
    passed: bool

    def as_dict(self) -> dict[str, float | str | bool]:
        return {
            "runtime": self.runtime,
            "mode": self.mode,
            "artifact_path": str(self.artifact_path),
            "max_abs_diff": float(self.max_abs_diff),
            "mean_abs_diff": float(self.mean_abs_diff),
            "cosine_similarity": float(self.cosine_similarity),
            "threshold": float(self.threshold),
            "passed": bool(self.passed),
        }


@dataclass
class SanityReport:
    rows: list[SanityRow] = field(default_factory=list)

    def all_passed(self) -> bool:
        return all(r.passed for r in self.rows)

    def summary(self) -> str:
        if not self.rows:
            return "(no sanity rows)"
        lines: list[str] = []
        for r in self.rows:
            mark = "[PASS]" if r.passed else "[FAIL]"
            lines.append(
                f"  {mark} {r.runtime:>7s} {r.mode:>5s}  "
                f"max_abs={r.max_abs_diff:.2e}  "
                f"mean_abs={r.mean_abs_diff:.2e}  "
                f"cos={r.cosine_similarity:.4f}  "
                f"(threshold {r.threshold:.2e})"
            )
        return "\n".join(lines)

    def as_dict(self) -> list[dict[str, float | str | bool]]:
        return [r.as_dict() for r in self.rows]


def _make_sample_inputs(
    *,
    batch_size: int = 4,
    input_size: int = 224,
    seed: int = 0,
) -> np.ndarray:
    """Generate a small deterministic batch of NCHW inputs.

    Uses a fixed seed so the same input is fed to all runtimes —
    differences are then purely due to conversion drift.
    """
    rng = np.random.default_rng(seed)
    # Range matches typical ImageNet-normalized inputs:
    # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    # → roughly (-2.5, 2.5).
    return rng.standard_normal(
        (batch_size, 3, input_size, input_size)
    ).astype(np.float32)


def _pytorch_forward(
    model: nn.Module, sample_nchw: np.ndarray
) -> np.ndarray:
    """Run PyTorch forward pass and return logits as numpy (N, C)."""
    model = model.eval().cpu()
    with torch.inference_mode():
        x = torch.from_numpy(sample_nchw).float()
        logits = model(x)
    return logits.detach().cpu().numpy().astype(np.float32)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Per-sample cosine similarity averaged across the batch."""
    a_flat = a.reshape(a.shape[0], -1)
    b_flat = b.reshape(b.shape[0], -1)
    sims: list[float] = []
    for i in range(a_flat.shape[0]):
        denom = float(np.linalg.norm(a_flat[i]) * np.linalg.norm(b_flat[i]))
        if denom == 0.0:
            sims.append(0.0)
            continue
        sims.append(float(np.dot(a_flat[i], b_flat[i]) / denom))
    return float(np.mean(sims))


def _tflite_forward(
    tflite_path: str | Path, sample_nchw: np.ndarray
) -> np.ndarray:
    """Run TFLite forward pass and return logits as numpy.

    Mirrors the preprocessing logic in ``edge_eval.evaluate_tflite``:
    NCHW -> NHWC, optional int8 quantize/dequantize.
    """
    from src.evaluation.edge_eval import (  # local import to avoid cycle at module load
        _dequantize_output,
        _load_tflite_interpreter,
        _prepare_input,
    )

    interpreter = _load_tflite_interpreter(tflite_path)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    n, _, h, w = sample_nchw.shape
    interpreter.resize_tensor_input(input_detail["index"], [n, h, w, 3])
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    prepared = _prepare_input(sample_nchw, input_detail)
    interpreter.set_tensor(input_detail["index"], prepared)
    interpreter.invoke()
    raw = interpreter.get_tensor(output_detail["index"])
    return _dequantize_output(raw, output_detail).astype(np.float32)


def _coreml_forward(
    mlpackage_path: str | Path,
    sample_nchw: np.ndarray,
    compute_unit: str = "cpu_only",
) -> np.ndarray:
    """Run CoreML forward pass and return logits.

    We use ``compute_unit="cpu_only"`` for the sanity check because the
    ANE pipeline can sometimes have subtle rounding differences vs the
    canonical reference; CPU path matches PyTorch numerics best.
    """
    from src.evaluation.edge_eval_coreml import (  # local import for lazy macOS load
        _load_coreml_model,
        _predict_coreml,
    )

    model = _load_coreml_model(mlpackage_path, compute_unit=compute_unit)
    logits_chunks: list[np.ndarray] = []
    for i in range(sample_nchw.shape[0]):
        logits = _predict_coreml(model, sample_nchw[i : i + 1])
        logits_chunks.append(logits.reshape(1, -1))
    return np.concatenate(logits_chunks, axis=0).astype(np.float32)


def verify_tflite(
    model: nn.Module,
    tflite_paths: dict[str, str | Path],
    *,
    sample_inputs: np.ndarray | None = None,
    batch_size: int = 4,
    input_size: int = 224,
) -> SanityReport:
    """Compare PyTorch vs each TFLite mode on the same sample inputs.

    Args:
        model: The PyTorch student (already loaded).
        tflite_paths: ``{"fp32": Path(...), "int8": Path(...), ...}``.
        sample_inputs: Optional pre-generated NCHW batch; if None, a
            deterministic random batch is generated.
        batch_size, input_size: Used only when ``sample_inputs is None``.

    Returns:
        ``SanityReport`` with one row per mode.
    """
    if sample_inputs is None:
        sample_inputs = _make_sample_inputs(
            batch_size=batch_size, input_size=input_size
        )

    reference = _pytorch_forward(model, sample_inputs)
    report = SanityReport()
    for mode, path in tflite_paths.items():
        path = Path(path)
        threshold = _DRIFT_THRESHOLDS.get(("tflite", mode), 1e-2)
        try:
            tflite_out = _tflite_forward(path, sample_inputs)
            diff = np.abs(reference - tflite_out)
            row = SanityRow(
                runtime="tflite",
                mode=mode,
                artifact_path=path,
                max_abs_diff=float(diff.max()),
                mean_abs_diff=float(diff.mean()),
                cosine_similarity=_cosine_similarity(reference, tflite_out),
                threshold=threshold,
                passed=bool(diff.max() <= threshold),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "TFLite sanity check failed for mode=%s: %s: %s",
                mode,
                type(exc).__name__,
                exc,
            )
            row = SanityRow(
                runtime="tflite",
                mode=mode,
                artifact_path=path,
                max_abs_diff=float("nan"),
                mean_abs_diff=float("nan"),
                cosine_similarity=float("nan"),
                threshold=threshold,
                passed=False,
            )
        report.rows.append(row)

    return report


def verify_coreml(
    model: nn.Module,
    coreml_paths: dict[str, str | Path],
    *,
    sample_inputs: np.ndarray | None = None,
    batch_size: int = 4,
    input_size: int = 224,
    compute_unit: str = "cpu_only",
) -> SanityReport:
    """Compare PyTorch vs each CoreML mode."""
    if sample_inputs is None:
        sample_inputs = _make_sample_inputs(
            batch_size=batch_size, input_size=input_size
        )

    reference = _pytorch_forward(model, sample_inputs)
    report = SanityReport()
    for mode, path in coreml_paths.items():
        path = Path(path)
        threshold = _DRIFT_THRESHOLDS.get(("coreml", mode), 1e-2)
        try:
            coreml_out = _coreml_forward(
                path, sample_inputs, compute_unit=compute_unit
            )
            diff = np.abs(reference - coreml_out)
            row = SanityRow(
                runtime="coreml",
                mode=mode,
                artifact_path=path,
                max_abs_diff=float(diff.max()),
                mean_abs_diff=float(diff.mean()),
                cosine_similarity=_cosine_similarity(reference, coreml_out),
                threshold=threshold,
                passed=bool(diff.max() <= threshold),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "CoreML sanity check failed for mode=%s: %s: %s",
                mode,
                type(exc).__name__,
                exc,
            )
            row = SanityRow(
                runtime="coreml",
                mode=mode,
                artifact_path=path,
                max_abs_diff=float("nan"),
                mean_abs_diff=float("nan"),
                cosine_similarity=float("nan"),
                threshold=threshold,
                passed=False,
            )
        report.rows.append(row)
    return report
