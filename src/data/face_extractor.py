"""Face extraction pipeline: video -> cropped face JPEGs + metadata rows.

Uses RetinaFace (via ``insightface``) as the primary detector and falls
back to MTCNN (via ``facenet-pytorch``) when RetinaFace returns no
bounding box on a frame. Both detectors are lazy-initialised so tests
that mock :class:`FaceExtractor` never import the heavy deps.

Called by: scripts/01_extract_faces.py
Reads: .mp4 video files under ``{drive}/datasets/raw/<dataset>/``
Writes: ``{output_dir}/{video_id}/{frame_idx:04d}.jpg`` (quality=95, 224x224)
Returns per video: list[dict] with keys
    face_path (str), frame_idx (int), video_id (str),
    label (int 0/1), dataset (str), generation (str), technique (str)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
#  Detector protocol + concrete adapters
# --------------------------------------------------------------------------- #


class FaceDetector(Protocol):
    """Minimal interface every detector must implement."""

    def detect(self, image_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Return a list of (x1, y1, x2, y2) boxes in pixel coordinates."""
        ...


class RetinaFaceDetector:
    """InsightFace RetinaFace wrapper. GPU if available, CPU otherwise."""

    def __init__(self, det_size: int = 640) -> None:
        self._det_size = det_size
        self._app: Any | None = None  # lazy

    def _ensure_ready(self) -> None:
        if self._app is not None:
            return
        from insightface.app import FaceAnalysis  # lazy import

        app = FaceAnalysis(allowed_modules=["detection"])
        # ctx_id=0 -> first GPU; InsightFace falls back to CPU if none.
        app.prepare(ctx_id=0, det_size=(self._det_size, self._det_size))
        self._app = app

    def detect(self, image_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        self._ensure_ready()
        assert self._app is not None
        faces = self._app.get(image_bgr)
        boxes: list[tuple[int, int, int, int]] = []
        for f in faces:
            x1, y1, x2, y2 = (int(v) for v in f.bbox.astype(int))
            boxes.append((x1, y1, x2, y2))
        return boxes


class MTCNNDetector:
    """facenet-pytorch MTCNN wrapper used as a fallback."""

    def __init__(self, device: str | None = None) -> None:
        self._device = device
        self._mtcnn: Any | None = None  # lazy

    def _ensure_ready(self) -> None:
        if self._mtcnn is not None:
            return
        import torch
        from facenet_pytorch import MTCNN  # lazy import

        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._mtcnn = MTCNN(keep_all=True, device=device)

    def detect(self, image_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        self._ensure_ready()
        assert self._mtcnn is not None
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        boxes, _ = self._mtcnn.detect(rgb)
        if boxes is None:
            return []
        return [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2 in boxes]


# --------------------------------------------------------------------------- #
#  Extractor
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ExtractorConfig:
    frames_per_video: int = 32
    output_size: int = 224
    margin: float = 0.30
    jpeg_quality: int = 95


class FaceExtractor:
    """End-to-end face extractor: sample frames, detect, crop, save."""

    def __init__(
        self,
        primary: FaceDetector | None = None,
        fallback: FaceDetector | None = None,
        config: ExtractorConfig | None = None,
    ) -> None:
        self.primary: FaceDetector = primary or RetinaFaceDetector()
        self.fallback: FaceDetector = fallback or MTCNNDetector()
        self.cfg: ExtractorConfig = config or ExtractorConfig()

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def extract_from_video(
        self,
        video_path: str | Path,
        output_dir: str | Path,
        *,
        video_id: str,
        label: int,
        dataset: str,
        generation: str,
        technique: str,
    ) -> list[dict[str, Any]]:
        """Extract face crops from one video and return metadata rows."""
        video_path = Path(video_path)
        if not video_path.is_file():
            raise FileNotFoundError(f"Video not found: {video_path}")

        out_dir = Path(output_dir) / video_id
        out_dir.mkdir(parents=True, exist_ok=True)

        frames = list(self._sample_frames(video_path))
        if not frames:
            logger.warning("No frames read from video: %s", video_path)
            return []

        rows: list[dict[str, Any]] = []
        for frame_idx, frame_bgr in frames:
            box = self._detect_largest(frame_bgr)
            if box is None:
                logger.warning(
                    "No face detected: video=%s frame=%d", video_id, frame_idx
                )
                continue

            crop = self._crop_with_margin(frame_bgr, box, self.cfg.margin)
            if crop.size == 0:
                logger.warning(
                    "Empty crop after margin: video=%s frame=%d",
                    video_id,
                    frame_idx,
                )
                continue
            resized = cv2.resize(
                crop,
                (self.cfg.output_size, self.cfg.output_size),
                interpolation=cv2.INTER_AREA,
            )

            out_path = out_dir / f"{frame_idx:04d}.jpg"
            cv2.imwrite(
                str(out_path),
                resized,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(self.cfg.jpeg_quality)],
            )

            rows.append(
                {
                    "face_path": str(out_path),
                    "frame_idx": int(frame_idx),
                    "video_id": video_id,
                    "label": int(label),
                    "dataset": dataset,
                    "generation": generation,
                    "technique": technique,
                }
            )

        logger.info(
            "Extracted %d/%d faces from %s", len(rows), len(frames), video_id
        )
        return rows

    # ------------------------------------------------------------------ #
    #  Internals
    # ------------------------------------------------------------------ #

    def _sample_frames(self, video_path: Path):
        """Yield (frame_idx, frame_bgr) for uniformly-sampled frames."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error("Failed to open video: %s", video_path)
            return
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                return
            n = self.cfg.frames_per_video
            if total <= n:
                indices = list(range(total))
            else:
                step = total / n
                indices = [int(i * step) for i in range(n)]

            wanted = iter(indices)
            next_idx = next(wanted, None)
            while next_idx is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, next_idx)
                ok, frame = cap.read()
                if not ok or frame is None:
                    logger.warning(
                        "Failed to read frame %d from %s", next_idx, video_path
                    )
                    next_idx = next(wanted, None)
                    continue
                yield next_idx, frame
                next_idx = next(wanted, None)
        finally:
            cap.release()

    def _detect_largest(
        self, frame_bgr: np.ndarray
    ) -> tuple[int, int, int, int] | None:
        """Try primary detector, fall back to secondary. Return largest box."""
        boxes = self._safe_detect(self.primary, frame_bgr)
        if not boxes:
            boxes = self._safe_detect(self.fallback, frame_bgr)
        if not boxes:
            return None
        return max(boxes, key=_box_area)

    @staticmethod
    def _safe_detect(
        detector: FaceDetector, frame_bgr: np.ndarray
    ) -> list[tuple[int, int, int, int]]:
        try:
            return detector.detect(frame_bgr)
        except Exception as exc:  # noqa: BLE001 — detectors throw varied exceptions
            logger.warning("Detector failed: %s", exc)
            return []

    @staticmethod
    def _crop_with_margin(
        image: np.ndarray,
        box: tuple[int, int, int, int],
        margin: float,
    ) -> np.ndarray:
        """Expand ``box`` by ``margin`` fraction and crop safely."""
        h, w = image.shape[:2]
        x1, y1, x2, y2 = box
        bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)
        mx, my = int(bw * margin), int(bh * margin)
        x1 = max(x1 - mx, 0)
        y1 = max(y1 - my, 0)
        x2 = min(x2 + mx, w)
        y2 = min(y2 + my, h)
        return image[y1:y2, x1:x2]


def _box_area(box: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)
