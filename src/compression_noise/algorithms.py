from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import json

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    x: int
    y: int
    w: int
    h: int
    score: float

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h, "score": self.score}


class VisionAlgorithm(ABC):
    name: str

    @abstractmethod
    def detect(self, frame: np.ndarray, frame_index: int) -> list[Detection]:
        """Return bounding-box detections for one frame."""


class MotionContourDetector(VisionAlgorithm):
    name = "motion"

    def __init__(self, min_area_ratio: float = 0.0008) -> None:
        self.min_area_ratio = min_area_ratio
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=120,
            varThreshold=36,
            detectShadows=True,
        )

    def detect(self, frame: np.ndarray, frame_index: int) -> list[Detection]:
        mask = self.subtractor.apply(frame)
        _, mask = cv2.threshold(mask, 244, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=2)

        frame_area = frame.shape[0] * frame.shape[1]
        min_area = frame_area * self.min_area_ratio
        return _contour_boxes(mask, min_area)


class EdgeContourDetector(VisionAlgorithm):
    name = "edges"

    def __init__(self, min_area_ratio: float = 0.0015) -> None:
        self.min_area_ratio = min_area_ratio

    def detect(self, frame: np.ndarray, frame_index: int) -> list[Detection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 80, 180)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        frame_area = frame.shape[0] * frame.shape[1]
        min_area = frame_area * self.min_area_ratio
        return _contour_boxes(closed, min_area)


def build_algorithm(name: str) -> VisionAlgorithm:
    if name == "motion":
        return MotionContourDetector()
    if name == "edges":
        return EdgeContourDetector()
    raise ValueError(f"Unknown algorithm: {name}")


def run_algorithm(
    video_path: Path,
    output_path: Path,
    algorithm_name: str,
    sample_every: int,
) -> None:
    algorithm = build_algorithm(algorithm_name)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    frames = []
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % sample_every == 0:
            detections = algorithm.detect(frame, frame_index)
            frames.append(
                {
                    "frame_index": frame_index,
                    "time_seconds": frame_index / fps if fps else None,
                    "detections": [detection.to_dict() for detection in detections],
                }
            )
        else:
            algorithm.detect(frame, frame_index)
        frame_index += 1

    capture.release()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "video_path": str(video_path),
        "algorithm": algorithm_name,
        "sample_every": sample_every,
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "frames": frames,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _contour_boxes(mask: np.ndarray, min_area: float) -> list[Detection]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        detections.append(Detection(x=x, y=y, w=w, h=h, score=float(area)))

    detections.sort(key=lambda box: box.score, reverse=True)
    return detections
