from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class CompressionLevel:
    name: str
    crf: int
    scale: float


@dataclass(frozen=True)
class VideoSpec:
    id: str
    category: str
    path: Path


@dataclass(frozen=True)
class ExperimentConfig:
    raw_dir: Path
    compressed_dir: Path
    outputs_dir: Path
    sample_every: int
    algorithms: list[str]
    compression_levels: list[CompressionLevel]
    videos: list[VideoSpec]


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    return ExperimentConfig(
        raw_dir=Path(data.get("raw_dir", "data/raw")),
        compressed_dir=Path(data.get("compressed_dir", "data/compressed")),
        outputs_dir=Path(data.get("outputs_dir", "outputs")),
        sample_every=int(data.get("sample_every", 10)),
        algorithms=list(data.get("algorithms", ["motion"])),
        compression_levels=[
            CompressionLevel(
                name=str(level["name"]),
                crf=int(level["crf"]),
                scale=float(level["scale"]),
            )
            for level in data.get("compression_levels", [])
        ],
        videos=[
            VideoSpec(
                id=str(video["id"]),
                category=str(video.get("category", "uncategorized")),
                path=Path(video["path"]),
            )
            for video in data.get("videos", [])
        ],
    )
