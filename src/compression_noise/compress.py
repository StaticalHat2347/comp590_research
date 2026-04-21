from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

from .config import CompressionLevel, VideoSpec


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("FFmpeg was not found. Install it with `brew install ffmpeg`.")


def compressed_video_path(
    compressed_dir: Path,
    video: VideoSpec,
    level: CompressionLevel,
) -> Path:
    return compressed_dir / video.category / video.id / f"{level.name}.mp4"


def compress_video(
    source_path: Path,
    output_path: Path,
    level: CompressionLevel,
    overwrite: bool = False,
) -> None:
    ensure_ffmpeg()
    if output_path.exists() and not overwrite:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-i",
        str(source_path),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(level.crf),
        "-pix_fmt",
        "yuv420p",
    ]
    if level.scale != 1.0:
        command.extend(["-vf", f"scale=trunc(iw*{level.scale}/2)*2:trunc(ih*{level.scale}/2)*2"])
    command.append(str(output_path))

    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg failed for {source_path} at {level.name}:\n{result.stderr.strip()}"
        )
