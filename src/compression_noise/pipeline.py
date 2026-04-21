from __future__ import annotations

from pathlib import Path
import argparse
import shutil

from .algorithms import run_algorithm
from .compress import compressed_video_path, compress_video
from .config import ExperimentConfig, load_config
from .evaluation import evaluate_pair, write_summary
from .visualization import plot_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compression-noise vision experiment pipeline")
    parser.add_argument(
        "command",
        choices=["compress", "detect", "evaluate", "plot", "all"],
        help="Pipeline stage to run",
    )
    parser.add_argument("--config", default="project_config.json", help="Path to JSON config")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing artifacts")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.command in {"compress", "all"}:
        compress_all(config, overwrite=args.overwrite)
    if args.command in {"detect", "all"}:
        detect_all(config, overwrite=args.overwrite)
    if args.command in {"evaluate", "all"}:
        evaluate_all(config)
    if args.command in {"plot", "all"}:
        plot_summary(config.outputs_dir / "metrics" / "summary.csv", config.outputs_dir / "plots")


def compress_all(config: ExperimentConfig, overwrite: bool = False) -> None:
    for video in config.videos:
        _require_file(video.path)
        for level in config.compression_levels:
            output_path = compressed_video_path(config.compressed_dir, video, level)
            print(f"compress {video.id} -> {level.name}")
            compress_video(video.path, output_path, level, overwrite=overwrite)


def detect_all(config: ExperimentConfig, overwrite: bool = False) -> None:
    for video in config.videos:
        _require_file(video.path)
        for algorithm in config.algorithms:
            baseline_output = _detection_path(config.outputs_dir, video.id, "original", algorithm)
            if overwrite or not baseline_output.exists():
                print(f"detect {algorithm} baseline {video.id}")
                run_algorithm(video.path, baseline_output, algorithm, config.sample_every)

            for level in config.compression_levels:
                compressed = compressed_video_path(config.compressed_dir, video, level)
                _require_file(compressed)
                output = _detection_path(config.outputs_dir, video.id, level.name, algorithm)
                if overwrite or not output.exists():
                    print(f"detect {algorithm} {video.id} {level.name}")
                    run_algorithm(compressed, output, algorithm, config.sample_every)


def evaluate_all(config: ExperimentConfig) -> None:
    rows = []
    for video in config.videos:
        for algorithm in config.algorithms:
            baseline = _detection_path(config.outputs_dir, video.id, "original", algorithm)
            _require_file(baseline)
            for index, level in enumerate(config.compression_levels):
                candidate = _detection_path(config.outputs_dir, video.id, level.name, algorithm)
                _require_file(candidate)
                metrics = evaluate_pair(baseline, candidate)
                rows.append(
                    {
                        "video_id": video.id,
                        "category": video.category,
                        "algorithm": algorithm,
                        "compression_level": level.name,
                        "compression_order": index,
                        "crf": level.crf,
                        "scale": level.scale,
                        **metrics,
                    }
                )

    write_summary(rows, config.outputs_dir / "metrics" / "summary.csv")


def _detection_path(outputs_dir: Path, video_id: str, level_name: str, algorithm: str) -> Path:
    return outputs_dir / "detections" / video_id / level_name / f"{algorithm}.json"


def _require_file(path: Path) -> None:
    if not path.exists():
        hint = ""
        if path.name == "project_config.json" and Path("project_config.example.json").exists():
            hint = " Copy project_config.example.json to project_config.json or pass --config."
        raise FileNotFoundError(f"Required file not found: {path}.{hint}")


def clean_outputs(config: ExperimentConfig) -> None:
    for directory in [config.compressed_dir, config.outputs_dir]:
        if directory.exists():
            shutil.rmtree(directory)


if __name__ == "__main__":
    main()
