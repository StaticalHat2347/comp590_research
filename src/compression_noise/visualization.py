from __future__ import annotations

from pathlib import Path
import csv
from collections import defaultdict

import matplotlib.pyplot as plt


def plot_summary(summary_csv: Path, plots_dir: Path) -> None:
    data = _read_summary(summary_csv)
    if not data:
        return

    plots_dir.mkdir(parents=True, exist_ok=True)
    by_algorithm = defaultdict(list)
    for row in data:
        by_algorithm[row["algorithm"]].append(row)

    for algorithm, alg_data in by_algorithm.items():
        fig, axis = plt.subplots(figsize=(9, 5))
        by_video = defaultdict(list)
        for row in alg_data:
            by_video[row["video_id"]].append(row)
        for video_id, video_data in by_video.items():
            ordered = sorted(video_data, key=lambda row: row["compression_order"])
            axis.plot(
                [row["compression_order"] for row in ordered],
                [row["score"] for row in ordered],
                marker="o",
                label=video_id,
            )
        axis.set_title(f"Compression Robustness: {algorithm}")
        axis.set_xlabel("Compression level, low to high")
        axis.set_ylabel("Normalized performance score")
        axis.set_ylim(0, 1.05)
        axis.grid(True, alpha=0.3)
        axis.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / f"{algorithm}_score_curve.png", dpi=180)
        plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5))
    score_groups = defaultdict(list)
    for row in data:
        score_groups[(row["algorithm"], row["compression_order"])].append(row["score"])
    averaged = defaultdict(list)
    for (algorithm, compression_order), scores in score_groups.items():
        averaged[algorithm].append(
            {
                "compression_order": compression_order,
                "score": sum(scores) / len(scores),
            }
        )
    for algorithm, alg_data in averaged.items():
        ordered = sorted(alg_data, key=lambda row: row["compression_order"])
        axis.plot(
            [row["compression_order"] for row in ordered],
            [row["score"] for row in ordered],
            marker="o",
            label=algorithm,
        )
    axis.set_title("Average Algorithm Robustness")
    axis.set_xlabel("Compression level, low to high")
    axis.set_ylabel("Mean normalized performance score")
    axis.set_ylim(0, 1.05)
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "algorithm_comparison.png", dpi=180)
    plt.close(fig)


def _read_summary(summary_csv: Path) -> list[dict]:
    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            parsed = dict(row)
            parsed["compression_order"] = int(parsed["compression_order"])
            parsed["score"] = float(parsed["score"])
            rows.append(parsed)
    return rows
