from __future__ import annotations

from pathlib import Path
import csv
import json


def evaluate_pair(
    baseline_path: Path,
    candidate_path: Path,
    iou_threshold: float = 0.5,
) -> dict:
    baseline = _load_normalized_frames(baseline_path)
    candidate = _load_normalized_frames(candidate_path)
    frame_ids = sorted(set(baseline) | set(candidate))

    total_tp = 0
    total_fp = 0
    total_fn = 0
    ious = []

    for frame_id in frame_ids:
        base_boxes = baseline.get(frame_id, [])
        cand_boxes = candidate.get(frame_id, [])
        matches = _match_boxes(base_boxes, cand_boxes, iou_threshold)
        matched_base = {base_idx for base_idx, _, _ in matches}
        matched_cand = {cand_idx for _, cand_idx, _ in matches}

        total_tp += len(matches)
        total_fn += len(base_boxes) - len(matched_base)
        total_fp += len(cand_boxes) - len(matched_cand)
        ious.extend(iou for _, _, iou in matches)

    if total_tp == total_fp == total_fn == 0:
        precision = 1.0
        recall = 1.0
        f1 = 1.0
    else:
        precision = _safe_div(total_tp, total_tp + total_fp)
        recall = _safe_div(total_tp, total_tp + total_fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
    mean_iou = sum(ious) / len(ious) if ious else (1.0 if total_fp == total_fn == 0 else 0.0)
    score = f1 * mean_iou

    return {
        "score": score,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": mean_iou,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "matched_boxes": len(ious),
        "frame_count": len(frame_ids),
    }


def write_summary(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_normalized_frames(path: Path) -> dict[int, list[dict]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    width = float(payload.get("width") or 1.0)
    height = float(payload.get("height") or 1.0)
    return {
        int(frame["frame_index"]): [
            {
                "x": detection["x"] / width,
                "y": detection["y"] / height,
                "w": detection["w"] / width,
                "h": detection["h"] / height,
                "score": detection.get("score", 0.0),
            }
            for detection in frame.get("detections", [])
        ]
        for frame in payload.get("frames", [])
    }


def _match_boxes(base_boxes: list[dict], cand_boxes: list[dict], threshold: float) -> list[tuple[int, int, float]]:
    candidates = []
    for base_idx, base_box in enumerate(base_boxes):
        for cand_idx, cand_box in enumerate(cand_boxes):
            iou = _iou(base_box, cand_box)
            if iou >= threshold:
                candidates.append((base_idx, cand_idx, iou))

    candidates.sort(key=lambda item: item[2], reverse=True)
    used_base = set()
    used_cand = set()
    matches = []
    for base_idx, cand_idx, iou in candidates:
        if base_idx in used_base or cand_idx in used_cand:
            continue
        used_base.add(base_idx)
        used_cand.add(cand_idx)
        matches.append((base_idx, cand_idx, iou))
    return matches


def _iou(first: dict, second: dict) -> float:
    ax1, ay1 = first["x"], first["y"]
    ax2, ay2 = first["x"] + first["w"], first["y"] + first["h"]
    bx1, by1 = second["x"], second["y"]
    bx2, by2 = second["x"] + second["w"], second["y"] + second["h"]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0
