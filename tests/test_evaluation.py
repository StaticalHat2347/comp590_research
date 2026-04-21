import json

from compression_noise.evaluation import _iou, _match_boxes, evaluate_pair


def test_iou_perfect_match() -> None:
    box = {"x": 10, "y": 20, "w": 30, "h": 40}
    assert _iou(box, box) == 1.0


def test_iou_no_overlap() -> None:
    first = {"x": 0, "y": 0, "w": 10, "h": 10}
    second = {"x": 20, "y": 20, "w": 10, "h": 10}
    assert _iou(first, second) == 0.0


def test_match_boxes_greedy_unique_matches() -> None:
    base = [
        {"x": 0, "y": 0, "w": 10, "h": 10},
        {"x": 100, "y": 100, "w": 10, "h": 10},
    ]
    candidate = [
        {"x": 0, "y": 0, "w": 10, "h": 10},
        {"x": 0, "y": 0, "w": 10, "h": 10},
    ]
    assert len(_match_boxes(base, candidate, threshold=0.5)) == 1


def test_evaluate_pair_normalizes_video_dimensions(tmp_path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(
        json.dumps(
            {
                "width": 100,
                "height": 100,
                "frames": [
                    {
                        "frame_index": 0,
                        "detections": [{"x": 10, "y": 10, "w": 20, "h": 20}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    candidate.write_text(
        json.dumps(
            {
                "width": 50,
                "height": 50,
                "frames": [
                    {
                        "frame_index": 0,
                        "detections": [{"x": 5, "y": 5, "w": 10, "h": 10}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    metrics = evaluate_pair(baseline, candidate)
    assert metrics["score"] == 1.0
