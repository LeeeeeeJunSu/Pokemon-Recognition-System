from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize audio prediction accuracy by robustness condition "
            "(SNR, degradation, background track)."
        )
    )
    parser.add_argument(
        "path",
        type=Path,
        help=(
            "Path to a predictions JSON file, or a run directory containing "
            "predictions/test_predictions.json or predictions/inference_predictions.json."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output JSON path. Defaults to audio_condition_summary.json next to predictions.",
    )
    return parser.parse_args()


def resolve_predictions_path(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    candidates = [
        path / "predictions" / "test_predictions.json",
        path / "predictions" / "inference_predictions.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find predictions JSON under: {path}")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_sample_meta(prediction: dict[str, Any]) -> dict[str, Any]:
    sample_path = prediction.get("audio_path") or prediction.get("image_path")
    if not sample_path:
        return {}
    meta_path = Path(sample_path).expanduser().resolve().parent / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        payload = read_json(meta_path)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def is_correct(prediction: dict[str, Any]) -> bool | None:
    ground_truth = prediction.get("ground_truth_index")
    predicted = prediction.get("predicted_index")
    if ground_truth is None or predicted is None:
        return None
    return int(ground_truth) == int(predicted)


def init_bucket() -> dict[str, Any]:
    return {
        "samples": 0,
        "correct": 0,
        "accuracy": None,
        "mean_confidence": None,
        "_confidence_sum": 0.0,
        "_confidence_count": 0,
    }


def update_bucket(bucket: dict[str, Any], prediction: dict[str, Any], correct: bool | None) -> None:
    bucket["samples"] += 1
    if correct is True:
        bucket["correct"] += 1

    confidence = prediction.get("confidence")
    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError):
        return
    bucket["_confidence_sum"] += confidence_value
    bucket["_confidence_count"] += 1


def finalize_buckets(buckets: dict[str, dict[str, Any]], has_ground_truth: bool) -> dict[str, dict[str, Any]]:
    finalized: dict[str, dict[str, Any]] = {}
    for key, bucket in sorted(buckets.items()):
        samples = int(bucket["samples"])
        confidence_count = int(bucket["_confidence_count"])
        output = {
            "samples": samples,
            "correct": int(bucket["correct"]) if has_ground_truth else None,
            "accuracy": (float(bucket["correct"]) / samples) if has_ground_truth and samples else None,
            "mean_confidence": (
                float(bucket["_confidence_sum"]) / confidence_count
                if confidence_count
                else None
            ),
        }
        finalized[key] = output
    return finalized


def background_name(audio_meta: dict[str, Any]) -> str:
    background_path = audio_meta.get("background_path")
    if not background_path:
        return "none"
    return Path(str(background_path)).name


def summarize(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, dict[str, dict[str, Any]]] = {
        "by_split": defaultdict(init_bucket),
        "by_condition": defaultdict(init_bucket),
        "by_image_augmentation": defaultdict(init_bucket),
        "by_augmentation": defaultdict(init_bucket),
        "by_degradation": defaultdict(init_bucket),
        "by_snr_db": defaultdict(init_bucket),
        "by_background": defaultdict(init_bucket),
        "by_snr_and_degradation": defaultdict(init_bucket),
        "by_condition_and_snr": defaultdict(init_bucket),
    }
    overall = init_bucket()
    has_ground_truth = False

    for prediction in predictions:
        if not isinstance(prediction, dict):
            continue

        meta = load_sample_meta(prediction)
        audio_meta = meta.get("audio") if isinstance(meta.get("audio"), dict) else {}
        image_meta = meta.get("image") if isinstance(meta.get("image"), dict) else {}
        split = str(meta.get("split", "unknown"))
        condition = str(meta.get("condition", audio_meta.get("condition", "unknown")))
        image_augmentation = str(image_meta.get("augmentation", "none"))
        augmentation = str(audio_meta.get("augmentation", "unknown"))
        degradation = str(audio_meta.get("degradation", "none"))
        snr = audio_meta.get("snr_db")
        snr_key = "clean" if snr is None else f"{float(snr):g}dB"
        background = background_name(audio_meta)
        condition_key = f"snr={snr_key}|degradation={degradation}"
        condition_snr_key = f"condition={condition}|snr={snr_key}"

        correct = is_correct(prediction)
        has_ground_truth = has_ground_truth or correct is not None
        update_bucket(overall, prediction, correct)
        update_bucket(groups["by_split"][split], prediction, correct)
        update_bucket(groups["by_condition"][condition], prediction, correct)
        update_bucket(groups["by_image_augmentation"][image_augmentation], prediction, correct)
        update_bucket(groups["by_augmentation"][augmentation], prediction, correct)
        update_bucket(groups["by_degradation"][degradation], prediction, correct)
        update_bucket(groups["by_snr_db"][snr_key], prediction, correct)
        update_bucket(groups["by_background"][background], prediction, correct)
        update_bucket(groups["by_snr_and_degradation"][condition_key], prediction, correct)
        update_bucket(groups["by_condition_and_snr"][condition_snr_key], prediction, correct)

    return {
        "overall": finalize_buckets({"overall": overall}, has_ground_truth)["overall"],
        **{
            group_name: finalize_buckets(buckets, has_ground_truth)
            for group_name, buckets in groups.items()
        },
    }


def main() -> None:
    args = parse_args()
    predictions_path = resolve_predictions_path(args.path)
    payload = read_json(predictions_path)
    if not isinstance(payload, list):
        raise ValueError(f"Predictions JSON must contain a list: {predictions_path}")

    summary = summarize(payload)
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else predictions_path.with_name("audio_condition_summary.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved audio condition summary: {output_path}")


if __name__ == "__main__":
    main()
