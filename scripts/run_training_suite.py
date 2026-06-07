from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Engine.AudioModel import AudioModel
from Engine.ImageModel import ImageModel
from Engine.LateFusionModel import LateFusionModel
from Engine.MidFusionModel import MidFusionModel
from Engine.ScoreFusionModel import ScoreFusionModel


MODEL_REGISTRY = {
    "ImageModel": ImageModel,
    "AudioModel": AudioModel,
    "LateFusionModel": LateFusionModel,
    "MidFusionModel": MidFusionModel,
    "ScoreFusionModel": ScoreFusionModel,
}
DEFAULT_MODELS = [
    "ImageModel",
    "AudioModel",
    "LateFusionModel",
    "MidFusionModel",
    "ScoreFusionModel",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the project's model training jobs sequentially."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("Data/Pokemon/processed_v2_robust_multimodal"),
        help="Processed dataset root containing train/val/test.",
    )
    parser.add_argument(
        "--result-base",
        type=Path,
        default=Path("artifacts/training_runs"),
        help="Directory where per-model training run folders are created.",
    )
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help=(
            "Comma-separated model names to run. Valid names: "
            + ", ".join(MODEL_REGISTRY)
        ),
    )
    parser.add_argument(
        "--suite-name",
        default="stabilized",
        help="Short label embedded in result folder names and suite summary.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with later models when one training job fails.",
    )
    return parser.parse_args()


def slugify(value: str) -> str:
    cleaned = "".join(character if character.isalnum() else "-" for character in value.strip())
    parts = [part for part in cleaned.split("-") if part]
    return "-".join(parts) or "dataset"


def now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def timestamp_text() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_model_names(value: str) -> list[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [name for name in names if name not in MODEL_REGISTRY]
    if invalid:
        raise ValueError(
            "Unsupported model name(s): "
            + ", ".join(invalid)
            + ". Valid names: "
            + ", ".join(MODEL_REGISTRY)
        )
    return names


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    result_base = args.result_base.expanduser().resolve()
    model_names = parse_model_names(args.models)
    suite_stamp = timestamp_text()
    suite_slug = slugify(args.suite_name)
    dataset_slug = slugify(dataset_root.name)
    result_base.mkdir(parents=True, exist_ok=True)

    suite_summary: dict[str, Any] = {
        "suite_name": args.suite_name,
        "started_at": now_text(),
        "finished_at": None,
        "dataset_root": str(dataset_root),
        "result_base": str(result_base),
        "models": model_names,
        "runs": [],
    }
    suite_summary_path = result_base / f"{suite_stamp}_{suite_slug}_suite_summary.json"
    write_json(suite_summary_path, suite_summary)

    for index, model_name in enumerate(model_names, start=1):
        run_root = (
            result_base
            / f"{suite_stamp}_{index:03d}_{model_name}_{dataset_slug}_{suite_slug}"
        )
        started_at = now_text()
        print(f"[{index}/{len(model_names)}] {model_name} -> {run_root}", flush=True)

        run_record: dict[str, Any] = {
            "model_name": model_name,
            "result_root": str(run_root),
            "started_at": started_at,
            "finished_at": None,
            "status": "running",
            "error": None,
        }
        suite_summary["runs"].append(run_record)
        write_json(suite_summary_path, suite_summary)

        try:
            model = MODEL_REGISTRY[model_name]()
            model.Train(dataset_root, run_root)
        except Exception:
            run_record["status"] = "failed"
            run_record["finished_at"] = now_text()
            run_record["error"] = traceback.format_exc()
            write_json(suite_summary_path, suite_summary)
            print(run_record["error"], file=sys.stderr, flush=True)
            if not args.continue_on_error:
                suite_summary["finished_at"] = now_text()
                suite_summary["status"] = "failed"
                write_json(suite_summary_path, suite_summary)
                return 1
            continue

        run_record["status"] = "completed"
        run_record["finished_at"] = now_text()
        write_json(suite_summary_path, suite_summary)
        print(f"[{index}/{len(model_names)}] {model_name} completed", flush=True)

    suite_summary["finished_at"] = now_text()
    suite_summary["status"] = (
        "completed"
        if all(run["status"] == "completed" for run in suite_summary["runs"])
        else "completed_with_errors"
    )
    write_json(suite_summary_path, suite_summary)
    print(f"Suite summary written to: {suite_summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
