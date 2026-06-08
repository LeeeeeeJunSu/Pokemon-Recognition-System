from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a paired multimodal dataset by taking images from one processed "
            "dataset and audio from another processed dataset."
        )
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("Data/Pokemon/processed_v2_robust_multimodal"),
        help="Processed dataset root that provides image.png files.",
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=Path("Data/Pokemon/processed_robust_val30"),
        help="Processed dataset root that provides audio.wav files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("Data/Pokemon/processed_v2_images_val30_audio"),
        help="Output processed dataset root.",
    )
    parser.add_argument(
        "--train-driver",
        choices=("image", "audio"),
        default="image",
        help="Which source controls train sample count.",
    )
    parser.add_argument(
        "--val-driver",
        choices=("image", "audio"),
        default="audio",
        help="Which source controls validation sample count.",
    )
    parser.add_argument(
        "--test-driver",
        choices=("image", "audio"),
        default="audio",
        help="Which source controls test sample count.",
    )
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="Use hardlinks to avoid duplicating bytes, or copy files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output root before writing.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sample_dirs(root: Path, split: str, class_name: str, required_file: str) -> list[Path]:
    class_root = root / split / class_name
    if not class_root.exists():
        return []
    return sorted(
        sample_dir
        for sample_dir in class_root.iterdir()
        if sample_dir.is_dir() and (sample_dir / required_file).exists()
    )


def copy_or_link(src: Path, dst: Path, link_mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if link_mode == "hardlink":
        try:
            os.link(src, dst)
            return "hardlink"
        except OSError:
            pass
    shutil.copy2(src, dst)
    return "copy"


def choose_cycled(paths: list[Path], index: int) -> Path:
    if not paths:
        raise ValueError("Cannot choose from an empty path list.")
    return paths[index % len(paths)]


def stable_digest(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def safe_sample_id(driver: str, index: int, image_dir: Path, audio_dir: Path) -> str:
    digest = stable_digest(f"{driver}:{index}:{image_dir}:{audio_dir}")
    if driver == "image":
        base = f"{image_dir.name}__audio__{audio_dir.name}"
    else:
        base = f"{audio_dir.name}__image__{image_dir.name}"
    return f"{index:05d}_{digest}_{base}"[:180]


def build_meta(
    split: str,
    class_name: str,
    image_dir: Path,
    audio_dir: Path,
    image_root: Path,
    audio_root: Path,
    driver: str,
    image_storage: str,
    audio_storage: str,
) -> dict[str, Any]:
    image_meta = load_json(image_dir / "meta.json")
    audio_meta = load_json(audio_dir / "meta.json")
    merged = dict(image_meta or audio_meta)
    merged["split"] = split
    merged["class_name"] = class_name
    merged["common_name"] = merged.get("common_name", class_name)
    merged["condition"] = audio_meta.get("condition", image_meta.get("condition", split))
    merged["image"] = image_meta.get("image")
    merged["audio"] = audio_meta.get("audio")
    merged["hybrid"] = {
        "driver": driver,
        "image_root": str(image_root.as_posix()),
        "audio_root": str(audio_root.as_posix()),
        "image_sample_id": image_dir.name,
        "audio_sample_id": audio_dir.name,
        "image_storage": image_storage,
        "audio_storage": audio_storage,
        "note": (
            "image.png was sourced from image_root and audio.wav was sourced "
            "from audio_root; counterpart modality is cycled within the same split/class."
        ),
    }
    return merged


def class_names_from_labels(root: Path) -> list[str]:
    labels = load_json(root / "labels.json")
    if not labels:
        raise FileNotFoundError(f"Missing or empty labels.json under: {root}")
    if isinstance(labels, dict) and "classes" in labels:
        classes = labels["classes"]
        if isinstance(classes, list):
            return [str(item.get("class_name") or item.get("name")) for item in classes]
    if isinstance(labels, dict):
        return sorted(str(name) for name in labels)
    raise ValueError(f"Unsupported labels.json format: {root / 'labels.json'}")


def build_hybrid_dataset(args: argparse.Namespace) -> Counter:
    image_root = args.image_root.expanduser().resolve()
    audio_root = args.audio_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if args.overwrite and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        (output_root / split).mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_root / "labels.json", output_root / "labels.json")

    class_names = class_names_from_labels(image_root)
    summary: Counter = Counter()
    summary["classes"] = len(class_names)

    drivers = {
        "train": args.train_driver,
        "val": args.val_driver,
        "test": args.test_driver,
    }

    for split in SPLITS:
        driver = drivers[split]
        for class_name in class_names:
            image_samples = sample_dirs(image_root, split, class_name, "image.png")
            audio_samples = sample_dirs(audio_root, split, class_name, "audio.wav")
            if not image_samples:
                raise ValueError(f"No image samples for {split}/{class_name} in {image_root}")
            if not audio_samples:
                raise ValueError(f"No audio samples for {split}/{class_name} in {audio_root}")

            driver_samples = image_samples if driver == "image" else audio_samples
            summary[f"{split}_image_source_samples"] += len(image_samples)
            summary[f"{split}_audio_source_samples"] += len(audio_samples)
            summary[f"{split}_samples"] += len(driver_samples)
            summary[f"{split}_driver_{driver}"] += len(driver_samples)

            for index, driver_dir in enumerate(driver_samples):
                image_dir = driver_dir if driver == "image" else choose_cycled(image_samples, index)
                audio_dir = driver_dir if driver == "audio" else choose_cycled(audio_samples, index)
                sample_id = safe_sample_id(driver, index, image_dir, audio_dir)
                output_dir = output_root / split / class_name / sample_id
                image_storage = copy_or_link(
                    image_dir / "image.png",
                    output_dir / "image.png",
                    args.link_mode,
                )
                audio_storage = copy_or_link(
                    audio_dir / "audio.wav",
                    output_dir / "audio.wav",
                    args.link_mode,
                )
                write_json(
                    output_dir / "meta.json",
                    build_meta(
                        split=split,
                        class_name=class_name,
                        image_dir=image_dir,
                        audio_dir=audio_dir,
                        image_root=image_root,
                        audio_root=audio_root,
                        driver=driver,
                        image_storage=image_storage,
                        audio_storage=audio_storage,
                    ),
                )

    write_json(output_root / "dataset_summary.json", dict(sorted(summary.items())))
    return summary


def main() -> None:
    args = parse_args()
    summary = build_hybrid_dataset(args)
    print(f"Hybrid dataset written to: {args.output_root}")
    for key in sorted(summary):
        print(f"{key}: {summary[key]}")


if __name__ == "__main__":
    main()
