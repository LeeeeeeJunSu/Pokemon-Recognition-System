from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one class-balanced paired dataset for ImageModel, AudioModel, "
            "and fusion models. Every sample contains both image.png and audio.wav."
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
        default=Path("Data/Pokemon/processed_fair_paired"),
        help="Output processed dataset root.",
    )
    parser.add_argument("--train-count", type=int, default=70)
    parser.add_argument("--val-count", type=int, default=30)
    parser.add_argument("--test-count", type=int, default=25)
    parser.add_argument(
        "--link-mode",
        choices=("copy", "hardlink"),
        default="copy",
        help="Use copy for independent files or hardlink to save disk space.",
    )
    parser.add_argument(
        "--strict-unique-images",
        action="store_true",
        help="Fail if a split/class has fewer unique images than the requested count.",
    )
    parser.add_argument(
        "--allow-audio-reuse",
        action="store_true",
        help="Allow audio cycling when audio samples are fewer than the requested count.",
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


def stable_digest(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def safe_sample_component(value: str) -> str:
    safe = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value)
    ).strip("_")
    return safe or "unknown"


def pair_condition(split: str, image_meta: dict[str, Any], audio_meta: dict[str, Any]) -> str:
    return str(audio_meta.get("condition", image_meta.get("condition", split)))


def class_names_from_labels(root: Path) -> list[str]:
    labels = load_json(root / "labels.json")
    if not labels:
        raise FileNotFoundError(f"Missing labels.json under: {root}")
    if isinstance(labels, dict) and "classes" in labels:
        classes = labels["classes"]
        if isinstance(classes, list):
            return [str(item.get("class_name") or item.get("name")) for item in classes]
    if isinstance(labels, dict):
        return sorted(str(name) for name in labels)
    raise ValueError(f"Unsupported labels.json format: {root / 'labels.json'}")


def sample_dirs(root: Path, split: str, class_name: str, required_file: str) -> list[Path]:
    class_root = root / split / class_name
    if not class_root.exists():
        return []
    return sorted(
        sample_dir
        for sample_dir in class_root.iterdir()
        if sample_dir.is_dir() and (sample_dir / required_file).exists()
    )


def choose_cycled(paths: list[Path], index: int) -> Path:
    if not paths:
        raise ValueError("Cannot choose from an empty path list.")
    return paths[index % len(paths)]


def copy_or_link(src: Path, dst: Path, link_mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if link_mode == "hardlink":
        try:
            import os

            os.link(src, dst)
            return "hardlink"
        except OSError:
            pass
    shutil.copy2(src, dst)
    return "copy"


def build_meta(
    split: str,
    class_name: str,
    pair_index: int,
    requested_count: int,
    image_dir: Path,
    audio_dir: Path,
    image_root: Path,
    audio_root: Path,
    image_reused: bool,
    audio_reused: bool,
    image_storage: str,
    audio_storage: str,
    image_meta: dict[str, Any] | None = None,
    audio_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if image_meta is None:
        image_meta = load_json(image_dir / "meta.json")
    if audio_meta is None:
        audio_meta = load_json(audio_dir / "meta.json")
    merged = dict(audio_meta or image_meta)
    merged["split"] = split
    merged["class_name"] = class_name
    merged["common_name"] = merged.get("common_name", class_name)
    merged["condition"] = pair_condition(split, image_meta, audio_meta)
    merged["image"] = image_meta.get("image")
    merged["audio"] = audio_meta.get("audio")
    merged["fair_paired"] = {
        "pair_index": pair_index,
        "requested_count_per_class": requested_count,
        "image_root": str(image_root.as_posix()),
        "audio_root": str(audio_root.as_posix()),
        "image_sample_id": image_dir.name,
        "audio_sample_id": audio_dir.name,
        "image_reused_within_split_class": image_reused,
        "audio_reused_within_split_class": audio_reused,
        "image_storage": image_storage,
        "audio_storage": audio_storage,
        "evaluation_unit": "paired_scenario",
    }
    return merged


def build_dataset(args: argparse.Namespace) -> Counter:
    image_root = args.image_root.expanduser().resolve()
    audio_root = args.audio_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    counts = {
        "train": max(0, int(args.train_count)),
        "val": max(0, int(args.val_count)),
        "test": max(0, int(args.test_count)),
    }
    strict_audio = not args.allow_audio_reuse

    if args.overwrite and output_root.exists():
        shutil.rmtree(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output root is not empty. Use --overwrite: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        (output_root / split).mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_root / "labels.json", output_root / "labels.json")

    class_names = class_names_from_labels(image_root)
    summary: Counter = Counter()
    summary["classes"] = len(class_names)
    summary["evaluation_unit_paired_scenario"] = 1

    for split in SPLITS:
        requested_count = counts[split]
        for class_name in class_names:
            image_samples = sample_dirs(image_root, split, class_name, "image.png")
            audio_samples = sample_dirs(audio_root, split, class_name, "audio.wav")
            if not image_samples:
                raise ValueError(f"No image samples for {split}/{class_name}: {image_root}")
            if not audio_samples:
                raise ValueError(f"No audio samples for {split}/{class_name}: {audio_root}")
            if args.strict_unique_images and len(image_samples) < requested_count:
                raise ValueError(
                    f"{split}/{class_name} has {len(image_samples)} images, "
                    f"but {requested_count} were requested."
                )
            if strict_audio and len(audio_samples) < requested_count:
                raise ValueError(
                    f"{split}/{class_name} has {len(audio_samples)} audio samples, "
                    f"but {requested_count} were requested."
                )

            summary[f"{split}_image_source_samples"] += len(image_samples)
            summary[f"{split}_audio_source_samples"] += len(audio_samples)
            summary[f"{split}_samples"] += requested_count
            summary[f"{split}_requested_count_per_class"] = requested_count

            for index in range(requested_count):
                image_dir = choose_cycled(image_samples, index)
                audio_dir = choose_cycled(audio_samples, index)
                image_meta = load_json(image_dir / "meta.json")
                audio_meta = load_json(audio_dir / "meta.json")
                condition = pair_condition(split, image_meta, audio_meta)
                condition_slug = safe_sample_component(condition)
                image_reused = index >= len(image_samples)
                audio_reused = index >= len(audio_samples)
                summary[f"{split}_image_reused_pairs"] += int(image_reused)
                summary[f"{split}_audio_reused_pairs"] += int(audio_reused)

                digest = stable_digest(
                    f"{split}:{condition}:{class_name}:{index}:{image_dir}:{audio_dir}"
                )
                sample_id = f"{split}_{condition_slug}_{index:03d}_{digest}"
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
                meta = build_meta(
                    split=split,
                    class_name=class_name,
                    pair_index=index,
                    requested_count=requested_count,
                    image_dir=image_dir,
                    audio_dir=audio_dir,
                    image_root=image_root,
                    audio_root=audio_root,
                    image_reused=image_reused,
                    audio_reused=audio_reused,
                    image_storage=image_storage,
                    audio_storage=audio_storage,
                    image_meta=image_meta,
                    audio_meta=audio_meta,
                )
                condition = str(meta.get("condition", condition))
                summary[f"{split}_condition_{condition}"] += 1
                audio_meta = meta.get("audio") or {}
                summary[f"{split}_audio_augmentation_{audio_meta.get('augmentation', 'unknown')}"] += 1
                summary[f"{split}_audio_degradation_{audio_meta.get('degradation', 'unknown')}"] += 1
                write_json(output_dir / "meta.json", meta)

    write_json(output_root / "dataset_summary.json", dict(sorted(summary.items())))
    return summary


def main() -> None:
    args = parse_args()
    summary = build_dataset(args)
    print(f"Fair paired dataset written to: {args.output_root}")
    for key in sorted(summary):
        print(f"{key}: {summary[key]}")


if __name__ == "__main__":
    main()
