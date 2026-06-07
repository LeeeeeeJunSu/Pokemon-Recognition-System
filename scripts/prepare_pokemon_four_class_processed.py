from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

from prepare_pokemon_processed import (
    DEFAULT_SAMPLE_RATE,
    build_augmented_records,
    collect_audio_paths,
    collect_image_paths,
    create_background_audio_variant,
    create_sample,
    ensure_clean_directory,
    prepare_audio_cache,
    read_pokemon_map,
    resolve_audio_background_root,
    resolve_second_test_image_root,
    safe_stem,
    select_augmented_audio_variant,
    stable_digest,
    write_wav,
    write_labels_json,
)


TARGET_CLASSES = ("Dragonair", "Dratini", "Cubone", "Marowak")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a four-class Pokemon image/audio dataset and duplicate validation "
            "samples into test for trainer compatibility."
        )
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("Data/Pokemon/1차 공유본"),
    )
    parser.add_argument(
        "--second-train-root",
        type=Path,
        default=Path("Data/Pokemon/2차 공유본 Train"),
    )
    parser.add_argument(
        "--second-test-root",
        type=Path,
        default=Path("Data/Pokemon/2차 공유본 Test"),
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("Data/Pokemon/processed_4class"),
    )
    parser.add_argument(
        "--target-first-train-count",
        type=int,
        default=200,
        help="First shared train images are augmented to at least this count per class.",
    )
    parser.add_argument(
        "--audio-augment-copies",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--audio-background-root",
        type=Path,
        default=Path("Data/Pokemon/BGM"),
    )
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--audio-mix-offset-step", type=float, default=3.0)
    parser.add_argument("--audio-mix-snr-db", type=float, default=16.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--disable-camera-augmentation",
        action="store_true",
    )
    return parser.parse_args()


def split_half_train_val(
    paths: list[Path],
    rng: random.Random,
) -> tuple[list[Path], list[Path]]:
    shuffled = list(paths)
    rng.shuffle(shuffled)
    train_count = len(shuffled) // 2
    return sorted(shuffled[:train_count]), sorted(shuffled[train_count:])


def resolve_target_classes(raw_root: Path) -> list[tuple[int, str]]:
    pokemon_map = read_pokemon_map(raw_root)
    ids_by_name = {name.lower(): pokemon_id for pokemon_id, name in pokemon_map.items()}
    missing = [name for name in TARGET_CLASSES if name.lower() not in ids_by_name]
    if missing:
        raise ValueError(f"Pokemon IDs not found for: {', '.join(missing)}")
    return [(ids_by_name[name.lower()], name) for name in TARGET_CLASSES]


def validate_inputs(
    raw_root: Path,
    second_train_root: Path,
    second_test_root: Path,
    background_root: Path | None,
) -> None:
    required = (
        raw_root / "Image" / "train",
        raw_root / "cry",
        raw_root / "pokemon_dict.json",
        second_train_root,
        second_test_root,
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required dataset path not found: {path}")
    if background_root is None or not background_root.exists():
        raise FileNotFoundError(f"Audio background root not found: {background_root}")


def create_synthetic_backgrounds(
    cache_root: Path,
    sample_rate: int,
    seed: int,
) -> list[Path]:
    cache_root.mkdir(parents=True, exist_ok=True)
    duration_seconds = 30
    sample_count = sample_rate * duration_seconds
    time = np.arange(sample_count, dtype=np.float32) / sample_rate
    paths: list[Path] = []

    for index in range(4):
        generator = np.random.default_rng(seed + index)
        white_noise = generator.normal(0.0, 0.018, sample_count).astype(np.float32)
        low_noise = np.convolve(
            white_noise,
            np.ones(320, dtype=np.float32) / 320.0,
            mode="same",
        )
        hum_frequency = 50.0 if index % 2 == 0 else 60.0
        hum = (
            0.008 * np.sin(2.0 * np.pi * hum_frequency * time + index)
            + 0.004 * np.sin(2.0 * np.pi * hum_frequency * 2.0 * time)
        ).astype(np.float32)
        ambience = white_noise * 0.35 + low_noise * 2.5 + hum
        path = cache_root / f"synthetic_background_{index:02d}.wav"
        write_wav(path, ambience, sample_rate)
        paths.append(path)

    return paths


def build_dataset(args: argparse.Namespace) -> Counter:
    raw_root = args.raw_root.expanduser().resolve()
    second_train_root = args.second_train_root.expanduser().resolve()
    second_test_root = resolve_second_test_image_root(
        args.second_test_root.expanduser().resolve()
    )
    processed_root = args.processed_root.expanduser().resolve()
    background_root = resolve_audio_background_root(
        raw_root,
        args.audio_background_root,
    )
    validate_inputs(raw_root, second_train_root, second_test_root, background_root)

    classes = resolve_target_classes(raw_root)
    rng = random.Random(args.seed)
    ensure_clean_directory(processed_root)
    for split in ("train", "val"):
        (processed_root / split).mkdir(parents=True, exist_ok=True)

    background_paths = collect_audio_paths(background_root)
    synthetic_backgrounds = False
    if not background_paths:
        background_paths = create_synthetic_backgrounds(
            processed_root / "_synthetic_backgrounds",
            args.sample_rate,
            args.seed,
        )
        synthetic_backgrounds = True

    label_indices = write_labels_json(processed_root, classes)
    train_audio_cache = prepare_audio_cache(
        raw_root=raw_root,
        processed_root=processed_root,
        pokemon_ids=[pokemon_id for pokemon_id, _ in classes],
        sample_rate=args.sample_rate,
        audio_augment_copies=max(0, args.audio_augment_copies),
        background_paths=[],
        audio_mix_offset_step=args.audio_mix_offset_step,
        audio_mix_snr_db=args.audio_mix_snr_db,
        rng=rng,
    )
    val_audio_cache_root = processed_root / "_audio_cache_bgm"
    val_background_cache = {}

    summary: Counter = Counter(
        {
            "classes": len(classes),
            "audio_background_files": len(background_paths),
            "audio_background_synthetic_fallback": int(synthetic_backgrounds),
        }
    )
    camera_augmentation = not args.disable_camera_augmentation
    train_image_augmentation = "camera" if camera_augmentation else "light"

    for pokemon_id, class_name in classes:
        label_index = label_indices[pokemon_id]
        first_train_paths = collect_image_paths(
            raw_root / "Image" / "train",
            class_name,
        )
        second_train_paths = collect_image_paths(second_train_root, class_name)
        second_test_paths = collect_image_paths(second_test_root, class_name)
        second_test_train_paths, val_paths = split_half_train_val(
            second_test_paths,
            rng,
        )
        train_audio_variants = train_audio_cache.get(pokemon_id, [])

        if not first_train_paths:
            raise ValueError(f"No first shared train images found for {class_name}")
        if not second_train_paths:
            raise ValueError(f"No second shared train images found for {class_name}")
        if not second_test_paths:
            raise ValueError(f"No second shared test images found for {class_name}")
        if not train_audio_variants:
            raise ValueError(f"No train audio variants created for {class_name}")

        summary["first_shared_train_images"] += len(first_train_paths)
        summary["second_shared_train_images"] += len(second_train_paths)
        summary["second_shared_test_images"] += len(second_test_paths)
        summary["second_shared_test_train_split_images"] += len(
            second_test_train_paths
        )
        summary["second_shared_test_val_split_images"] += len(val_paths)

        first_train_records = build_augmented_records(
            first_train_paths,
            max(0, args.target_first_train_count),
            train_image_augmentation,
            rng,
        )
        for index, (image_path, image_augmentation) in enumerate(
            first_train_records
        ):
            sample_id = (
                f"{pokemon_id:03d}_first_{safe_stem(image_path.stem)}_{index:04d}_"
                f"{stable_digest(f'four:train:first:{pokemon_id}:{image_path}:{index}')}"
            )
            audio_variant = select_augmented_audio_variant(
                train_audio_variants,
                index,
            )
            create_sample(
                processed_root=processed_root,
                split="train",
                class_name=class_name,
                label_index=label_index,
                pokemon_id=pokemon_id,
                image_path=image_path,
                audio_variant=audio_variant,
                sample_id=sample_id,
                rng=rng,
                image_augmentation=image_augmentation,
                camera_augmentation=camera_augmentation,
                source_dataset="1차 공유본",
                source_split="Image/train",
                processing_note=(
                    "train image uses the existing deterministic augmentation; "
                    "train audio uses an augmented first-shared cry"
                ),
            )
            summary["train_samples"] += 1
            summary["train_first_shared_augmented_samples"] += 1

        for index, image_path in enumerate(second_train_paths):
            sample_id = (
                f"{pokemon_id:03d}_second_train_{safe_stem(image_path.stem)}_"
                f"{index:05d}_{stable_digest(f'four:train:second:{pokemon_id}:{image_path}:{index}')}"
            )
            audio_variant = select_augmented_audio_variant(
                train_audio_variants,
                index,
            )
            create_sample(
                processed_root=processed_root,
                split="train",
                class_name=class_name,
                label_index=label_index,
                pokemon_id=pokemon_id,
                image_path=image_path,
                audio_variant=audio_variant,
                sample_id=sample_id,
                rng=rng,
                image_augmentation="origin",
                camera_augmentation=False,
                source_dataset="2차 공유본 Train",
                source_split=class_name.lower(),
                processing_note=(
                    "train image is copied/resized without augmentation; "
                    "train audio uses an augmented first-shared cry"
                ),
            )
            summary["train_samples"] += 1
            summary["train_second_shared_original_samples"] += 1

        audio_offset = len(first_train_records) + len(second_train_paths)
        for index, image_path in enumerate(second_test_train_paths):
            sample_id = (
                f"{pokemon_id:03d}_second_test_train_{safe_stem(image_path.stem)}_"
                f"{index:05d}_{stable_digest(f'four:train:second-test:{pokemon_id}:{image_path}:{index}')}"
            )
            audio_variant = select_augmented_audio_variant(
                train_audio_variants,
                audio_offset + index,
            )
            create_sample(
                processed_root=processed_root,
                split="train",
                class_name=class_name,
                label_index=label_index,
                pokemon_id=pokemon_id,
                image_path=image_path,
                audio_variant=audio_variant,
                sample_id=sample_id,
                rng=rng,
                image_augmentation=train_image_augmentation,
                camera_augmentation=camera_augmentation,
                source_dataset="2차 공유본 Test",
                source_split=class_name.lower(),
                processing_note=(
                    "one half of second-shared test is image-augmented for train; "
                    "train audio uses an augmented first-shared cry"
                ),
            )
            summary["train_samples"] += 1
            summary["train_second_shared_test_augmented_samples"] += 1

        for index, image_path in enumerate(val_paths):
            sample_id = (
                f"{pokemon_id:03d}_second_test_val_{safe_stem(image_path.stem)}_"
                f"{index:05d}_{stable_digest(f'four:val:second-test:{pokemon_id}:{image_path}:{index}')}"
            )
            audio_variant = create_background_audio_variant(
                raw_root=raw_root,
                cache_root=val_audio_cache_root,
                pokemon_id=pokemon_id,
                sample_rate=args.sample_rate,
                background_paths=background_paths,
                background_cache=val_background_cache,
                audio_mix_offset_step=args.audio_mix_offset_step,
                audio_mix_snr_db=args.audio_mix_snr_db,
                rng=rng,
                variant_id=f"val_{sample_id}",
            )
            create_sample(
                processed_root=processed_root,
                split="val",
                class_name=class_name,
                label_index=label_index,
                pokemon_id=pokemon_id,
                image_path=image_path,
                audio_variant=audio_variant,
                sample_id=sample_id,
                rng=rng,
                image_augmentation="origin",
                camera_augmentation=False,
                source_dataset="2차 공유본 Test",
                source_split=class_name.lower(),
                processing_note=(
                    "the other half of second-shared test is unaugmented validation image; "
                    "validation audio uses first-shared cry mixed with background audio"
                ),
            )
            summary["val_samples"] += 1
            summary["val_second_shared_test_original_samples"] += 1
            summary["val_audio_background_mix_samples"] += 1

    test_root = processed_root / "test"
    shutil.copytree(processed_root / "val", test_root)
    summary["test_samples"] = summary["val_samples"]
    summary["test_copied_from_val_samples"] = summary["val_samples"]

    with (processed_root / "dataset_summary.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(dict(sorted(summary.items())), handle, ensure_ascii=False, indent=2)

    for cache_name in (
        "_audio_cache",
        "_audio_cache_bgm",
        "_synthetic_backgrounds",
    ):
        cache_root = processed_root / cache_name
        if cache_root.exists():
            shutil.rmtree(cache_root)

    return summary


def main() -> None:
    args = parse_args()
    summary = build_dataset(args)
    print(f"Four-class dataset written to: {args.processed_root}")
    for key in sorted(summary):
        print(f"{key}: {summary[key]}")


if __name__ == "__main__":
    main()
