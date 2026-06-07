from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
POKEMON_NAME_CORRECTIONS = {
    "Exeggcutor": "Exeggutor",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create train_v2/val_v2/test_v2 image directories. Existing train/test "
            "image folders are folded into train_v2, and the independent test dump "
            "is split class-wise into train/val/test."
        )
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("Data/Pokemon/raw/Image"),
        help="Root containing train, test, and extracted image folders.",
    )
    parser.add_argument(
        "--pokemon-dict",
        type=Path,
        default=Path("Data/Pokemon/raw/pokemon_dict.json"),
        help="pokemon_dict.json used to define canonical class names.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.6,
        help="Train fraction for test-20260606T125511Z-3-001/test.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation fraction for test-20260606T125511Z-3-001/test.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Test fraction for test-20260606T125511Z-3-001/test.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic per-class splits.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing train_v2/val_v2/test_v2 before creating them.",
    )
    return parser.parse_args()


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def safe_stem(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "image"


def short_digest(value: str, length: int = 12) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def read_canonical_classes(pokemon_dict: Path) -> list[str]:
    payload = json.loads(pokemon_dict.read_text(encoding="utf-8"))
    classes: list[tuple[int, str]] = []
    for key, value in payload.items():
        class_name = POKEMON_NAME_CORRECTIONS.get(str(value), str(value))
        classes.append((int(key), class_name))
    return [class_name for _, class_name in sorted(classes)]


def collect_class_dirs(root: Path) -> dict[str, Path]:
    dirs: dict[str, Path] = {}
    if not root.exists():
        raise FileNotFoundError(f"Missing image source root: {root}")
    for path in root.iterdir():
        if path.is_dir():
            dirs[normalize_name(path.name)] = path
    return dirs


def collect_image_files(class_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def clean_output_dirs(image_root: Path, overwrite: bool) -> None:
    for split in ("train_v2", "val_v2", "test_v2"):
        path = image_root / split
        if path.exists():
            if not overwrite:
                raise FileExistsError(f"Output directory already exists: {path}")
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


def allocate_counts(total: int, train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    if total <= 0:
        return 0, 0, 0
    if total == 1:
        return 1, 0, 0
    if total == 2:
        return 1, 1, 0

    ratio_sum = train_ratio + val_ratio + test_ratio
    if ratio_sum <= 0:
        raise ValueError("Split ratios must sum to a positive value.")
    train_ratio /= ratio_sum
    val_ratio /= ratio_sum
    test_ratio /= ratio_sum

    val_count = max(1, int(round(total * val_ratio)))
    test_count = max(1, int(round(total * test_ratio)))
    train_count = total - val_count - test_count
    while train_count < 1:
        if val_count >= test_count and val_count > 1:
            val_count -= 1
        elif test_count > 1:
            test_count -= 1
        else:
            break
        train_count = total - val_count - test_count
    return train_count, val_count, test_count


def destination_name(source: Path, source_tag: str, class_name: str) -> str:
    digest = short_digest(f"{source_tag}:{class_name}:{source.resolve()}:{source.stat().st_size}")
    return f"{source_tag}__{safe_stem(source.stem)}__{digest}{source.suffix.lower()}"


def copy_image(
    source: Path,
    image_root: Path,
    split: str,
    class_name: str,
    source_tag: str,
) -> dict[str, Any]:
    destination_dir = image_root / split / class_name
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / destination_name(source, source_tag, class_name)
    if destination.exists():
        raise FileExistsError(f"Destination image already exists: {destination}")
    shutil.copy2(source, destination)
    return {
        "split": split,
        "class_name": class_name,
        "source_tag": source_tag,
        "source_path": str(source.as_posix()),
        "destination_path": str(destination.as_posix()),
    }


def split_independent_files(
    files: list[Path],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    rng: random.Random,
) -> dict[str, list[Path]]:
    shuffled = list(files)
    rng.shuffle(shuffled)
    train_count, val_count, test_count = allocate_counts(
        len(shuffled),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    train_files = sorted(shuffled[:train_count])
    val_files = sorted(shuffled[train_count : train_count + val_count])
    test_files = sorted(shuffled[train_count + val_count : train_count + val_count + test_count])
    return {
        "train_v2": train_files,
        "val_v2": val_files,
        "test_v2": test_files,
    }


def build_dataset(args: argparse.Namespace) -> tuple[Counter, list[dict[str, Any]]]:
    image_root = args.image_root.resolve()
    pokemon_dict = args.pokemon_dict.resolve()
    classes = read_canonical_classes(pokemon_dict)

    train_sources = [
        ("raw_train", image_root / "train"),
        ("raw_test_as_train", image_root / "test"),
        ("extra_train", image_root / "train-20260606T125513Z-3-001" / "train"),
    ]
    split_source_tag = "extra_test_split"
    split_source = image_root / "test-20260606T125511Z-3-001" / "test"

    source_dirs = {tag: collect_class_dirs(root) for tag, root in train_sources}
    split_dirs = collect_class_dirs(split_source)

    missing: list[str] = []
    for class_name in classes:
        normalized = normalize_name(class_name)
        for tag, _ in train_sources:
            if normalized not in source_dirs[tag]:
                missing.append(f"{tag}:{class_name}")
        if normalized not in split_dirs:
            missing.append(f"{split_source_tag}:{class_name}")
    if missing:
        raise FileNotFoundError("Missing class directories: " + ", ".join(missing[:20]))

    clean_output_dirs(image_root, overwrite=args.overwrite)
    manifest: list[dict[str, Any]] = []
    summary: Counter = Counter()
    rng = random.Random(args.seed)

    for class_name in classes:
        normalized = normalize_name(class_name)

        for tag, _ in train_sources:
            for source in collect_image_files(source_dirs[tag][normalized]):
                record = copy_image(
                    source=source,
                    image_root=image_root,
                    split="train_v2",
                    class_name=class_name,
                    source_tag=tag,
                )
                manifest.append(record)
                summary["train_v2_files"] += 1
                summary[f"train_v2_from_{tag}"] += 1

        split_files = collect_image_files(split_dirs[normalized])
        per_class_rng = random.Random(f"{args.seed}:{class_name}")
        split_map = split_independent_files(
            split_files,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            rng=per_class_rng,
        )
        for split, files in split_map.items():
            for source in files:
                record = copy_image(
                    source=source,
                    image_root=image_root,
                    split=split,
                    class_name=class_name,
                    source_tag=split_source_tag,
                )
                manifest.append(record)
                summary[f"{split}_files"] += 1
                summary[f"{split}_from_{split_source_tag}"] += 1
        summary["classes"] += 1

    manifest_json = image_root / "image_split_v2_manifest.json"
    manifest_csv = image_root / "image_split_v2_manifest.csv"
    summary_json = image_root / "image_split_v2_summary.json"
    manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_json.write_text(json.dumps(dict(sorted(summary.items())), ensure_ascii=False, indent=2), encoding="utf-8")
    with manifest_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "class_name", "source_tag", "source_path", "destination_path"],
        )
        writer.writeheader()
        writer.writerows(manifest)

    return summary, manifest


def main() -> None:
    args = parse_args()
    summary, _ = build_dataset(args)
    print(f"Created image dataset v2 under: {args.image_root}")
    for key in sorted(summary):
        print(f"{key}: {summary[key]}")


if __name__ == "__main__":
    main()
