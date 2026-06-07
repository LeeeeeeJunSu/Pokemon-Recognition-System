from __future__ import annotations

import argparse
import json
from pathlib import Path

from create_resized_pokemon_dataset import build_resized_dataset


EXPECTED_CLASS_COUNT = 151
IMAGE_SIZE = (100, 100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a 100x100 copy of the full 151-class processed Pokemon dataset. "
            "Images are resized while audio files and train/val/test splits are preserved."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("Data/Pokemon/processed"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("Data/Pokemon/processed_100x100"),
    )
    return parser.parse_args()


def validate_full_dataset(source_root: Path) -> dict[str, int]:
    labels_path = source_root / "labels.json"
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels.json: {labels_path}")

    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    if len(labels) != EXPECTED_CLASS_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_CLASS_COUNT} classes, found {len(labels)}: {labels_path}"
        )

    split_counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        split_root = source_root / split
        if not split_root.exists():
            raise FileNotFoundError(f"Missing split directory: {split_root}")

        class_count = sum(1 for path in split_root.iterdir() if path.is_dir())
        if class_count != EXPECTED_CLASS_COUNT:
            raise ValueError(
                f"Expected {EXPECTED_CLASS_COUNT} {split} classes, found {class_count}"
            )
        split_counts[split] = sum(1 for _ in split_root.glob("*/*/image.png"))

    return split_counts


def main() -> None:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    split_counts = validate_full_dataset(source_root)

    resized_count, meta_count = build_resized_dataset(
        source_root=source_root,
        output_root=output_root,
        size=IMAGE_SIZE,
    )
    expected_images = sum(split_counts.values())
    if resized_count != expected_images or meta_count != expected_images:
        raise RuntimeError(
            "Generated file counts do not match the source dataset: "
            f"expected={expected_images}, images={resized_count}, meta={meta_count}"
        )

    print(f"Full 100x100 dataset written to: {output_root}")
    print(f"classes: {EXPECTED_CLASS_COUNT}")
    for split, count in split_counts.items():
        print(f"{split}_samples: {count}")
    print(f"resized_images: {resized_count}")
    print(f"updated_meta_files: {meta_count}")


if __name__ == "__main__":
    main()
