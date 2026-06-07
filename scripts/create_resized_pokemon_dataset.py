from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image, ImageOps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy a processed Pokemon dataset and resize only its image.png files. "
            "Audio files and dataset structure are preserved."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("Data/Pokemon/processed_4class"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("Data/Pokemon/processed_4class_100x100"),
    )
    parser.add_argument("--width", type=int, default=100)
    parser.add_argument("--height", type=int, default=100)
    return parser.parse_args()


def resize_image(source: Path, destination: Path, size: tuple[int, int]) -> None:
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        resized = image.resize(size, Image.Resampling.LANCZOS)
        resized.save(destination, format="PNG")


def update_meta(path: Path, size: tuple[int, int]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    image_meta = payload.setdefault("image", {})
    image_meta["dataset_stored_size"] = [size[0], size[1]]
    image_meta["resolution_experiment"] = (
        "stored at reduced resolution for use with a matching model input size"
    )
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_resized_dataset(
    source_root: Path,
    output_root: Path,
    size: tuple[int, int],
) -> tuple[int, int]:
    if not source_root.exists():
        raise FileNotFoundError(f"Source dataset not found: {source_root}")
    if size[0] <= 0 or size[1] <= 0:
        raise ValueError(f"Image size must be positive: {size}")

    if output_root.exists():
        shutil.rmtree(output_root)
    shutil.copytree(source_root, output_root)

    resized_count = 0
    meta_count = 0
    for image_path in output_root.rglob("image.png"):
        resize_image(image_path, image_path, size)
        resized_count += 1

    for meta_path in output_root.rglob("meta.json"):
        update_meta(meta_path, size)
        meta_count += 1

    summary_path = output_root / "dataset_summary.json"
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.exists()
        else {}
    )
    summary["source_dataset_root"] = str(source_root.as_posix())
    summary["stored_image_width"] = size[0]
    summary["stored_image_height"] = size[1]
    summary["resized_image_files"] = resized_count
    summary["model_input_note"] = (
        "Use model configs with image_size=100 and patch_size=10 for a true "
        "100x100 ViT input experiment."
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return resized_count, meta_count


def main() -> None:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    size = (args.width, args.height)
    resized_count, meta_count = build_resized_dataset(
        source_root,
        output_root,
        size,
    )
    print(f"Resized dataset written to: {output_root}")
    print(f"image_size: {size[0]}x{size[1]}")
    print(f"resized_images: {resized_count}")
    print(f"updated_meta_files: {meta_count}")


if __name__ == "__main__":
    main()
