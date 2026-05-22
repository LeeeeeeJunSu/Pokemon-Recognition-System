from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from urllib.request import urlopen

from PIL import Image


IMAGE_SOURCES = (
    ("images_inat", "images_inat.csv"),
    ("images_nabirds", "images_nabirds.csv"),
)
DEFAULT_DOWNLOAD_URL = "https://ml-inat-competition-datasets.s3.amazonaws.com/ssw60.tar.gz"
DEFAULT_ARCHIVE_MD5 = "af0a54ea1a897d130d91be8ffe0de81c"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the official SSW60 archive when needed, extract the raw files, "
            "optionally strip video assets, and build the project's processed layout."
        )
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("Data/SSW60/raw"),
        help="Extracted SSW60 root directory.",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("Data/SSW60/processed"),
        help="Output directory for the processed dataset.",
    )
    parser.add_argument(
        "--archive-path",
        type=Path,
        default=Path("Data/SSW60/ssw60.tar.gz"),
        help="Local path for the downloaded SSW60 tar.gz archive.",
    )
    parser.add_argument(
        "--download-url",
        type=str,
        default=DEFAULT_DOWNLOAD_URL,
        help="Official SSW60 archive URL.",
    )
    parser.add_argument(
        "--expected-md5",
        type=str,
        default=DEFAULT_ARCHIVE_MD5,
        help="Expected MD5 checksum for the SSW60 tar.gz archive.",
    )
    parser.add_argument(
        "--skip-md5-check",
        action="store_true",
        help="Skip archive checksum verification.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help=(
            "If the source dataset has no paired validation split, move this fraction "
            "of train samples into val."
        ),
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Do not access the network. Use an existing extracted raw dataset or archive only.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload the archive even if a local archive already exists.",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep the tar.gz archive after extraction.",
    )
    parser.add_argument(
        "--keep-videos",
        action="store_true",
        help="Keep extracted video assets instead of removing them.",
    )
    parser.add_argument(
        "--download-timeout-seconds",
        type=int,
        default=60,
        help="HTTP timeout in seconds for archive download.",
    )
    return parser.parse_args()


def sort_asset_id(value: str) -> tuple[int, str]:
    try:
        return (0, f"{int(value):020d}")
    except ValueError:
        return (1, value)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_raw_root(raw_root: Path) -> Path:
    if (raw_root / "taxa.csv").exists():
        return raw_root
    nested = raw_root / "ssw60"
    if (nested / "taxa.csv").exists():
        return nested
    raise FileNotFoundError(f"Could not locate SSW60 metadata under: {raw_root}")


def raw_dataset_exists(raw_root: Path) -> bool:
    try:
        normalize_raw_root(raw_root)
        return True
    except FileNotFoundError:
        return False


def compute_md5(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_md5(path: Path, expected_md5: str) -> None:
    actual_md5 = compute_md5(path)
    if actual_md5.lower() != expected_md5.lower():
        raise ValueError(
            f"Archive checksum mismatch for {path}. "
            f"Expected {expected_md5}, got {actual_md5}."
        )


def download_file(url: str, destination: Path, timeout_seconds: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".part")
    if temp_path.exists():
        temp_path.unlink()

    print(f"Downloading archive from: {url}")
    with urlopen(url, timeout=timeout_seconds) as response, temp_path.open("wb") as handle:
        total_bytes_header = response.headers.get("Content-Length")
        total_bytes = int(total_bytes_header) if total_bytes_header else None
        downloaded_bytes = 0
        last_reported_percent = -1

        while True:
            chunk = response.read(8 * 1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            downloaded_bytes += len(chunk)

            if total_bytes:
                percent = int(downloaded_bytes * 100 / total_bytes)
                if percent >= last_reported_percent + 5 or percent == 100:
                    print(
                        f"download_progress: {percent}% "
                        f"({downloaded_bytes / (1024 ** 3):.2f}GB / {total_bytes / (1024 ** 3):.2f}GB)"
                    )
                    last_reported_percent = percent
            elif downloaded_bytes % (512 * 1024 * 1024) < len(chunk):
                print(f"downloaded: {downloaded_bytes / (1024 ** 3):.2f}GB")

    temp_path.replace(destination)


def _safe_extract_tar(archive_path: Path, destination_root: Path) -> None:
    destination_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        destination_root_resolved = destination_root.resolve()
        for member in members:
            target_path = (destination_root / member.name).resolve()
            if os.path.commonpath([str(destination_root_resolved), str(target_path)]) != str(
                destination_root_resolved
            ):
                raise ValueError(f"Unsafe archive member path: {member.name}")
        try:
            archive.extractall(destination_root, filter="data")
        except TypeError:
            archive.extractall(destination_root)


def remove_video_assets(raw_root: Path) -> Counter:
    summary: Counter = Counter()

    video_dir = raw_root / "video_ml"
    if video_dir.exists():
        shutil.rmtree(video_dir)
        summary["directories_removed"] += 1

    video_csv = raw_root / "video_ml.csv"
    if video_csv.exists():
        video_csv.unlink()
        summary["files_removed"] += 1

    for video_file in raw_root.rglob("*.mp4"):
        if video_file.exists():
            video_file.unlink()
            summary["files_removed"] += 1

    return summary


def ensure_raw_dataset_ready(args: argparse.Namespace) -> Path:
    raw_root = args.raw_root.expanduser().resolve()
    archive_path = args.archive_path.expanduser().resolve()

    if raw_dataset_exists(raw_root):
        normalized = normalize_raw_root(raw_root)
        print(f"Using existing raw dataset: {normalized}")
    else:
        archive_needs_download = args.force_download or not archive_path.exists()

        if not archive_needs_download and args.expected_md5 and not args.skip_md5_check:
            try:
                print(f"Verifying existing archive checksum: {archive_path}")
                verify_md5(archive_path, args.expected_md5)
            except ValueError:
                if args.skip_download:
                    raise
                print("Existing archive checksum mismatch. Redownloading.")
                archive_path.unlink(missing_ok=True)
                archive_needs_download = True

        if archive_needs_download:
            if args.skip_download:
                raise FileNotFoundError(
                    "Raw dataset not found and downloading is disabled. "
                    "Provide an extracted raw dataset or a valid local archive."
                )
            download_file(args.download_url, archive_path, args.download_timeout_seconds)

        if args.expected_md5 and not args.skip_md5_check:
            print(f"Verifying archive checksum: {archive_path}")
            verify_md5(archive_path, args.expected_md5)

        print(f"Extracting archive into: {raw_root}")
        _safe_extract_tar(archive_path, raw_root)
        normalized = normalize_raw_root(raw_root)

    if not args.keep_videos:
        removal_summary = remove_video_assets(normalized)
        print(
            "removed_video_assets: "
            f"{removal_summary['directories_removed']} directories, "
            f"{removal_summary['files_removed']} files"
        )

    if archive_path.exists() and not args.keep_archive:
        archive_path.unlink()
        print(f"Removed archive: {archive_path}")

    return normalize_raw_root(raw_root)


def build_taxa_map(raw_root: Path) -> dict[int, dict[str, str]]:
    taxa_rows = read_csv_rows(raw_root / "taxa.csv")
    taxa_map: dict[int, dict[str, str]] = {}
    for row in taxa_rows:
        taxa_map[int(row["label"])] = row
    return taxa_map


def collect_audio_assets(raw_root: Path) -> dict[tuple[str, int], list[dict[str, str]]]:
    groups: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in read_csv_rows(raw_root / "audio_ml.csv"):
        asset_id = row["asset_id"]
        path = raw_root / "audio_ml" / f"{asset_id}.wav"
        if not path.exists():
            continue
        label = int(row["label"])
        split = row["split"]
        groups[(split, label)].append(
            {
                "asset_id": asset_id,
                "path": str(path),
            }
        )

    for assets in groups.values():
        assets.sort(key=lambda item: sort_asset_id(item["asset_id"]))
    return groups


def collect_image_assets(raw_root: Path) -> dict[tuple[str, int], list[dict[str, str]]]:
    groups: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for source_name, csv_name in IMAGE_SOURCES:
        for row in read_csv_rows(raw_root / csv_name):
            asset_id = row["asset_id"]
            path = raw_root / source_name / f"{asset_id}.jpg"
            if not path.exists():
                continue
            label = int(row["label"])
            split = row["split"]
            groups[(split, label)].append(
                {
                    "asset_id": asset_id,
                    "path": str(path),
                    "source": source_name,
                }
            )

    for assets in groups.values():
        assets.sort(key=lambda item: (item["source"], sort_asset_id(item["asset_id"])))
    return groups


def ensure_clean_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_labels_json(
    processed_root: Path,
    taxa_map: dict[int, dict[str, str]],
    included_labels: set[int],
) -> None:
    payload: dict[str, dict[str, object]] = {}
    for label in sorted(included_labels):
        taxa = taxa_map.get(label, {})
        payload[str(label)] = {
            "index": label,
            "species_code": taxa.get("species_code"),
            "common_name": taxa.get("common_name"),
            "scientific_name": taxa.get("scientific_name"),
            "family": taxa.get("family"),
            "order": taxa.get("order"),
        }

    with (processed_root / "labels.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def update_sample_meta(sample_dir: Path, split: str, derived_from: str | None = None) -> None:
    meta_path = sample_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["split"] = split
    if derived_from is not None:
        meta["derived_split_from"] = derived_from
        meta["split_strategy"] = "classwise deterministic slice from processed train split"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def link_or_copy_audio(src: Path, dst: Path) -> str:
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def convert_image_to_png(src: Path, dst: Path) -> None:
    with Image.open(src) as image:
        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGB")
        image.save(dst, format="PNG")


def build_processed_dataset(raw_root: Path, processed_root: Path) -> Counter:
    taxa_map = build_taxa_map(raw_root)
    audio_groups = collect_audio_assets(raw_root)
    image_groups = collect_image_assets(raw_root)
    included_labels: set[int] = set()
    summary: Counter = Counter()

    ensure_clean_directory(processed_root)
    for split_name in ("train", "val", "test"):
        (processed_root / split_name).mkdir(parents=True, exist_ok=True)

    all_keys = sorted(set(audio_groups) | set(image_groups), key=lambda item: (item[0], item[1]))
    for split, label in all_keys:
        audio_assets = audio_groups.get((split, label), [])
        image_assets = image_groups.get((split, label), [])
        pair_count = min(len(audio_assets), len(image_assets))

        summary[f"{split}_audio_assets"] += len(audio_assets)
        summary[f"{split}_image_assets"] += len(image_assets)
        summary[f"{split}_paired_samples"] += pair_count

        if pair_count == 0:
            continue

        included_labels.add(label)
        class_dir = processed_root / split / str(label)
        class_dir.mkdir(parents=True, exist_ok=True)
        taxa = taxa_map.get(label, {})

        for image_asset, audio_asset in zip(image_assets[:pair_count], audio_assets[:pair_count]):
            sample_id = (
                f"img-{image_asset['source']}-{image_asset['asset_id']}"
                f"__aud-{audio_asset['asset_id']}"
            )
            sample_dir = class_dir / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)

            image_src = Path(image_asset["path"])
            audio_src = Path(audio_asset["path"])
            image_dst = sample_dir / "image.png"
            audio_dst = sample_dir / "audio.wav"

            convert_image_to_png(image_src, image_dst)
            audio_storage = link_or_copy_audio(audio_src, audio_dst)

            meta = {
                "dataset": "SSW60",
                "split": split,
                "label": label,
                "class_name": str(label),
                "species_code": taxa.get("species_code"),
                "common_name": taxa.get("common_name"),
                "scientific_name": taxa.get("scientific_name"),
                "pairing_strategy": "same split and label, sorted by source and asset_id",
                "image": {
                    "asset_id": int(image_asset["asset_id"]),
                    "source": image_asset["source"],
                    "original_path": str(image_src.as_posix()),
                },
                "audio": {
                    "asset_id": int(audio_asset["asset_id"]),
                    "original_path": str(audio_src.as_posix()),
                    "storage": audio_storage,
                },
            }
            with (sample_dir / "meta.json").open("w", encoding="utf-8") as handle:
                json.dump(meta, handle, ensure_ascii=False, indent=2)

    write_labels_json(processed_root, taxa_map, included_labels)
    return summary


def create_validation_from_train(
    processed_root: Path,
    summary: Counter,
    val_ratio: float,
) -> int:
    if val_ratio <= 0:
        return 0

    train_root = processed_root / "train"
    val_root = processed_root / "val"
    moved_total = 0

    for class_dir in sorted(train_root.iterdir(), key=lambda path: int(path.name)):
        if not class_dir.is_dir():
            continue

        sample_dirs = sorted(path for path in class_dir.iterdir() if path.is_dir())
        if len(sample_dirs) < 2:
            continue

        move_count = max(1, round(len(sample_dirs) * val_ratio))
        move_count = min(move_count, len(sample_dirs) - 1)
        destination_class_dir = val_root / class_dir.name
        destination_class_dir.mkdir(parents=True, exist_ok=True)

        for sample_dir in sample_dirs[:move_count]:
            destination = destination_class_dir / sample_dir.name
            shutil.move(str(sample_dir), str(destination))
            update_sample_meta(destination, split="val", derived_from="train")
            moved_total += 1

    if moved_total:
        summary["train_paired_samples"] -= moved_total
        summary["train_audio_assets"] -= moved_total
        summary["train_image_assets"] -= moved_total
        summary["val_paired_samples"] += moved_total
        summary["val_audio_assets"] += moved_total
        summary["val_image_assets"] += moved_total

    return moved_total


def main() -> None:
    args = parse_args()
    raw_root = ensure_raw_dataset_ready(args)
    processed_root = args.processed_root.resolve()
    summary = build_processed_dataset(raw_root, processed_root)
    moved_to_val = 0
    if summary["val_paired_samples"] == 0:
        moved_to_val = create_validation_from_train(
            processed_root,
            summary,
            args.val_ratio,
        )

    print(f"Processed dataset written to: {args.processed_root}")
    if moved_to_val:
        print(f"derived_val_samples_from_train: {moved_to_val}")
    for key in sorted(summary):
        print(f"{key}: {summary[key]}")


if __name__ == "__main__":
    main()
