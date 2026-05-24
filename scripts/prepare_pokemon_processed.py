from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import random
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
DEFAULT_SAMPLE_RATE = 22050
RAW_CLASS_ALIASES = {
    "Farfetchd": ("Farfetch",),
}
POKEMON_NAME_CORRECTIONS = {
    "Exeggcutor": "Exeggutor",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the project's paired image/audio dataset layout from the Pokemon raw "
            "images and cries. Train images can be expanded with camera-style "
            "augmentation for better real-world camera robustness."
        )
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("Data/Pokemon/raw"),
        help="Raw Pokemon dataset root. Expected: Image/train, Image/test, cry, pokemon_dict.json.",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("Data/Pokemon/processed"),
        help="Output directory for the processed dataset.",
    )
    parser.add_argument(
        "--target-train-count",
        type=int,
        default=80,
        help="Minimum number of train samples to create per class after augmentation.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of raw train images to reserve for validation before augmentation.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help="Output wav sample rate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic splits and augmentation choices.",
    )
    parser.add_argument(
        "--disable-camera-augmentation",
        action="store_true",
        help="Use lighter image augmentation instead of camera-style synthetic scenes.",
    )
    parser.add_argument(
        "--audio-augment-copies",
        type=int,
        default=4,
        help=(
            "Number of deterministic augmented cry files to cache per Pokemon for train samples. "
            "Set 0 to reuse only the original cry."
        ),
    )
    return parser.parse_args()


def ensure_clean_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def read_pokemon_map(raw_root: Path) -> dict[int, str]:
    map_path = raw_root / "pokemon_dict.json"
    if not map_path.exists():
        raise FileNotFoundError(f"Missing pokemon_dict.json: {map_path}")

    payload = json.loads(map_path.read_text(encoding="utf-8"))
    pokemon_map = {
        int(key): POKEMON_NAME_CORRECTIONS.get(str(value), str(value))
        for key, value in payload.items()
    }
    if not pokemon_map:
        raise ValueError(f"No Pokemon labels found in: {map_path}")
    return pokemon_map


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def safe_stem(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "sample"


def stable_digest(value: str, length: int = 10) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def collect_image_paths(split_root: Path, class_name: str) -> list[Path]:
    candidates = (class_name, *RAW_CLASS_ALIASES.get(class_name, ()))
    class_dir = split_root / class_name
    for candidate in candidates:
        candidate_dir = split_root / candidate
        if candidate_dir.exists():
            class_dir = candidate_dir
            break

    if not class_dir.exists():
        normalized_candidates = {normalize_name(candidate) for candidate in candidates}
        matches = [
            path
            for path in split_root.iterdir()
            if path.is_dir() and normalize_name(path.name) in normalized_candidates
        ]
        if matches:
            class_dir = matches[0]

    if not class_dir.exists():
        return []

    return sorted(
        path
        for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def write_labels_json(
    processed_root: Path,
    classes: list[tuple[int, str]],
) -> dict[int, int]:
    pokemon_id_to_index = {pokemon_id: index for index, (pokemon_id, _) in enumerate(classes)}
    payload: dict[str, dict[str, Any]] = {}
    for pokemon_id, class_name in classes:
        payload[class_name] = {
            "index": pokemon_id_to_index[pokemon_id],
            "pokemon_id": pokemon_id,
            "class_name": class_name,
            "common_name": class_name,
        }

    with (processed_root / "labels.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return pokemon_id_to_index


def load_audio(path: Path, sample_rate: int) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf

        audio, source_rate = sf.read(path, always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
    except Exception:
        try:
            import torch
            import torchaudio

            waveform, source_rate = torchaudio.load(str(path))
            audio = waveform.mean(dim=0).numpy().astype(np.float32)
        except Exception as error:
            raise RuntimeError(f"Could not read audio file: {path}") from error

    if source_rate != sample_rate:
        audio = resample_audio(audio, source_rate, sample_rate)
    return normalize_audio(audio), sample_rate


def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio.astype(np.float32)

    try:
        from scipy.signal import resample_poly

        divisor = math.gcd(source_rate, target_rate)
        return resample_poly(audio, target_rate // divisor, source_rate // divisor).astype(np.float32)
    except Exception:
        duration = audio.shape[0] / float(source_rate)
        target_length = max(1, int(round(duration * target_rate)))
        source_positions = np.linspace(0.0, 1.0, num=audio.shape[0], endpoint=False)
        target_positions = np.linspace(0.0, 1.0, num=target_length, endpoint=False)
        return np.interp(target_positions, source_positions, audio).astype(np.float32)


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    audio = np.nan_to_num(audio.astype(np.float32))
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    return np.clip(audio, -1.0, 1.0)


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import soundfile as sf

        sf.write(path, normalize_audio(audio), sample_rate)
    except Exception:
        from scipy.io import wavfile

        wavfile.write(path, sample_rate, (normalize_audio(audio) * 32767).astype(np.int16))


def augment_audio(audio: np.ndarray, rng: random.Random) -> np.ndarray:
    augmented = audio.copy()

    gain = rng.uniform(0.75, 1.25)
    augmented *= gain

    if augmented.size > 4:
        shift = rng.randint(-max(1, augmented.size // 8), max(1, augmented.size // 8))
        augmented = np.roll(augmented, shift)

    if rng.random() < 0.55:
        noise_std = rng.uniform(0.001, 0.008)
        augmented += rng_np_normal(rng, augmented.shape[0], noise_std)

    if rng.random() < 0.35:
        speed = rng.uniform(0.9, 1.1)
        new_length = max(1, int(round(augmented.shape[0] / speed)))
        x_old = np.linspace(0.0, 1.0, num=augmented.shape[0], endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=new_length, endpoint=False)
        stretched = np.interp(x_new, x_old, augmented).astype(np.float32)
        if stretched.shape[0] >= audio.shape[0]:
            start = (stretched.shape[0] - audio.shape[0]) // 2
            augmented = stretched[start : start + audio.shape[0]]
        else:
            augmented = np.pad(stretched, (0, audio.shape[0] - stretched.shape[0]))

    return normalize_audio(augmented)


def rng_np_normal(rng: random.Random, length: int, std: float) -> np.ndarray:
    seed = rng.randint(0, 2**32 - 1)
    return np.random.default_rng(seed).normal(0.0, std, length).astype(np.float32)


def prepare_audio_cache(
    raw_root: Path,
    processed_root: Path,
    pokemon_ids: list[int],
    sample_rate: int,
    audio_augment_copies: int,
    rng: random.Random,
) -> dict[int, list[dict[str, Any]]]:
    cache_root = processed_root / "_audio_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    audio_cache: dict[int, list[dict[str, Any]]] = {}

    for pokemon_id in pokemon_ids:
        source_path = raw_root / "cry" / f"{pokemon_id}.ogg"
        if not source_path.exists():
            continue

        audio, _ = load_audio(source_path, sample_rate)
        variants: list[dict[str, Any]] = []

        original_path = cache_root / f"{pokemon_id:03d}_origin.wav"
        write_wav(original_path, audio, sample_rate)
        variants.append(
            {
                "path": original_path,
                "augmentation": "origin",
                "source_path": source_path,
            }
        )

        for index in range(audio_augment_copies):
            augmented_path = cache_root / f"{pokemon_id:03d}_aug_{index:02d}.wav"
            augmented = augment_audio(audio, rng)
            write_wav(augmented_path, augmented, sample_rate)
            variants.append(
                {
                    "path": augmented_path,
                    "augmentation": "gain_shift_noise_speed",
                    "source_path": source_path,
                }
            )

        audio_cache[pokemon_id] = variants

    return audio_cache


def open_rgba(path: Path) -> Image.Image:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        return image.convert("RGBA")


def fit_on_canvas(image: Image.Image, size: int, rng: random.Random) -> Image.Image:
    image = trim_transparent(image)
    max_side = max(image.size)
    if max_side <= 0:
        return Image.new("RGBA", (size, size), (255, 255, 255, 255))

    scale = rng.uniform(0.62, 0.88) * size / max_side
    new_size = (
        max(1, int(round(image.width * scale))),
        max(1, int(round(image.height * scale))),
    )
    sprite = image.resize(new_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    max_x = max(0, size - sprite.width)
    max_y = max(0, size - sprite.height)
    x = rng.randint(0, max_x) if max_x else 0
    y = rng.randint(0, max_y) if max_y else 0
    canvas.alpha_composite(sprite, (x, y))
    return canvas


def trim_transparent(image: Image.Image) -> Image.Image:
    if image.mode != "RGBA":
        return image
    alpha = image.getchannel("A")
    bbox = alpha.getbbox()
    if not bbox:
        return image
    return image.crop(bbox)


def random_background(size: int, rng: random.Random) -> Image.Image:
    base_colors = [
        (242, 239, 232),
        (225, 229, 232),
        (214, 224, 209),
        (235, 226, 214),
        (222, 218, 228),
        (245, 245, 240),
    ]
    color = tuple(clamp_channel(channel + rng.randint(-18, 18)) for channel in rng.choice(base_colors))
    background = Image.new("RGB", (size, size), color)

    noise_seed = rng.randint(0, 2**32 - 1)
    noise = np.random.default_rng(noise_seed).normal(0, rng.uniform(3, 10), (size, size, 1))
    bg_np = np.array(background).astype(np.float32) + noise

    if rng.random() < 0.7:
        axis = np.linspace(rng.uniform(-20, 10), rng.uniform(5, 24), size, dtype=np.float32)
        if rng.random() < 0.5:
            bg_np += axis[:, None, None]
        else:
            bg_np += axis[None, :, None]

    return Image.fromarray(np.clip(bg_np, 0, 255).astype(np.uint8), mode="RGB")


def clamp_channel(value: int) -> int:
    return max(0, min(255, value))


def alpha_blur_mask(image: Image.Image, radius: float) -> Image.Image:
    alpha = image.getchannel("A")
    return alpha.filter(ImageFilter.GaussianBlur(radius=radius))


def composite_shadow(background: Image.Image, foreground: Image.Image, rng: random.Random) -> Image.Image:
    shadow = Image.new("RGBA", foreground.size, (0, 0, 0, 0))
    shadow_alpha = alpha_blur_mask(foreground, rng.uniform(4.0, 10.0))
    opacity = rng.randint(35, 85)
    shadow.putalpha(shadow_alpha.point(lambda value: int(value * opacity / 255)))

    offset_x = rng.randint(-8, 12)
    offset_y = rng.randint(5, 16)
    layer = Image.new("RGBA", foreground.size, (0, 0, 0, 0))
    layer.alpha_composite(shadow, (offset_x, offset_y))
    layer.alpha_composite(foreground, (0, 0))

    combined = background.convert("RGBA")
    combined.alpha_composite(layer)
    return combined.convert("RGB")


def perspective_coefficients(source: list[tuple[float, float]], target: list[tuple[float, float]]) -> list[float]:
    matrix = []
    vector = []
    for (x, y), (u, v) in zip(source, target):
        matrix.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        matrix.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        vector.append(u)
        vector.append(v)
    return np.linalg.solve(np.array(matrix, dtype=np.float64), np.array(vector, dtype=np.float64)).tolist()


def apply_camera_image_augmentation(image: Image.Image, rng: random.Random, size: int = 256) -> Image.Image:
    fitted = fit_on_canvas(image, size=size, rng=rng)

    if rng.random() < 0.5:
        fitted = ImageOps.mirror(fitted)

    fitted = fitted.rotate(
        rng.uniform(-18.0, 18.0),
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=(0, 0, 0, 0),
    )

    background = random_background(size, rng)
    composed = composite_shadow(background, fitted, rng)

    if rng.random() < 0.7:
        margin = rng.uniform(4.0, 20.0)
        source = [(0, 0), (size, 0), (size, size), (0, size)]
        target = [
            (rng.uniform(0, margin), rng.uniform(0, margin)),
            (size - rng.uniform(0, margin), rng.uniform(0, margin)),
            (size - rng.uniform(0, margin), size - rng.uniform(0, margin)),
            (rng.uniform(0, margin), size - rng.uniform(0, margin)),
        ]
        coeffs = perspective_coefficients(target, source)
        composed = composed.transform(
            (size, size),
            Image.Transform.PERSPECTIVE,
            coeffs,
            Image.Resampling.BICUBIC,
        )

    composed = ImageEnhance.Brightness(composed).enhance(rng.uniform(0.72, 1.28))
    composed = ImageEnhance.Contrast(composed).enhance(rng.uniform(0.78, 1.25))
    composed = ImageEnhance.Color(composed).enhance(rng.uniform(0.82, 1.2))

    if rng.random() < 0.45:
        composed = composed.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 1.1)))

    if rng.random() < 0.45:
        composed = apply_cutout(composed, rng)

    if rng.random() < 0.6:
        composed = apply_sensor_noise(composed, rng)

    if rng.random() < 0.55:
        composed = apply_jpeg_degradation(composed, rng)

    return composed.resize((224, 224), Image.Resampling.LANCZOS)


def apply_light_image_augmentation(image: Image.Image, rng: random.Random) -> Image.Image:
    rgb = Image.new("RGB", image.size, (255, 255, 255))
    rgb.paste(image.convert("RGBA"), mask=image.convert("RGBA").getchannel("A"))
    if rng.random() < 0.5:
        rgb = ImageOps.mirror(rgb)
    rgb = rgb.rotate(
        rng.uniform(-12.0, 12.0),
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=(255, 255, 255),
    )
    rgb = ImageEnhance.Brightness(rgb).enhance(rng.uniform(0.85, 1.15))
    rgb = ImageEnhance.Contrast(rgb).enhance(rng.uniform(0.9, 1.15))
    rgb = ImageEnhance.Color(rgb).enhance(rng.uniform(0.9, 1.15))
    return ImageOps.contain(rgb, (224, 224), Image.Resampling.LANCZOS)


def apply_cutout(image: Image.Image, rng: random.Random) -> Image.Image:
    output = image.copy()
    width, height = output.size
    draw_color = tuple(clamp_channel(channel + rng.randint(-20, 20)) for channel in output.getpixel((0, 0)))
    holes = rng.randint(1, 3)
    for _ in range(holes):
        hole_width = rng.randint(max(8, width // 14), max(12, width // 5))
        hole_height = rng.randint(max(8, height // 14), max(12, height // 5))
        x = rng.randint(0, max(0, width - hole_width))
        y = rng.randint(0, max(0, height - hole_height))
        patch = Image.new("RGB", (hole_width, hole_height), draw_color)
        output.paste(patch, (x, y))
    return output


def apply_sensor_noise(image: Image.Image, rng: random.Random) -> Image.Image:
    array = np.array(image).astype(np.float32)
    seed = rng.randint(0, 2**32 - 1)
    noise = np.random.default_rng(seed).normal(0, rng.uniform(2.0, 8.0), array.shape)
    return Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8), mode="RGB")


def apply_jpeg_degradation(image: Image.Image, rng: random.Random) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=rng.randint(55, 88))
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        return compressed.convert("RGB")


def save_original_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    image = open_rgba(src)
    canvas = Image.new("RGB", image.size, (255, 255, 255))
    canvas.paste(image, mask=image.getchannel("A"))
    ImageOps.contain(canvas, (224, 224), Image.Resampling.LANCZOS).save(dst, format="PNG")


def copy_audio_variant(variant: dict[str, Any], destination: Path) -> str:
    source = Path(variant["path"])
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return "copy"
    except OSError:
        shutil.copyfile(source, destination)
        return "copyfile"


def write_meta(
    sample_dir: Path,
    dataset: str,
    split: str,
    class_name: str,
    label_index: int,
    pokemon_id: int,
    image_source: Path,
    audio_source: Path,
    image_augmentation: str,
    audio_augmentation: str,
    audio_storage: str,
) -> None:
    meta = {
        "dataset": dataset,
        "split": split,
        "label": label_index,
        "class_name": class_name,
        "pokemon_id": pokemon_id,
        "common_name": class_name,
        "image": {
            "original_path": str(image_source.as_posix()),
            "augmentation": image_augmentation,
        },
        "audio": {
            "original_path": str(audio_source.as_posix()),
            "augmentation": audio_augmentation,
            "storage": audio_storage,
        },
    }
    with (sample_dir / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)


def create_sample(
    processed_root: Path,
    split: str,
    class_name: str,
    label_index: int,
    pokemon_id: int,
    image_path: Path,
    audio_variant: dict[str, Any],
    sample_id: str,
    rng: random.Random,
    image_augmentation: str,
    camera_augmentation: bool,
) -> None:
    sample_dir = processed_root / split / class_name / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    image_dst = sample_dir / "image.png"
    if image_augmentation == "origin":
        save_original_image(image_path, image_dst)
    else:
        image = open_rgba(image_path)
        if camera_augmentation:
            output = apply_camera_image_augmentation(image, rng)
        else:
            output = apply_light_image_augmentation(image, rng)
        output.save(image_dst, format="PNG")

    audio_dst = sample_dir / "audio.wav"
    audio_storage = copy_audio_variant(audio_variant, audio_dst)
    write_meta(
        sample_dir=sample_dir,
        dataset="Pokemon",
        split=split,
        class_name=class_name,
        label_index=label_index,
        pokemon_id=pokemon_id,
        image_source=image_path,
        audio_source=Path(audio_variant["source_path"]),
        image_augmentation=image_augmentation,
        audio_augmentation=str(audio_variant["augmentation"]),
        audio_storage=audio_storage,
    )


def split_train_val(paths: list[Path], val_ratio: float, rng: random.Random) -> tuple[list[Path], list[Path]]:
    shuffled = list(paths)
    rng.shuffle(shuffled)
    if len(shuffled) < 2 or val_ratio <= 0:
        return sorted(shuffled), []

    val_count = max(1, round(len(shuffled) * val_ratio))
    val_count = min(val_count, len(shuffled) - 1)
    val_paths = sorted(shuffled[:val_count])
    train_paths = sorted(shuffled[val_count:])
    return train_paths, val_paths


def build_processed_dataset(args: argparse.Namespace) -> Counter:
    raw_root = args.raw_root.expanduser().resolve()
    processed_root = args.processed_root.expanduser().resolve()
    image_root = raw_root / "Image"

    if not raw_root.exists():
        raise FileNotFoundError(f"Raw root not found: {raw_root}")
    if not (image_root / "train").exists():
        raise FileNotFoundError(f"Missing train image directory: {image_root / 'train'}")
    if not (image_root / "test").exists():
        raise FileNotFoundError(f"Missing test image directory: {image_root / 'test'}")
    if not (raw_root / "cry").exists():
        raise FileNotFoundError(f"Missing cry directory: {raw_root / 'cry'}")

    rng = random.Random(args.seed)
    pokemon_map = read_pokemon_map(raw_root)
    classes: list[tuple[int, str]] = []
    for pokemon_id, class_name in sorted(pokemon_map.items()):
        has_cry = (raw_root / "cry" / f"{pokemon_id}.ogg").exists()
        has_train = bool(collect_image_paths(image_root / "train", class_name))
        has_test = bool(collect_image_paths(image_root / "test", class_name))
        if has_cry and (has_train or has_test):
            classes.append((pokemon_id, class_name))

    if not classes:
        raise ValueError(f"No usable Pokemon classes found under: {raw_root}")

    ensure_clean_directory(processed_root)
    for split in ("train", "val", "test"):
        (processed_root / split).mkdir(parents=True, exist_ok=True)

    label_indices = write_labels_json(processed_root, classes)
    audio_cache = prepare_audio_cache(
        raw_root=raw_root,
        processed_root=processed_root,
        pokemon_ids=[pokemon_id for pokemon_id, _ in classes],
        sample_rate=args.sample_rate,
        audio_augment_copies=max(0, args.audio_augment_copies),
        rng=rng,
    )

    summary: Counter = Counter()
    camera_augmentation = not args.disable_camera_augmentation

    for pokemon_id, class_name in classes:
        label_index = label_indices[pokemon_id]
        train_source = collect_image_paths(image_root / "train", class_name)
        test_source = collect_image_paths(image_root / "test", class_name)
        train_paths, val_paths = split_train_val(train_source, args.val_ratio, rng)
        audio_variants = audio_cache.get(pokemon_id, [])
        if not audio_variants:
            continue

        summary["classes"] += 1
        summary["raw_train_images"] += len(train_paths)
        summary["raw_val_images"] += len(val_paths)
        summary["raw_test_images"] += len(test_source)

        for split, paths in (("val", val_paths), ("test", test_source)):
            for index, image_path in enumerate(paths):
                sample_id = f"{pokemon_id:03d}_{safe_stem(image_path.stem)}_{index:03d}"
                create_sample(
                    processed_root=processed_root,
                    split=split,
                    class_name=class_name,
                    label_index=label_index,
                    pokemon_id=pokemon_id,
                    image_path=image_path,
                    audio_variant=audio_variants[0],
                    sample_id=sample_id,
                    rng=rng,
                    image_augmentation="origin",
                    camera_augmentation=camera_augmentation,
                )
                summary[f"{split}_samples"] += 1

        if not train_paths:
            continue

        train_records: list[tuple[Path, str]] = [(path, "origin") for path in train_paths]
        target_count = max(len(train_records), args.target_train_count)
        while len(train_records) < target_count:
            train_records.append((rng.choice(train_paths), "camera" if camera_augmentation else "light"))

        for index, (image_path, image_augmentation) in enumerate(train_records):
            digest = stable_digest(f"{pokemon_id}:{image_path}:{index}:{image_augmentation}")
            sample_id = f"{pokemon_id:03d}_{safe_stem(image_path.stem)}_{index:04d}_{digest}"
            audio_variant = audio_variants[index % len(audio_variants)]
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
            )
            summary["train_samples"] += 1
            if image_augmentation != "origin":
                summary["augmented_train_images"] += 1

    summary_path = processed_root / "dataset_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(dict(sorted(summary.items())), handle, ensure_ascii=False, indent=2)

    if (processed_root / "_audio_cache").exists():
        shutil.rmtree(processed_root / "_audio_cache")

    return summary


def main() -> None:
    args = parse_args()
    summary = build_processed_dataset(args)
    print(f"Processed dataset written to: {args.processed_root}")
    for key in sorted(summary):
        print(f"{key}: {summary[key]}")


if __name__ == "__main__":
    main()
