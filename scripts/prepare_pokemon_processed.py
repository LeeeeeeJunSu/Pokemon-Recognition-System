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
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aiff", ".aif"}
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
            "images and cries. Train images can be expanded with phone-camera-style "
            "augmentation, and train audio can be mixed with background audio plus "
            "mild microphone effects for better real-world recording robustness."
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
        default=200,
        help="Minimum number of train samples to create per class after augmentation.",
    )
    parser.add_argument(
        "--target-val-count",
        type=int,
        default=50,
        help="Minimum number of validation samples to create per class after augmentation.",
    )
    parser.add_argument(
        "--target-test-count",
        type=int,
        default=50,
        help="Minimum number of test samples to create per class after augmentation.",
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
        help="Use lighter image augmentation instead of phone-camera-style backgrounds.",
    )
    parser.add_argument(
        "--audio-augment-copies",
        type=int,
        default=4,
        help=(
            "Number of deterministic augmented cry files to cache per Pokemon. "
            "Set 0 to reuse only the original cry."
        ),
    )
    parser.add_argument(
        "--audio-background-root",
        type=Path,
        default=None,
        help=(
            "Optional background audio directory for SNR-based cry/background mixing. "
            "If omitted, the script uses a background directory under raw root when present."
        ),
    )
    parser.add_argument(
        "--audio-mix-offset-step",
        type=float,
        default=3.0,
        help="Step size, in seconds, for selecting deterministic background offsets.",
    )
    parser.add_argument(
        "--audio-mix-snr-db",
        type=float,
        default=16.0,
        help="Base target SNR in dB when mixing Pokemon cries with background audio.",
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


def collect_audio_paths(root: Path | None) -> list[Path]:
    if root is None or not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )


def resolve_audio_background_root(raw_root: Path, requested_root: Path | None) -> Path | None:
    if requested_root is not None:
        resolved = requested_root.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Audio background root not found: {resolved}")
        return resolved

    for name in ("bgm", "BGM", "background", "backgrounds", "Background", "Backgrounds"):
        candidate = raw_root / name
        if candidate.exists():
            return candidate
    return None


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


def mix_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    signal_rms = float(np.sqrt(np.mean(signal**2) + 1e-12))
    noise_rms = float(np.sqrt(np.mean(noise**2) + 1e-12))
    target_noise_rms = signal_rms / (10 ** (snr_db / 20.0))
    mixed = signal + noise * (target_noise_rms / noise_rms)
    return normalize_audio(mixed)


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import soundfile as sf

        sf.write(path, normalize_audio(audio), sample_rate)
    except Exception:
        from scipy.io import wavfile

        wavfile.write(path, sample_rate, (normalize_audio(audio) * 32767).astype(np.int16))


def shift_audio(audio: np.ndarray, shift_samples: int) -> np.ndarray:
    if shift_samples == 0 or audio.size == 0:
        return audio.copy()

    shift_samples = max(-audio.shape[0] + 1, min(audio.shape[0] - 1, shift_samples))
    if shift_samples == 0:
        return audio.copy()

    shifted = np.zeros_like(audio)
    if shift_samples > 0:
        shifted[shift_samples:] = audio[:-shift_samples]
    else:
        shifted[:shift_samples] = audio[-shift_samples:]
    return shifted


def time_stretch_to_length(audio: np.ndarray, speed: float, target_length: int) -> np.ndarray:
    if audio.shape[0] <= 1 or target_length <= 1:
        return audio.copy()

    new_length = max(1, int(round(audio.shape[0] / speed)))
    x_old = np.linspace(0.0, 1.0, num=audio.shape[0], endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=new_length, endpoint=False)
    stretched = np.interp(x_new, x_old, audio).astype(np.float32)
    if stretched.shape[0] >= target_length:
        start = (stretched.shape[0] - target_length) // 2
        return stretched[start : start + target_length]
    return np.pad(stretched, (0, target_length - stretched.shape[0]))


def apply_room_reflection(audio: np.ndarray, sample_rate: int, rng: random.Random) -> np.ndarray:
    reflected = audio.copy()
    for _ in range(rng.randint(1, 2)):
        delay = int(round(rng.uniform(18.0, 55.0) * sample_rate / 1000.0))
        if delay <= 0 or delay >= audio.shape[0]:
            continue
        reflected[delay:] += audio[:-delay] * rng.uniform(0.05, 0.16)
    return reflected


def apply_soft_bandlimit(audio: np.ndarray) -> np.ndarray:
    if audio.shape[0] < 3:
        return audio.copy()
    kernel = np.array([0.08, 0.84, 0.08], dtype=np.float32)
    return np.convolve(audio, kernel, mode="same").astype(np.float32)


def add_microphone_noise(audio: np.ndarray, rng: random.Random) -> np.ndarray:
    rms = float(np.sqrt(np.mean(audio**2) + 1e-12))
    noise_std = max(0.0004, rms * rng.uniform(0.006, 0.025))
    return audio + rng_np_normal(rng, audio.shape[0], noise_std)


def augment_audio(audio: np.ndarray, sample_rate: int, rng: random.Random) -> np.ndarray:
    augmented = audio.copy()
    if augmented.size == 0:
        return augmented

    gain = rng.uniform(0.82, 1.12)
    augmented *= gain

    if rng.random() < 0.12:
        augmented = time_stretch_to_length(augmented, rng.uniform(0.985, 1.015), audio.shape[0])

    if rng.random() < 0.75:
        max_early_ms = 80.0
        max_late_ms = 120.0
        shift_ms = rng.uniform(-max_late_ms, max_early_ms)
        shift = int(round(shift_ms * sample_rate / 1000.0))
        augmented = shift_audio(augmented, shift)

    if rng.random() < 0.5:
        augmented = apply_room_reflection(augmented, sample_rate, rng)

    if rng.random() < 0.6:
        augmented = apply_soft_bandlimit(augmented)

    if rng.random() < 0.65:
        augmented = add_microphone_noise(augmented, rng)

    return normalize_audio(augmented)


def rng_np_normal(rng: random.Random, length: int, std: float) -> np.ndarray:
    seed = rng.randint(0, 2**32 - 1)
    return np.random.default_rng(seed).normal(0.0, std, length).astype(np.float32)


def try_mix_with_background(
    audio: np.ndarray,
    background_paths: list[Path],
    background_cache: dict[Path, np.ndarray],
    sample_rate: int,
    offset_step_sec: float,
    snr_db: float,
    rng: random.Random,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    if not background_paths:
        return None

    candidates = list(background_paths)
    rng.shuffle(candidates)
    offset_step_samples = max(1, int(round(max(0.001, offset_step_sec) * sample_rate)))

    for background_path in candidates:
        if background_path not in background_cache:
            try:
                background_cache[background_path], _ = load_audio(background_path, sample_rate)
            except RuntimeError:
                continue

        background = background_cache[background_path]
        if audio.shape[0] > background.shape[0]:
            continue

        max_start = background.shape[0] - audio.shape[0]
        offset_count = (max_start // offset_step_samples) + 1
        offset_index = rng.randrange(offset_count)
        start = offset_index * offset_step_samples
        end = start + audio.shape[0]
        mixed_snr_db = rng.uniform(max(3.0, snr_db - 2.5), snr_db + 3.5)
        mixed = mix_at_snr(audio, background[start:end], mixed_snr_db)
        return mixed, {
            "background_path": background_path,
            "offset_sec": round(start / float(sample_rate), 3),
            "snr_db": round(mixed_snr_db, 2),
        }

    return None


def prepare_audio_cache(
    raw_root: Path,
    processed_root: Path,
    pokemon_ids: list[int],
    sample_rate: int,
    audio_augment_copies: int,
    background_paths: list[Path],
    audio_mix_offset_step: float,
    audio_mix_snr_db: float,
    rng: random.Random,
) -> dict[int, list[dict[str, Any]]]:
    cache_root = processed_root / "_audio_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    audio_cache: dict[int, list[dict[str, Any]]] = {}
    background_cache: dict[Path, np.ndarray] = {}

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
            background_mix = None
            if background_paths and rng.random() < 0.75:
                background_mix = try_mix_with_background(
                    audio=audio,
                    background_paths=background_paths,
                    background_cache=background_cache,
                    sample_rate=sample_rate,
                    offset_step_sec=audio_mix_offset_step,
                    snr_db=audio_mix_snr_db,
                    rng=rng,
                )

            details: dict[str, Any] = {}
            if background_mix is None:
                augmented = augment_audio(audio, sample_rate, rng)
                augmentation = "microphone_recording"
            else:
                mixed, details = background_mix
                augmented = augment_audio(mixed, sample_rate, rng)
                augmentation = "background_mix_microphone"
            write_wav(augmented_path, augmented, sample_rate)
            variant = {
                "path": augmented_path,
                "augmentation": augmentation,
                "source_path": source_path,
            }
            variant.update(details)
            variants.append(variant)

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

    scale = rng.uniform(0.7, 0.92) * size / max_side
    new_size = (
        max(1, int(round(image.width * scale))),
        max(1, int(round(image.height * scale))),
    )
    sprite = image.resize(new_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    max_x = max(0, size - sprite.width)
    max_y = max(0, size - sprite.height)
    jitter = int(round(size * 0.07))
    x = int(round((size - sprite.width) / 2 + rng.randint(-jitter, jitter)))
    y = int(round((size - sprite.height) / 2 + rng.randint(-jitter, jitter)))
    x = max(0, min(max_x, x))
    y = max(0, min(max_y, y))
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
    noise = np.random.default_rng(noise_seed).normal(0, rng.uniform(2, 6), (size, size, 1))
    bg_np = np.array(background).astype(np.float32) + noise

    if rng.random() < 0.7:
        axis = np.linspace(rng.uniform(-12, 6), rng.uniform(4, 14), size, dtype=np.float32)
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
    shadow_alpha = alpha_blur_mask(foreground, rng.uniform(3.0, 7.0))
    opacity = rng.randint(20, 55)
    shadow.putalpha(shadow_alpha.point(lambda value: int(value * opacity / 255)))

    offset_x = rng.randint(-4, 8)
    offset_y = rng.randint(3, 12)
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

    if rng.random() < 0.18:
        fitted = ImageOps.mirror(fitted)

    fitted = fitted.rotate(
        rng.uniform(-10.0, 10.0),
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=(0, 0, 0, 0),
    )

    background = random_background(size, rng)
    composed = composite_shadow(background, fitted, rng)

    if rng.random() < 0.45:
        margin = rng.uniform(2.0, 12.0)
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

    composed = ImageEnhance.Brightness(composed).enhance(rng.uniform(0.86, 1.16))
    composed = ImageEnhance.Contrast(composed).enhance(rng.uniform(0.9, 1.14))
    composed = ImageEnhance.Color(composed).enhance(rng.uniform(0.9, 1.12))

    if rng.random() < 0.35:
        composed = composed.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.15, 0.65)))

    if rng.random() < 0.18:
        composed = apply_cutout(composed, rng)

    if rng.random() < 0.45:
        composed = apply_sensor_noise(composed, rng)

    if rng.random() < 0.45:
        composed = apply_jpeg_degradation(composed, rng)

    return composed.resize((224, 224), Image.Resampling.LANCZOS)


def apply_light_image_augmentation(image: Image.Image, rng: random.Random) -> Image.Image:
    rgb = Image.new("RGB", image.size, (255, 255, 255))
    rgb.paste(image.convert("RGBA"), mask=image.convert("RGBA").getchannel("A"))
    if rng.random() < 0.25:
        rgb = ImageOps.mirror(rgb)
    rgb = rgb.rotate(
        rng.uniform(-8.0, 8.0),
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=(255, 255, 255),
    )
    rgb = ImageEnhance.Brightness(rgb).enhance(rng.uniform(0.92, 1.08))
    rgb = ImageEnhance.Contrast(rgb).enhance(rng.uniform(0.95, 1.08))
    rgb = ImageEnhance.Color(rgb).enhance(rng.uniform(0.95, 1.08))
    return ImageOps.contain(rgb, (224, 224), Image.Resampling.LANCZOS)


def apply_cutout(image: Image.Image, rng: random.Random) -> Image.Image:
    output = image.copy()
    width, height = output.size
    occluder_colors = [
        (235, 230, 220),
        (222, 224, 226),
        (48, 48, 46),
        (202, 177, 151),
    ]
    draw_color = tuple(clamp_channel(channel + rng.randint(-10, 10)) for channel in rng.choice(occluder_colors))
    hole_width = rng.randint(max(6, width // 18), max(8, width // 9))
    hole_height = rng.randint(max(6, height // 18), max(8, height // 9))
    if rng.random() < 0.7:
        side = rng.choice(("left", "right", "top", "bottom"))
        if side == "left":
            x = 0
            y = rng.randint(0, max(0, height - hole_height))
        elif side == "right":
            x = width - hole_width
            y = rng.randint(0, max(0, height - hole_height))
        elif side == "top":
            x = rng.randint(0, max(0, width - hole_width))
            y = 0
        else:
            x = rng.randint(0, max(0, width - hole_width))
            y = height - hole_height
    else:
        x = rng.randint(0, max(0, width - hole_width))
        y = rng.randint(0, max(0, height - hole_height))
    patch = Image.new("RGB", (hole_width, hole_height), draw_color)
    if rng.random() < 0.5:
        patch = patch.filter(ImageFilter.GaussianBlur(radius=0.6))
    output.paste(patch, (x, y))
    return output


def apply_sensor_noise(image: Image.Image, rng: random.Random) -> Image.Image:
    array = np.array(image).astype(np.float32)
    seed = rng.randint(0, 2**32 - 1)
    noise = np.random.default_rng(seed).normal(0, rng.uniform(1.0, 4.5), array.shape)
    return Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8), mode="RGB")


def apply_jpeg_degradation(image: Image.Image, rng: random.Random) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=rng.randint(72, 95))
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
    audio_background: Path | None = None,
    audio_mix_offset_sec: float | None = None,
    audio_mix_snr_db: float | None = None,
) -> None:
    audio_meta: dict[str, Any] = {
        "original_path": str(audio_source.as_posix()),
        "augmentation": audio_augmentation,
        "storage": audio_storage,
    }
    if audio_background is not None:
        audio_meta["background_path"] = str(audio_background.as_posix())
    if audio_mix_offset_sec is not None:
        audio_meta["offset_sec"] = audio_mix_offset_sec
    if audio_mix_snr_db is not None:
        audio_meta["snr_db"] = audio_mix_snr_db

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
        "audio": audio_meta,
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
        audio_background=(
            Path(audio_variant["background_path"])
            if "background_path" in audio_variant
            else None
        ),
        audio_mix_offset_sec=audio_variant.get("offset_sec"),
        audio_mix_snr_db=audio_variant.get("snr_db"),
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


def build_augmented_records(
    paths: list[Path],
    target_count: int,
    image_augmentation: str,
    rng: random.Random,
) -> list[tuple[Path, str]]:
    if not paths:
        return []

    records: list[tuple[Path, str]] = [(path, image_augmentation) for path in paths]
    target_count = max(len(records), target_count)
    while len(records) < target_count:
        records.append((rng.choice(paths), image_augmentation))
    return records


def select_augmented_audio_variant(audio_variants: list[dict[str, Any]], index: int) -> dict[str, Any]:
    if len(audio_variants) <= 1:
        return audio_variants[0]
    augmented_variants = audio_variants[1:]
    return augmented_variants[index % len(augmented_variants)]


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
    audio_background_root = resolve_audio_background_root(raw_root, args.audio_background_root)
    background_paths = collect_audio_paths(audio_background_root)
    if args.audio_background_root is not None and not background_paths:
        raise ValueError(f"No background audio files found under: {audio_background_root}")
    audio_cache = prepare_audio_cache(
        raw_root=raw_root,
        processed_root=processed_root,
        pokemon_ids=[pokemon_id for pokemon_id, _ in classes],
        sample_rate=args.sample_rate,
        audio_augment_copies=max(0, args.audio_augment_copies),
        background_paths=background_paths,
        audio_mix_offset_step=args.audio_mix_offset_step,
        audio_mix_snr_db=args.audio_mix_snr_db,
        rng=rng,
    )

    summary: Counter = Counter()
    camera_augmentation = not args.disable_camera_augmentation
    summary["audio_background_files"] = len(background_paths)

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

        image_augmentation = "camera" if camera_augmentation else "light"
        split_records = {
            "train": build_augmented_records(
                train_paths,
                max(0, args.target_train_count),
                image_augmentation,
                rng,
            ),
            "val": build_augmented_records(
                val_paths,
                max(0, args.target_val_count),
                image_augmentation,
                rng,
            ),
            "test": build_augmented_records(
                test_source,
                max(0, args.target_test_count),
                image_augmentation,
                rng,
            ),
        }

        for split, records in split_records.items():
            for index, (image_path, record_image_augmentation) in enumerate(records):
                digest = stable_digest(
                    f"{split}:{pokemon_id}:{image_path}:{index}:{record_image_augmentation}"
                )
                sample_id = f"{pokemon_id:03d}_{safe_stem(image_path.stem)}_{index:04d}_{digest}"
                audio_variant = select_augmented_audio_variant(audio_variants, index)
                create_sample(
                    processed_root=processed_root,
                    split=split,
                    class_name=class_name,
                    label_index=label_index,
                    pokemon_id=pokemon_id,
                    image_path=image_path,
                    audio_variant=audio_variant,
                    sample_id=sample_id,
                    rng=rng,
                    image_augmentation=record_image_augmentation,
                    camera_augmentation=camera_augmentation,
                )
                summary[f"{split}_samples"] += 1
                summary[f"augmented_{split}_images"] += 1

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
