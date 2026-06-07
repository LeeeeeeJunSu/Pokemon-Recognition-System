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
    "Sandslash": ("Alolan Sandslash",),
}
POKEMON_NAME_CORRECTIONS = {
    "Exeggcutor": "Exeggutor",
}
EVALUATION_CONDITIONS = {
    "clean_sanity",
    "audio_hard",
    "image_hard",
    "both_hard",
    "balanced",
}
DEGRADATION_MODES = {
    "none",
    "tiny_speaker",
    "clipped_speaker",
    "phone_codec",
    "dropout",
    "crackle",
}


def parse_string_sweep(value: str | None, default: list[str]) -> list[str]:
    if value is None:
        return list(default)
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    return items or list(default)


def parse_float_sweep(value: str | None, default: list[float]) -> list[float]:
    items = parse_string_sweep(value, [str(item) for item in default])
    return [float(item) for item in items]


def validate_condition_sweep(name: str, conditions: list[str]) -> list[str]:
    invalid = sorted(set(conditions) - EVALUATION_CONDITIONS)
    if invalid:
        raise ValueError(f"Unsupported {name} condition(s): {', '.join(invalid)}")
    return conditions


def validate_degradation_sweep(name: str, modes: list[str]) -> list[str]:
    invalid = sorted(set(modes) - DEGRADATION_MODES)
    if invalid:
        raise ValueError(f"Unsupported {name} degradation mode(s): {', '.join(invalid)}")
    return modes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the project's Pokemon dataset layout as a compact robustness "
            "benchmark. Evaluation samples are controlled corruption/degradation "
            "conditions, not ordinary random augmentation."
        )
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("Data/Pokemon/1차 공유본"),
        help="First shared Pokemon dataset root. Expected: Image/train, cry, pokemon_dict.json.",
    )
    parser.add_argument(
        "--second-train-root",
        type=Path,
        default=Path("Data/Pokemon/2차 공유본 Train"),
        help="Second shared train image root. Expected: <class_name>/*.png.",
    )
    parser.add_argument(
        "--second-test-root",
        type=Path,
        default=Path("Data/Pokemon/2차 공유본 Test"),
        help="Second shared test image root. Expected: test/<class_name>/*.png or <class_name>/*.png.",
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
        default=70,
        help="Number of train samples to create per class.",
    )
    parser.add_argument(
        "--target-val-count",
        type=int,
        default=15,
        help="Number of validation samples to create per class.",
    )
    parser.add_argument(
        "--target-test-count",
        type=int,
        default=25,
        help="Number of test samples to create per class.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Deprecated. First shared images are all used for train.",
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
        "--processed-image-size",
        type=int,
        default=224,
        help="Maximum saved image side length for processed image.png files.",
    )
    parser.add_argument(
        "--audio-only",
        action="store_true",
        help="Create audio.wav and meta.json only, without image.png.",
    )
    parser.add_argument(
        "--use-image-v2",
        action="store_true",
        help="Use raw/Image/train_v2, val_v2, and test_v2 as fixed image splits.",
    )
    parser.add_argument(
        "--image-v2-augment-prefix",
        default="extra_test_split__",
        help=(
            "In --use-image-v2 mode, train_v2 images whose filenames start with this "
            "prefix receive image augmentation. Other train_v2 images are copied/resized only."
        ),
    )
    parser.add_argument(
        "--image-v2-extra-augment-copies",
        type=int,
        default=0,
        help=(
            "Additional augmented copies to create for matching train_v2 images. "
            "Default 0 keeps the sample count unchanged."
        ),
    )
    parser.add_argument(
        "--image-v2-eval-image-augmentation",
        action="store_true",
        help=(
            "Allow val_v2/test_v2 image augmentation for image_hard/both_hard conditions. "
            "By default v2 validation/test images remain origin."
        ),
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
            "Background audio directory for SNR-based cry/background mixing. "
            "If omitted, common background directories under raw root are tried."
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
    parser.add_argument(
        "--train-mix-bgm",
        action="store_true",
        help="Mix some train audio variants with disjoint training BGM tracks.",
    )
    parser.add_argument(
        "--train-bgm-fraction",
        type=float,
        default=0.34,
        help="Approximate fraction of train audio variants that use BGM when --train-mix-bgm is set.",
    )
    parser.add_argument(
        "--train-degradation-fraction",
        type=float,
        default=0.35,
        help="Approximate fraction of non-origin train variants with speaker/codec degradation.",
    )
    parser.add_argument(
        "--train-degradation-modes",
        default="tiny_speaker,clipped_speaker,phone_codec,dropout,crackle",
        help="Comma-separated degradation modes available for train audio.",
    )
    parser.add_argument(
        "--val-degradation",
        default="none",
        help="Legacy single validation degradation mode.",
    )
    parser.add_argument(
        "--val-degradation-sweep",
        default="none,tiny_speaker,phone_codec",
        help="Comma-separated validation degradation modes for hard audio conditions.",
    )
    parser.add_argument(
        "--test-degradation-sweep",
        default="none,tiny_speaker,clipped_speaker,phone_codec,dropout",
        help="Comma-separated test degradation modes for hard audio conditions.",
    )
    parser.add_argument(
        "--val-snr-db",
        type=float,
        default=6.0,
        help="Legacy single validation SNR value.",
    )
    parser.add_argument(
        "--val-snr-sweep",
        default="6,0",
        help="Comma-separated validation SNR values for hard audio conditions.",
    )
    parser.add_argument(
        "--test-snr-sweep",
        default="12,6,0,-6",
        help="Comma-separated test SNR values for hard audio conditions.",
    )
    parser.add_argument(
        "--val-condition-sweep",
        default="clean_sanity,audio_hard,balanced",
        help="Comma-separated validation conditions.",
    )
    parser.add_argument(
        "--test-condition-sweep",
        default="clean_sanity,audio_hard,image_hard,both_hard,balanced",
        help="Comma-separated test conditions.",
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
    class_dirs: list[Path] = []
    for candidate in candidates:
        candidate_dir = split_root / candidate
        if candidate_dir.exists():
            class_dirs.append(candidate_dir)

    if not class_dirs:
        normalized_candidates = {normalize_name(candidate) for candidate in candidates}
        class_dirs = [
            path
            for path in split_root.iterdir()
            if path.is_dir() and normalize_name(path.name) in normalized_candidates
        ]

    if not class_dirs:
        return []

    paths: list[Path] = []
    seen_dirs: set[Path] = set()
    for class_dir in class_dirs:
        resolved_dir = class_dir.resolve()
        if resolved_dir in seen_dirs:
            continue
        seen_dirs.add(resolved_dir)
        paths.extend(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    return sorted(paths)


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


def partition_background_tracks(
    background_paths: list[Path],
    train_fraction: float,
    val_fraction: float,
    train_enabled: bool,
    rng: random.Random,
) -> tuple[list[Path], list[Path], list[Path]]:
    paths = list(background_paths)
    rng.shuffle(paths)
    if not paths:
        return [], [], []
    if len(paths) < 3:
        return (paths if train_enabled else []), paths, paths

    train_count = int(round(len(paths) * max(0.0, min(1.0, train_fraction)))) if train_enabled else 0
    train_count = min(max(1, train_count), len(paths) - 2) if train_enabled else 0
    remaining = len(paths) - train_count
    val_count = int(round(remaining * max(0.0, min(1.0, val_fraction))))
    val_count = min(max(1, val_count), remaining - 1)

    train_paths = paths[:train_count]
    val_paths = paths[train_count : train_count + val_count]
    test_paths = paths[train_count + val_count :]
    return train_paths, val_paths, test_paths


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


def apply_tiny_speaker(audio: np.ndarray) -> np.ndarray:
    if audio.shape[0] < 5:
        return audio.copy()
    kernel = np.array([0.04, 0.12, 0.68, 0.12, 0.04], dtype=np.float32)
    filtered = np.convolve(audio, kernel, mode="same").astype(np.float32)
    return normalize_audio(filtered * 0.82)


def apply_clipped_speaker(audio: np.ndarray) -> np.ndarray:
    driven = audio * 1.8
    clipped = np.clip(driven, -0.55, 0.55)
    return normalize_audio(clipped)


def apply_phone_codec(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if audio.shape[0] < 8:
        return audio.copy()
    codec_rate = 8000
    down = resample_audio(audio, sample_rate, codec_rate)
    up = resample_audio(down, codec_rate, sample_rate)
    if up.shape[0] < audio.shape[0]:
        up = np.pad(up, (0, audio.shape[0] - up.shape[0]))
    up = up[: audio.shape[0]]
    quantized = np.round(up * 128.0) / 128.0
    return normalize_audio(quantized.astype(np.float32))


def apply_dropout(audio: np.ndarray, sample_rate: int, rng: random.Random) -> np.ndarray:
    output = audio.copy()
    if output.size == 0:
        return output
    for _ in range(rng.randint(1, 3)):
        length = int(round(rng.uniform(18.0, 70.0) * sample_rate / 1000.0))
        if length <= 0 or length >= output.shape[0]:
            continue
        start = rng.randint(0, output.shape[0] - length)
        output[start : start + length] *= rng.uniform(0.0, 0.25)
    return normalize_audio(output)


def apply_crackle(audio: np.ndarray, rng: random.Random) -> np.ndarray:
    output = audio.copy()
    if output.size == 0:
        return output
    impulse_count = max(1, output.shape[0] // 9000)
    for _ in range(impulse_count):
        index = rng.randrange(output.shape[0])
        output[index] += rng.choice((-1.0, 1.0)) * rng.uniform(0.12, 0.45)
    return normalize_audio(output)


def apply_audio_degradation(
    audio: np.ndarray,
    mode: str,
    sample_rate: int,
    rng: random.Random,
) -> np.ndarray:
    if mode == "none":
        return normalize_audio(audio)
    if mode == "tiny_speaker":
        return apply_tiny_speaker(audio)
    if mode == "clipped_speaker":
        return apply_clipped_speaker(audio)
    if mode == "phone_codec":
        return apply_phone_codec(audio, sample_rate)
    if mode == "dropout":
        return apply_dropout(audio, sample_rate, rng)
    if mode == "crackle":
        return apply_crackle(audio, rng)
    raise ValueError(f"Unsupported audio degradation mode: {mode}")


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
    jitter_snr: bool = True,
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
        mixed_snr_db = rng.uniform(max(-12.0, snr_db - 2.5), snr_db + 3.5) if jitter_snr else snr_db
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
    train_bgm_fraction: float,
    train_degradation_fraction: float,
    train_degradation_modes: list[str],
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
                "degradation": "none",
                "source_path": source_path,
            }
        )

        for index in range(audio_augment_copies):
            augmented_path = cache_root / f"{pokemon_id:03d}_aug_{index:02d}.wav"
            background_mix = None
            if background_paths and rng.random() < max(0.0, min(1.0, train_bgm_fraction)):
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
            degradation = "none"
            if train_degradation_modes and rng.random() < max(0.0, min(1.0, train_degradation_fraction)):
                degradation = rng.choice(train_degradation_modes)
                augmented = apply_audio_degradation(augmented, degradation, sample_rate, rng)
            write_wav(augmented_path, augmented, sample_rate)
            variant = {
                "path": augmented_path,
                "augmentation": augmentation,
                "degradation": degradation,
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


def apply_camera_image_augmentation(
    image: Image.Image,
    rng: random.Random,
    output_size: int,
    canvas_size: int = 256,
) -> Image.Image:
    size = max(canvas_size, output_size)
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

    return composed.resize((output_size, output_size), Image.Resampling.LANCZOS)


def apply_light_image_augmentation(
    image: Image.Image,
    rng: random.Random,
    output_size: int,
) -> Image.Image:
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
    return ImageOps.contain(rgb, (output_size, output_size), Image.Resampling.LANCZOS)


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


def save_png(image: Image.Image, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    image.save(dst, format="PNG", optimize=True, compress_level=9)


def save_original_image(src: Path, dst: Path, output_size: int) -> None:
    image = open_rgba(src)
    canvas = Image.new("RGB", image.size, (255, 255, 255))
    canvas.paste(image, mask=image.getchannel("A"))
    save_png(ImageOps.contain(canvas, (output_size, output_size), Image.Resampling.LANCZOS), dst)


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
    image_source: Path | None,
    audio_source: Path,
    image_augmentation: str,
    audio_augmentation: str,
    audio_storage: str,
    condition: str,
    audio_degradation: str,
    audio_background: Path | None = None,
    audio_mix_offset_sec: float | None = None,
    audio_mix_snr_db: float | None = None,
    source_dataset: str | None = None,
    source_split: str | None = None,
    processing_note: str | None = None,
) -> None:
    audio_meta: dict[str, Any] = {
        "original_path": str(audio_source.as_posix()),
        "augmentation": audio_augmentation,
        "condition": condition,
        "degradation": audio_degradation,
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
        "condition": condition,
        "image": (
            {
                "original_path": str(image_source.as_posix()),
                "augmentation": image_augmentation,
            }
            if image_source is not None
            else None
        ),
        "audio": audio_meta,
    }
    if source_dataset is not None:
        meta["source_dataset"] = source_dataset
    if source_split is not None:
        meta["source_split"] = source_split
    if processing_note is not None:
        meta["processing_note"] = processing_note
    with (sample_dir / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)


def create_sample(
    processed_root: Path,
    split: str,
    class_name: str,
    label_index: int,
    pokemon_id: int,
    image_path: Path | None,
    audio_variant: dict[str, Any],
    sample_id: str,
    rng: random.Random,
    image_augmentation: str,
    camera_augmentation: bool,
    condition: str,
    processed_image_size: int,
    source_dataset: str | None = None,
    source_split: str | None = None,
    processing_note: str | None = None,
) -> None:
    if processed_image_size <= 0:
        raise ValueError("--processed-image-size must be a positive integer.")

    sample_dir = processed_root / split / class_name / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    if image_path is not None:
        image_dst = sample_dir / "image.png"
        if image_augmentation == "origin":
            save_original_image(image_path, image_dst, processed_image_size)
        else:
            image = open_rgba(image_path)
            if camera_augmentation:
                output = apply_camera_image_augmentation(image, rng, processed_image_size)
            else:
                output = apply_light_image_augmentation(image, rng, processed_image_size)
            save_png(output, image_dst)

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
        condition=condition,
        audio_degradation=str(audio_variant.get("degradation", "none")),
        audio_background=(
            Path(audio_variant["background_path"])
            if "background_path" in audio_variant
            else None
        ),
        audio_mix_offset_sec=audio_variant.get("offset_sec"),
        audio_mix_snr_db=audio_variant.get("snr_db"),
        source_dataset=source_dataset,
        source_split=source_split,
        processing_note=processing_note,
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


def split_half_val_test(paths: list[Path], rng: random.Random) -> tuple[list[Path], list[Path]]:
    shuffled = list(paths)
    rng.shuffle(shuffled)
    if len(shuffled) < 2:
        return sorted(shuffled), []
    val_count = len(shuffled) // 2
    return sorted(shuffled[:val_count]), sorted(shuffled[val_count:])


def resolve_second_test_image_root(root: Path) -> Path:
    for child_name in ("test", "Test"):
        child = root / child_name
        if child.exists() and child.is_dir():
            return child
    return root


def create_background_audio_variant(
    raw_root: Path,
    cache_root: Path,
    pokemon_id: int,
    sample_rate: int,
    background_paths: list[Path],
    background_cache: dict[Path, np.ndarray],
    audio_mix_offset_step: float,
    audio_mix_snr_db: float,
    rng: random.Random,
    variant_id: str,
) -> dict[str, Any]:
    if not background_paths:
        raise ValueError("BGM/background audio files are required for validation/test audio.")

    cache_root.mkdir(parents=True, exist_ok=True)
    source_path = raw_root / "cry" / f"{pokemon_id}.ogg"
    if not source_path.exists():
        raise FileNotFoundError(f"Missing Pokemon cry: {source_path}")

    audio, _ = load_audio(source_path, sample_rate)
    background_mix = try_mix_with_background(
        audio=audio,
        background_paths=background_paths,
        background_cache=background_cache,
        sample_rate=sample_rate,
        offset_step_sec=audio_mix_offset_step,
        snr_db=audio_mix_snr_db,
        rng=rng,
    )
    if background_mix is None:
        raise RuntimeError(f"Could not mix BGM with Pokemon cry: {source_path}")

    mixed, details = background_mix
    augmented = augment_audio(mixed, sample_rate, rng)
    safe_variant_id = safe_stem(variant_id)
    augmented_path = cache_root / f"{pokemon_id:03d}_{safe_variant_id}.wav"
    write_wav(augmented_path, augmented, sample_rate)
    variant = {
        "path": augmented_path,
        "augmentation": "background_mix_microphone",
        "source_path": source_path,
    }
    variant.update(details)
    return variant


def hard_audio_condition(condition: str) -> bool:
    return condition in {"audio_hard", "both_hard", "balanced"}


def hard_image_condition(condition: str) -> bool:
    return condition in {"image_hard", "both_hard", "balanced"}


def image_augmentation_for_condition(base_augmentation: str, condition: str, audio_only: bool) -> str:
    if audio_only:
        return "none"
    return base_augmentation if hard_image_condition(condition) else "origin"


def choose_cycled(paths: list[Path], index: int) -> Path | None:
    if not paths:
        return None
    return paths[index % len(paths)]


def image_v2_source_tag(image_path: Path) -> str:
    if "__" not in image_path.name:
        return "unknown"
    return image_path.name.split("__", 1)[0]


def should_augment_image_v2_train(image_path: Path, augment_prefix: str) -> bool:
    return bool(augment_prefix) and image_path.name.startswith(augment_prefix)


def load_source_audio_cached(
    raw_root: Path,
    pokemon_id: int,
    sample_rate: int,
    source_audio_cache: dict[int, np.ndarray],
) -> tuple[np.ndarray, Path]:
    source_path = raw_root / "cry" / f"{pokemon_id}.ogg"
    if not source_path.exists():
        raise FileNotFoundError(f"Missing Pokemon cry: {source_path}")
    if pokemon_id not in source_audio_cache:
        source_audio_cache[pokemon_id], _ = load_audio(source_path, sample_rate)
    return source_audio_cache[pokemon_id], source_path


def create_condition_audio_variant(
    raw_root: Path,
    cache_root: Path,
    pokemon_id: int,
    sample_rate: int,
    condition: str,
    snr_values: list[float],
    degradation_modes: list[str],
    background_paths: list[Path],
    background_cache: dict[Path, np.ndarray],
    source_audio_cache: dict[int, np.ndarray],
    audio_mix_offset_step: float,
    rng: random.Random,
    variant_id: str,
    index: int,
) -> dict[str, Any]:
    cache_root.mkdir(parents=True, exist_ok=True)
    source_audio, source_path = load_source_audio_cached(raw_root, pokemon_id, sample_rate, source_audio_cache)
    audio = source_audio.copy()
    details: dict[str, Any] = {}
    augmentation = "origin"
    degradation = "none"

    if hard_audio_condition(condition):
        snr_db = snr_values[index % len(snr_values)] if snr_values else 6.0
        background_mix = try_mix_with_background(
            audio=source_audio,
            background_paths=background_paths,
            background_cache=background_cache,
            sample_rate=sample_rate,
            offset_step_sec=audio_mix_offset_step,
            snr_db=snr_db,
            rng=rng,
            jitter_snr=False,
        )
        if background_mix is not None:
            audio, details = background_mix
            augmentation = "background_mix"
        else:
            augmentation = "degraded_origin"
            details["snr_db"] = "unmixed"

        degradation = degradation_modes[(index // max(1, len(snr_values))) % len(degradation_modes)]
        audio = apply_audio_degradation(audio, degradation, sample_rate, rng)

    safe_variant_id = safe_stem(variant_id)
    output_path = cache_root / f"{pokemon_id:03d}_{safe_variant_id}.wav"
    write_wav(output_path, audio, sample_rate)
    variant = {
        "path": output_path,
        "augmentation": augmentation,
        "condition": condition,
        "degradation": degradation,
        "source_path": source_path,
    }
    variant.update(details)
    return variant


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
    return build_processed_dataset_robust(args)

    raw_root = args.raw_root.expanduser().resolve()
    second_train_root = args.second_train_root.expanduser().resolve()
    second_test_root = resolve_second_test_image_root(args.second_test_root.expanduser().resolve())
    processed_root = args.processed_root.expanduser().resolve()
    image_root = raw_root / "Image"

    if not raw_root.exists():
        raise FileNotFoundError(f"Raw root not found: {raw_root}")
    if not second_train_root.exists():
        raise FileNotFoundError(f"Second shared train root not found: {second_train_root}")
    if not second_test_root.exists():
        raise FileNotFoundError(f"Second shared test root not found: {second_test_root}")
    if not (image_root / "train").exists():
        raise FileNotFoundError(f"Missing train image directory: {image_root / 'train'}")
    if not (raw_root / "cry").exists():
        raise FileNotFoundError(f"Missing cry directory: {raw_root / 'cry'}")

    rng = random.Random(args.seed)
    pokemon_map = read_pokemon_map(raw_root)
    classes: list[tuple[int, str]] = []
    for pokemon_id, class_name in sorted(pokemon_map.items()):
        has_cry = (raw_root / "cry" / f"{pokemon_id}.ogg").exists()
        has_first_train = bool(collect_image_paths(image_root / "train", class_name))
        has_second_train = bool(collect_image_paths(second_train_root, class_name))
        has_second_test = bool(collect_image_paths(second_test_root, class_name))
        if has_cry and (has_first_train or has_second_train or has_second_test):
            classes.append((pokemon_id, class_name))

    if not classes:
        raise ValueError(
            "No usable Pokemon classes found with first shared train, second shared images, and cries."
        )

    ensure_clean_directory(processed_root)
    for split in ("train", "val", "test"):
        (processed_root / split).mkdir(parents=True, exist_ok=True)

    label_indices = write_labels_json(processed_root, classes)
    background_root = resolve_audio_background_root(raw_root, args.audio_background_root)
    background_paths = collect_audio_paths(background_root)
    if not background_paths:
        raise ValueError(f"No BGM/background audio files found under: {background_root}")

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
    eval_audio_cache_root = processed_root / "_audio_cache_bgm"
    eval_background_cache: dict[Path, np.ndarray] = {}

    summary: Counter = Counter()
    camera_augmentation = not args.disable_camera_augmentation
    summary["audio_background_files"] = len(background_paths)
    summary["classes"] = len(classes)

    for pokemon_id, class_name in classes:
        label_index = label_indices[pokemon_id]
        first_train_paths = collect_image_paths(image_root / "train", class_name)
        second_train_paths = collect_image_paths(second_train_root, class_name)
        second_test_paths = collect_image_paths(second_test_root, class_name)
        val_paths, test_paths = split_half_val_test(second_test_paths, rng)
        train_audio_variants = train_audio_cache.get(pokemon_id, [])
        if not train_audio_variants:
            continue

        summary["first_shared_train_images"] += len(first_train_paths)
        summary["second_shared_train_images"] += len(second_train_paths)
        summary["second_shared_test_images"] += len(second_test_paths)
        summary["second_shared_val_split_images"] += len(val_paths)
        summary["second_shared_test_split_images"] += len(test_paths)

        image_augmentation = "camera" if camera_augmentation else "light"
        first_train_records = build_augmented_records(
            first_train_paths,
            max(0, args.target_train_count),
            image_augmentation,
            rng,
        )

        for index, (image_path, record_image_augmentation) in enumerate(first_train_records):
            digest = stable_digest(
                f"train:first:{pokemon_id}:{image_path}:{index}:{record_image_augmentation}"
            )
            sample_id = f"{pokemon_id:03d}_first_{safe_stem(image_path.stem)}_{index:04d}_{digest}"
            audio_variant = select_augmented_audio_variant(train_audio_variants, index)
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
                image_augmentation=record_image_augmentation,
                camera_augmentation=camera_augmentation,
                condition="train",
                processed_image_size=args.processed_image_size,
                source_dataset="1차 공유본",
                source_split="Image/train",
                processing_note=(
                    "first shared train sample: image expanded with deterministic augmentation; "
                    "audio uses microphone-style augmented cry variants"
                ),
            )
            summary["train_samples"] += 1
            summary["first_shared_augmented_train_samples"] += 1
            summary[f"train_image_augmentation_{record_image_augmentation}"] += 1
            summary[f"train_audio_augmentation_{audio_variant['augmentation']}"] += 1

        for index, image_path in enumerate(second_train_paths):
            digest = stable_digest(f"train:second:{pokemon_id}:{image_path}:{index}")
            sample_id = f"{pokemon_id:03d}_second_train_{safe_stem(image_path.stem)}_{index:05d}_{digest}"
            audio_variant = select_augmented_audio_variant(train_audio_variants, index)
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
                condition="train",
                processed_image_size=args.processed_image_size,
                source_dataset="2차 공유본 Train",
                source_split=class_name.lower(),
                processing_note=(
                    "second shared train sample: image copied/resized only; "
                    "audio uses microphone-style augmented cry variants"
                ),
            )
            summary["train_samples"] += 1
            summary["second_shared_train_samples"] += 1
            summary["train_image_augmentation_origin"] += 1
            summary[f"train_audio_augmentation_{audio_variant['augmentation']}"] += 1

        eval_split_paths = {
            "val": val_paths,
            "test": test_paths,
        }
        for split, paths in eval_split_paths.items():
            for index, image_path in enumerate(paths):
                digest = stable_digest(f"{split}:second_test:{pokemon_id}:{image_path}:{index}")
                sample_id = f"{pokemon_id:03d}_second_test_{safe_stem(image_path.stem)}_{index:05d}_{digest}"
                audio_variant = create_background_audio_variant(
                    raw_root=raw_root,
                    cache_root=eval_audio_cache_root,
                    pokemon_id=pokemon_id,
                    sample_rate=args.sample_rate,
                    background_paths=background_paths,
                    background_cache=eval_background_cache,
                    audio_mix_offset_step=args.audio_mix_offset_step,
                    audio_mix_snr_db=args.audio_mix_snr_db,
                    rng=rng,
                    variant_id=f"{split}_{sample_id}",
                )
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
                    image_augmentation="origin",
                    camera_augmentation=False,
                    condition=split,
                    processed_image_size=args.processed_image_size,
                    source_dataset="2차 공유본 Test",
                    source_split=class_name.lower(),
                    processing_note=(
                        "second shared test sample: image copied/resized only; "
                        "audio uses BGM-mixed augmented cry variants from the first shared audio"
                    ),
                )
                summary[f"{split}_samples"] += 1
                summary[f"second_shared_{split}_samples"] += 1
                summary[f"{split}_image_augmentation_origin"] += 1
                summary[f"{split}_audio_augmentation_{audio_variant['augmentation']}"] += 1

    summary_path = processed_root / "dataset_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(dict(sorted(summary.items())), handle, ensure_ascii=False, indent=2)

    for cache_name in ("_audio_cache", "_audio_cache_bgm"):
        cache_root = processed_root / cache_name
        if cache_root.exists():
            shutil.rmtree(cache_root)

    return summary


def build_processed_dataset_robust(args: argparse.Namespace) -> Counter:
    raw_root = args.raw_root.expanduser().resolve()
    processed_root = args.processed_root.expanduser().resolve()
    image_root = raw_root / "Image"
    second_train_root = args.second_train_root.expanduser().resolve()
    second_test_root = resolve_second_test_image_root(args.second_test_root.expanduser().resolve())

    if not raw_root.exists():
        raise FileNotFoundError(f"Raw root not found: {raw_root}")
    if not (raw_root / "cry").exists():
        raise FileNotFoundError(f"Missing cry directory: {raw_root / 'cry'}")
    if not args.audio_only and not (image_root / "train").exists():
        raise FileNotFoundError(f"Missing train image directory: {image_root / 'train'}")

    rng = random.Random(args.seed)
    train_degradation_modes = validate_degradation_sweep(
        "train",
        parse_string_sweep(args.train_degradation_modes, ["tiny_speaker", "phone_codec"]),
    )
    val_degradation_modes = validate_degradation_sweep(
        "val",
        parse_string_sweep(args.val_degradation_sweep or args.val_degradation, [args.val_degradation]),
    )
    test_degradation_modes = validate_degradation_sweep(
        "test",
        parse_string_sweep(args.test_degradation_sweep, ["none", "tiny_speaker", "phone_codec"]),
    )
    val_snr_values = parse_float_sweep(args.val_snr_sweep, [float(args.val_snr_db)])
    test_snr_values = parse_float_sweep(args.test_snr_sweep, [12.0, 6.0, 0.0, -6.0])
    val_conditions = validate_condition_sweep(
        "val",
        parse_string_sweep(args.val_condition_sweep, ["clean_sanity", "audio_hard", "balanced"]),
    )
    test_conditions = validate_condition_sweep(
        "test",
        parse_string_sweep(
            args.test_condition_sweep,
            ["clean_sanity", "audio_hard", "image_hard", "both_hard", "balanced"],
        ),
    )

    pokemon_map = read_pokemon_map(raw_root)
    classes: list[tuple[int, str]] = []
    for pokemon_id, class_name in sorted(pokemon_map.items()):
        has_cry = (raw_root / "cry" / f"{pokemon_id}.ogg").exists()
        if args.audio_only:
            has_source = has_cry
        else:
            image_paths: list[Path] = []
            if (image_root / "train").exists():
                image_paths.extend(collect_image_paths(image_root / "train", class_name))
            if (image_root / "test").exists():
                image_paths.extend(collect_image_paths(image_root / "test", class_name))
            if second_train_root.exists():
                image_paths.extend(collect_image_paths(second_train_root, class_name))
            if second_test_root.exists():
                image_paths.extend(collect_image_paths(second_test_root, class_name))
            has_source = has_cry and bool(image_paths)
        if has_source:
            classes.append((pokemon_id, class_name))

    if not classes:
        raise ValueError(
            "No usable Pokemon classes found with cries"
            + ("." if args.audio_only else " and images.")
        )

    ensure_clean_directory(processed_root)
    for split in ("train", "val", "test"):
        (processed_root / split).mkdir(parents=True, exist_ok=True)

    label_indices = write_labels_json(processed_root, classes)
    background_root = resolve_audio_background_root(raw_root, args.audio_background_root)
    background_paths = collect_audio_paths(background_root)
    hard_eval_requested = any(hard_audio_condition(condition) for condition in val_conditions + test_conditions)
    if hard_eval_requested and not background_paths:
        raise ValueError(
            "BGM/background audio files are required for hard audio evaluation conditions. "
            f"Resolved background root: {background_root}"
        )
    train_background_paths, val_background_paths, test_background_paths = partition_background_tracks(
        background_paths=background_paths,
        train_fraction=args.train_bgm_fraction,
        val_fraction=0.5,
        train_enabled=bool(args.train_mix_bgm),
        rng=rng,
    )

    train_audio_cache = prepare_audio_cache(
        raw_root=raw_root,
        processed_root=processed_root,
        pokemon_ids=[pokemon_id for pokemon_id, _ in classes],
        sample_rate=args.sample_rate,
        audio_augment_copies=max(0, args.audio_augment_copies),
        background_paths=train_background_paths if args.train_mix_bgm else [],
        audio_mix_offset_step=args.audio_mix_offset_step,
        audio_mix_snr_db=args.audio_mix_snr_db,
        train_bgm_fraction=args.train_bgm_fraction,
        train_degradation_fraction=args.train_degradation_fraction,
        train_degradation_modes=train_degradation_modes,
        rng=rng,
    )

    eval_audio_cache_root = processed_root / "_audio_cache_eval"
    val_background_cache: dict[Path, np.ndarray] = {}
    test_background_cache: dict[Path, np.ndarray] = {}
    source_audio_cache: dict[int, np.ndarray] = {}

    summary: Counter = Counter()
    camera_augmentation = not args.disable_camera_augmentation
    base_image_augmentation = "camera" if camera_augmentation else "light"
    summary["classes"] = len(classes)
    summary["audio_background_files"] = len(background_paths)
    summary["train_background_files"] = len(train_background_paths)
    summary["val_background_files"] = len(val_background_paths)
    summary["test_background_files"] = len(test_background_paths)

    for pokemon_id, class_name in classes:
        label_index = label_indices[pokemon_id]
        train_audio_variants = train_audio_cache.get(pokemon_id, [])
        if not train_audio_variants:
            continue

        first_train_paths = (
            collect_image_paths(image_root / "train", class_name)
            if not args.audio_only and (image_root / "train").exists()
            else []
        )
        raw_test_paths = (
            collect_image_paths(image_root / "test", class_name)
            if not args.audio_only and (image_root / "test").exists()
            else []
        )
        second_train_paths = (
            collect_image_paths(second_train_root, class_name)
            if not args.audio_only and second_train_root.exists()
            else []
        )
        second_test_paths = (
            collect_image_paths(second_test_root, class_name)
            if not args.audio_only and second_test_root.exists()
            else []
        )
        train_image_pool = first_train_paths + second_train_paths
        eval_image_pool = second_test_paths + raw_test_paths + first_train_paths + second_train_paths
        if not args.audio_only and (not train_image_pool or not eval_image_pool):
            continue

        summary["first_shared_train_images"] += len(first_train_paths)
        summary["second_shared_train_images"] += len(second_train_paths)
        summary["raw_test_images"] += len(raw_test_paths)
        summary["second_shared_test_images"] += len(second_test_paths)

        for index in range(max(0, args.target_train_count)):
            image_path = None if args.audio_only else choose_cycled(train_image_pool, index)
            image_augmentation = "none" if args.audio_only else base_image_augmentation
            image_stem = "audio_only" if image_path is None else safe_stem(image_path.stem)
            digest = stable_digest(f"train:{pokemon_id}:{image_path}:{index}:{image_augmentation}")
            sample_id = f"{pokemon_id:03d}_train_{image_stem}_{index:04d}_{digest}"
            audio_variant = select_augmented_audio_variant(train_audio_variants, index)
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
                condition="train",
                processed_image_size=args.processed_image_size,
                source_dataset="audio_only" if args.audio_only else "train_image_pool",
                source_split="train",
                processing_note=(
                    "train sample: audio uses deterministic recording/BGM/degradation variants; "
                    "image uses camera-style augmentation when available"
                ),
            )
            summary["train_samples"] += 1
            summary[f"train_image_augmentation_{image_augmentation}"] += 1
            summary[f"train_audio_augmentation_{audio_variant['augmentation']}"] += 1
            summary[f"train_degradation_{audio_variant.get('degradation', 'none')}"] += 1
            if args.audio_only:
                summary["audio_only_samples"] += 1

        eval_specs = (
            (
                "val",
                max(0, args.target_val_count),
                val_conditions,
                val_snr_values,
                val_degradation_modes,
                val_background_paths,
                val_background_cache,
            ),
            (
                "test",
                max(0, args.target_test_count),
                test_conditions,
                test_snr_values,
                test_degradation_modes,
                test_background_paths,
                test_background_cache,
            ),
        )
        for split, target_count, conditions, snr_values, degradation_modes, split_background_paths, split_background_cache in eval_specs:
            for index in range(target_count):
                condition = conditions[index % len(conditions)]
                image_path = None if args.audio_only else choose_cycled(eval_image_pool, index)
                image_augmentation = image_augmentation_for_condition(
                    base_image_augmentation,
                    condition,
                    args.audio_only,
                )
                image_stem = "audio_only" if image_path is None else safe_stem(image_path.stem)
                digest = stable_digest(f"{split}:{condition}:{pokemon_id}:{image_path}:{index}")
                sample_id = f"{pokemon_id:03d}_{split}_{condition}_{image_stem}_{index:04d}_{digest}"
                audio_variant = create_condition_audio_variant(
                    raw_root=raw_root,
                    cache_root=eval_audio_cache_root,
                    pokemon_id=pokemon_id,
                    sample_rate=args.sample_rate,
                    condition=condition,
                    snr_values=snr_values,
                    degradation_modes=degradation_modes,
                    background_paths=split_background_paths,
                    background_cache=split_background_cache,
                    source_audio_cache=source_audio_cache,
                    audio_mix_offset_step=args.audio_mix_offset_step,
                    rng=rng,
                    variant_id=sample_id,
                    index=index,
                )
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
                    image_augmentation=image_augmentation,
                    camera_augmentation=camera_augmentation,
                    condition=condition,
                    processed_image_size=args.processed_image_size,
                    source_dataset="audio_only" if args.audio_only else "eval_image_pool",
                    source_split=split,
                    processing_note=(
                        "evaluation sample: condition defines controlled robustness corruption; "
                        "metrics should be read as degradation robustness, not new-cry generalization"
                    ),
                )
                summary[f"{split}_samples"] += 1
                summary[f"{split}_condition_{condition}"] += 1
                summary[f"{split}_image_augmentation_{image_augmentation}"] += 1
                summary[f"{split}_audio_augmentation_{audio_variant['augmentation']}"] += 1
                summary[f"{split}_degradation_{audio_variant.get('degradation', 'none')}"] += 1
                summary[f"{split}_snr_db_{audio_variant.get('snr_db', 'clean')}"] += 1
                if args.audio_only:
                    summary["audio_only_samples"] += 1

    summary_path = processed_root / "dataset_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(dict(sorted(summary.items())), handle, ensure_ascii=False, indent=2)

    for cache_name in ("_audio_cache", "_audio_cache_eval"):
        cache_root = processed_root / cache_name
        if cache_root.exists():
            shutil.rmtree(cache_root)

    return summary


def build_processed_dataset_image_v2(args: argparse.Namespace) -> Counter:
    raw_root = args.raw_root.expanduser().resolve()
    processed_root = args.processed_root.expanduser().resolve()
    image_root = raw_root / "Image"
    train_v2_root = image_root / "train_v2"
    val_v2_root = image_root / "val_v2"
    test_v2_root = image_root / "test_v2"

    if not raw_root.exists():
        raise FileNotFoundError(f"Raw root not found: {raw_root}")
    if not (raw_root / "cry").exists():
        raise FileNotFoundError(f"Missing cry directory: {raw_root / 'cry'}")
    for split_root in (train_v2_root, val_v2_root, test_v2_root):
        if not split_root.exists():
            raise FileNotFoundError(f"Missing image v2 split directory: {split_root}")

    rng = random.Random(args.seed)
    train_degradation_modes = validate_degradation_sweep(
        "train",
        parse_string_sweep(args.train_degradation_modes, ["tiny_speaker", "phone_codec"]),
    )
    val_degradation_modes = validate_degradation_sweep(
        "val",
        parse_string_sweep(args.val_degradation_sweep or args.val_degradation, [args.val_degradation]),
    )
    test_degradation_modes = validate_degradation_sweep(
        "test",
        parse_string_sweep(args.test_degradation_sweep, ["none", "tiny_speaker", "phone_codec"]),
    )
    val_snr_values = parse_float_sweep(args.val_snr_sweep, [float(args.val_snr_db)])
    test_snr_values = parse_float_sweep(args.test_snr_sweep, [12.0, 6.0, 0.0, -6.0])
    val_conditions = validate_condition_sweep(
        "val",
        parse_string_sweep(args.val_condition_sweep, ["clean_sanity", "audio_hard", "balanced"]),
    )
    test_conditions = validate_condition_sweep(
        "test",
        parse_string_sweep(args.test_condition_sweep, ["clean_sanity", "audio_hard", "balanced"]),
    )
    if not args.image_v2_eval_image_augmentation:
        hard_eval_conditions = [
            condition
            for condition in val_conditions + test_conditions
            if condition in {"image_hard", "both_hard"}
        ]
        if hard_eval_conditions:
            raise ValueError(
                "image_hard/both_hard conditions require --image-v2-eval-image-augmentation "
                "in --use-image-v2 mode. Otherwise use audio-only style conditions such as "
                "clean_sanity,audio_hard,balanced."
            )

    pokemon_map = read_pokemon_map(raw_root)
    classes: list[tuple[int, str]] = []
    for pokemon_id, class_name in sorted(pokemon_map.items()):
        has_cry = (raw_root / "cry" / f"{pokemon_id}.ogg").exists()
        has_train = bool(collect_image_paths(train_v2_root, class_name))
        has_val = bool(collect_image_paths(val_v2_root, class_name))
        has_test = bool(collect_image_paths(test_v2_root, class_name))
        if has_cry and has_train and has_val and has_test:
            classes.append((pokemon_id, class_name))

    if not classes:
        raise ValueError("No usable Pokemon classes found with v2 image splits and cries.")

    ensure_clean_directory(processed_root)
    for split in ("train", "val", "test"):
        (processed_root / split).mkdir(parents=True, exist_ok=True)

    label_indices = write_labels_json(processed_root, classes)
    background_root = resolve_audio_background_root(raw_root, args.audio_background_root)
    background_paths = collect_audio_paths(background_root)
    hard_eval_requested = any(hard_audio_condition(condition) for condition in val_conditions + test_conditions)
    if hard_eval_requested and not background_paths:
        raise ValueError(
            "BGM/background audio files are required for hard audio evaluation conditions. "
            f"Resolved background root: {background_root}"
        )
    train_background_paths, val_background_paths, test_background_paths = partition_background_tracks(
        background_paths=background_paths,
        train_fraction=args.train_bgm_fraction,
        val_fraction=0.5,
        train_enabled=bool(args.train_mix_bgm),
        rng=rng,
    )

    train_audio_cache = prepare_audio_cache(
        raw_root=raw_root,
        processed_root=processed_root,
        pokemon_ids=[pokemon_id for pokemon_id, _ in classes],
        sample_rate=args.sample_rate,
        audio_augment_copies=max(0, args.audio_augment_copies),
        background_paths=train_background_paths if args.train_mix_bgm else [],
        audio_mix_offset_step=args.audio_mix_offset_step,
        audio_mix_snr_db=args.audio_mix_snr_db,
        train_bgm_fraction=args.train_bgm_fraction,
        train_degradation_fraction=args.train_degradation_fraction,
        train_degradation_modes=train_degradation_modes,
        rng=rng,
    )

    eval_audio_cache_root = processed_root / "_audio_cache_eval"
    val_background_cache: dict[Path, np.ndarray] = {}
    test_background_cache: dict[Path, np.ndarray] = {}
    source_audio_cache: dict[int, np.ndarray] = {}

    summary: Counter = Counter()
    camera_augmentation = not args.disable_camera_augmentation
    base_image_augmentation = "camera" if camera_augmentation else "light"
    extra_augment_copies = max(0, int(args.image_v2_extra_augment_copies))
    summary["classes"] = len(classes)
    summary["image_v2_mode"] = 1
    summary["image_v2_extra_augment_copies"] = extra_augment_copies
    summary["audio_background_files"] = len(background_paths)
    summary["train_background_files"] = len(train_background_paths)
    summary["val_background_files"] = len(val_background_paths)
    summary["test_background_files"] = len(test_background_paths)

    for pokemon_id, class_name in classes:
        label_index = label_indices[pokemon_id]
        train_audio_variants = train_audio_cache.get(pokemon_id, [])
        if not train_audio_variants:
            continue

        train_paths = collect_image_paths(train_v2_root, class_name)
        val_paths = collect_image_paths(val_v2_root, class_name)
        test_paths = collect_image_paths(test_v2_root, class_name)
        summary["image_v2_train_source_images"] += len(train_paths)
        summary["image_v2_val_source_images"] += len(val_paths)
        summary["image_v2_test_source_images"] += len(test_paths)

        train_sample_index = 0
        for index, image_path in enumerate(train_paths):
            source_tag = image_v2_source_tag(image_path)
            image_augmentation = (
                base_image_augmentation
                if should_augment_image_v2_train(image_path, args.image_v2_augment_prefix)
                else "origin"
            )
            digest = stable_digest(f"image_v2:train:{pokemon_id}:{image_path}:{index}:{image_augmentation}")
            sample_id = f"{pokemon_id:03d}_train_v2_{safe_stem(image_path.stem)}_{index:05d}_{digest}"
            audio_variant = select_augmented_audio_variant(train_audio_variants, train_sample_index)
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
                condition="train",
                processed_image_size=args.processed_image_size,
                source_dataset=f"image_v2:{source_tag}",
                source_split="train_v2",
                processing_note=(
                    "image v2 train sample: only configured source-prefix images receive "
                    "image augmentation; all other train images are copied/resized"
                ),
            )
            train_sample_index += 1
            summary["train_samples"] += 1
            summary[f"train_image_augmentation_{image_augmentation}"] += 1
            summary[f"train_image_source_{source_tag}"] += 1
            summary[f"train_audio_augmentation_{audio_variant['augmentation']}"] += 1
            summary[f"train_degradation_{audio_variant.get('degradation', 'none')}"] += 1

            if should_augment_image_v2_train(image_path, args.image_v2_augment_prefix):
                for copy_index in range(extra_augment_copies):
                    digest = stable_digest(
                        f"image_v2:train:extra:{pokemon_id}:{image_path}:{index}:{copy_index}:{base_image_augmentation}"
                    )
                    sample_id = (
                        f"{pokemon_id:03d}_train_v2_aug_{safe_stem(image_path.stem)}_"
                        f"{index:05d}_{copy_index:02d}_{digest}"
                    )
                    audio_variant = select_augmented_audio_variant(train_audio_variants, train_sample_index)
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
                        image_augmentation=base_image_augmentation,
                        camera_augmentation=camera_augmentation,
                        condition="train",
                        processed_image_size=args.processed_image_size,
                        source_dataset=f"image_v2:{source_tag}",
                        source_split="train_v2",
                        processing_note="additional image v2 train augmentation copy",
                    )
                    train_sample_index += 1
                    summary["train_samples"] += 1
                    summary[f"train_image_augmentation_{base_image_augmentation}"] += 1
                    summary[f"train_image_source_{source_tag}"] += 1
                    summary[f"train_audio_augmentation_{audio_variant['augmentation']}"] += 1
                    summary[f"train_degradation_{audio_variant.get('degradation', 'none')}"] += 1
                    summary["train_image_v2_extra_augmented_copies"] += 1

        eval_specs = (
            (
                "val",
                "val_v2",
                val_paths,
                val_conditions,
                val_snr_values,
                val_degradation_modes,
                val_background_paths,
                val_background_cache,
            ),
            (
                "test",
                "test_v2",
                test_paths,
                test_conditions,
                test_snr_values,
                test_degradation_modes,
                test_background_paths,
                test_background_cache,
            ),
        )
        for split, source_split, paths, conditions, snr_values, degradation_modes, split_background_paths, split_background_cache in eval_specs:
            for index, image_path in enumerate(paths):
                source_tag = image_v2_source_tag(image_path)
                condition = conditions[index % len(conditions)]
                image_augmentation = (
                    image_augmentation_for_condition(base_image_augmentation, condition, False)
                    if args.image_v2_eval_image_augmentation
                    else "origin"
                )
                digest = stable_digest(f"image_v2:{split}:{condition}:{pokemon_id}:{image_path}:{index}")
                sample_id = f"{pokemon_id:03d}_{source_split}_{condition}_{safe_stem(image_path.stem)}_{index:05d}_{digest}"
                audio_variant = create_condition_audio_variant(
                    raw_root=raw_root,
                    cache_root=eval_audio_cache_root,
                    pokemon_id=pokemon_id,
                    sample_rate=args.sample_rate,
                    condition=condition,
                    snr_values=snr_values,
                    degradation_modes=degradation_modes,
                    background_paths=split_background_paths,
                    background_cache=split_background_cache,
                    source_audio_cache=source_audio_cache,
                    audio_mix_offset_step=args.audio_mix_offset_step,
                    rng=rng,
                    variant_id=sample_id,
                    index=index,
                )
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
                    image_augmentation=image_augmentation,
                    camera_augmentation=camera_augmentation,
                    condition=condition,
                    processed_image_size=args.processed_image_size,
                    source_dataset=f"image_v2:{source_tag}",
                    source_split=source_split,
                    processing_note=(
                        "image v2 evaluation sample: validation/test images remain origin "
                        "unless --image-v2-eval-image-augmentation is enabled"
                    ),
                )
                summary[f"{split}_samples"] += 1
                summary[f"{split}_condition_{condition}"] += 1
                summary[f"{split}_image_augmentation_{image_augmentation}"] += 1
                summary[f"{split}_image_source_{source_tag}"] += 1
                summary[f"{split}_audio_augmentation_{audio_variant['augmentation']}"] += 1
                summary[f"{split}_degradation_{audio_variant.get('degradation', 'none')}"] += 1
                summary[f"{split}_snr_db_{audio_variant.get('snr_db', 'clean')}"] += 1

    summary_path = processed_root / "dataset_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(dict(sorted(summary.items())), handle, ensure_ascii=False, indent=2)

    for cache_name in ("_audio_cache", "_audio_cache_eval"):
        cache_root = processed_root / cache_name
        if cache_root.exists():
            shutil.rmtree(cache_root)

    return summary


def main() -> None:
    args = parse_args()
    if args.use_image_v2 and not args.audio_only:
        summary = build_processed_dataset_image_v2(args)
    else:
        summary = build_processed_dataset_robust(args)
    print(f"Processed dataset written to: {args.processed_root}")
    for key in sorted(summary):
        print(f"{key}: {summary[key]}")


if __name__ == "__main__":
    main()
