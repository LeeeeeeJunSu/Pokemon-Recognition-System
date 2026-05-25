import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aiff", ".aif"}


def print_progress(current: int, total: int, label: str = ""):
    """Print a single-line percentage progress update."""
    pct = (current / total * 100) if total else 100.0
    prefix = f"[{label}] " if label else ""
    print(f"\r{prefix}{current}/{total} ({pct:5.1f}%)", end="", flush=True)
    if current >= total:
        print()                              # newline when done


def load_audio(path: Path, target_sr: int) -> np.ndarray:
    """Load audio, downmix to mono, resample to target_sr, return float32."""
    data, sr = sf.read(str(path), always_2d=False)
    if data.ndim > 1:                       # stereo / multichannel -> mono
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    if sr != target_sr:
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
    return data


def mix_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Mix `signal` (animal) and `noise` (background) so the resulting
    signal-to-noise ratio equals `snr_db`. Peak-normalizes if clipping.
    """
    sig_rms = np.sqrt(np.mean(signal ** 2) + 1e-12)
    noise_rms = np.sqrt(np.mean(noise ** 2) + 1e-12)
    target_noise_rms = sig_rms / (10 ** (snr_db / 20))
    scale = target_noise_rms / noise_rms
    mixed = signal + scale * noise

    peak = np.max(np.abs(mixed))
    if peak > 1.0:                          # avoid clipping
        mixed = mixed / peak
    return mixed


def collect_audio_files(root: Path) -> list:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXTS)


def augment(
    cry_dir: Path,
    bgm_dir: Path,
    out_dir: Path,
    offset_step_sec: float = 3.0,
    snr_db: float = 10.0,
    sample_rate: int = 22050,
    output_ext: str = ".wav",
    cache_backgrounds: bool = True,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    a_files = collect_audio_files(cry_dir)
    b_files = collect_audio_files(bgm_dir)

    if not a_files:
        raise RuntimeError(f"No audio files found under {cry_dir}")
    if not b_files:
        raise RuntimeError(f"No audio files found under {bgm_dir}")

    print(f"Animal sounds : {len(a_files)}")
    print(f"Backgrounds   : {len(b_files)}")
    print(f"Pairs to scan : {len(a_files) * len(b_files)}")

    # Pre-load backgrounds once. Each B will be reused across all A files,
    # so caching saves a lot of I/O. If your B set is huge (many GB), set
    # cache_backgrounds=False to reload on demand instead.
    bg_cache = {}
    if cache_backgrounds:
        print("Pre-loading backgrounds into memory...")
        total_bg = len(b_files)
        for i, b_path in enumerate(b_files, start=1):
            try:
                bg_cache[b_path] = load_audio(b_path, sample_rate)
            except Exception as e:
                print(f"\n  [skip] {b_path.name}: {e}")
            print_progress(i, total_bg, label="bgm load")

    offset_step_samples = int(round(offset_step_sec * sample_rate))
    total_written = 0
    total_skipped_pairs = 0

    total_a = len(a_files)
    for a_idx, a_path in enumerate(a_files, start=1):
        try:
            a_data = load_audio(a_path, sample_rate)
        except Exception as e:
            print(f"\n[skip] {a_path.name}: {e}")
            print_progress(a_idx, total_a, label="cry")
            continue

        a_len = len(a_data)

        for b_path in b_files:
            if cache_backgrounds:
                b_data = bg_cache.get(b_path)
                if b_data is None:
                    continue
            else:
                try:
                    b_data = load_audio(b_path, sample_rate)
                except Exception:
                    continue

            b_len = len(b_data)
            if a_len > b_len:               # even offset 0 doesn't fit
                total_skipped_pairs += 1
                continue

            idx = 0
            while True:
                start = idx * offset_step_samples
                end = start + a_len
                if end > b_len:             # offset out of range -> stop sliding
                    break

                b_segment = b_data[start:end]
                mixed = mix_at_snr(a_data, b_segment, snr_db)

                offset_sec = idx * offset_step_sec
                out_name = (
                    f"{a_path.stem}__{b_path.stem}"
                    f"__off{offset_sec:05.1f}s{output_ext}"
                )
                sf.write(str(out_dir / out_name), mixed, sample_rate)
                total_written += 1
                idx += 1

        print_progress(a_idx, total_a, label="cry")

    print()
    print(f"Done.")
    print(f"  Wrote          : {total_written} files -> {out_dir}")
    print(f"  Skipped pairs  : {total_skipped_pairs} (A longer than B)")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cry_dir", required=True, type=Path,
                        help="Directory containing animal sound files (path A)")
    parser.add_argument("--bgm_dir", required=True, type=Path,
                        help="Directory containing background sound files (path B)")
    parser.add_argument("--out_dir", required=True, type=Path,
                        help="Output directory for augmented files")
    parser.add_argument("--offset_step", type=float, default=3.0,
                        help="Offset step in seconds (default: 3.0)")
    parser.add_argument("--snr_db", type=float, default=10.0,
                        help="SNR of animal vs background, dB (default: 10.0)")
    parser.add_argument("--sample_rate", type=int, default=22050,
                        help="Output sample rate (default: 22050)")
    parser.add_argument("--output_ext", type=str, default=".wav",
                        choices=[".wav", ".flac", ".ogg"],
                        help="Output file extension (default: .wav)")
    parser.add_argument("--no_cache_bg", action="store_true",
                        help="Don't pre-load backgrounds into memory "
                             "(slower but uses much less RAM)")
    args = parser.parse_args()

    augment(
        cry_dir=args.cry_dir,
        bgm_dir=args.bgm_dir,
        out_dir=args.out_dir,
        offset_step_sec=args.offset_step,
        snr_db=args.snr_db,
        sample_rate=args.sample_rate,
        output_ext=args.output_ext,
        cache_backgrounds=not args.no_cache_bg,
    )


if __name__ == "__main__":
    main()