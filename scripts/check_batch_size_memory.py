from __future__ import annotations

import argparse
import csv
import gc
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MODEL_SPECS = {
    "ImageModel": ("Engine.ImageModel", "ImageModel", "image"),
    "AudioModel": ("Engine.AudioModel", "AudioModel", "audio"),
    "LateFusionModel": ("Engine.LateFusionModel", "LateFusionModel", "multimodal"),
    "MidFusionModel": ("Engine.MidFusionModel", "MidFusionModel", "multimodal"),
    "ScoreFusionModel": ("Engine.ScoreFusionModel", "ScoreFusionModel", "multimodal"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe train-step GPU memory usage for project models at multiple batch sizes. "
            "This uses random tensors, so it does not read the dataset."
        )
    )
    parser.add_argument(
        "--models",
        default="LateFusionModel,MidFusionModel,ScoreFusionModel",
        help="Comma-separated model names or 'all'.",
    )
    parser.add_argument(
        "--batch-sizes",
        default="8,12,16,24,32",
        help="Comma-separated batch sizes to test.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=151,
        help="Number of output classes to build the classifier head with.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to probe. CUDA is required for memory reporting.",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Load pretrained weights. Default is off to avoid downloads; memory is usually equivalent.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use CUDA automatic mixed precision during the probe.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Override config.image_size for every probed model.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=None,
        help="Override config.patch_size for every probed model.",
    )
    parser.add_argument(
        "--resize-size",
        type=int,
        default=None,
        help="Override config.resize_size when present.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV output path.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional JSON output path.",
    )
    parser.add_argument(
        "--stop-on-oom",
        action="store_true",
        help="Stop testing larger batch sizes for a model after the first CUDA OOM.",
    )
    return parser.parse_args()


def parse_models(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(MODEL_SPECS)
    models = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(models) - set(MODEL_SPECS))
    if unknown:
        raise ValueError(f"Unknown model(s): {', '.join(unknown)}")
    return models


def parse_batch_sizes(value: str) -> list[int]:
    sizes = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("--batch-sizes must contain positive integers.")
    return sizes


def format_gib(value: int | float | None) -> str:
    if value is None:
        return ""
    return f"{float(value) / 1024**3:.2f}"


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda:0")
    return device


def apply_config_overrides(config: Any, args: argparse.Namespace) -> None:
    if args.image_size is not None and hasattr(config, "image_size"):
        config.image_size = args.image_size
    if args.patch_size is not None and hasattr(config, "patch_size"):
        config.patch_size = args.patch_size
    if args.resize_size is not None and hasattr(config, "resize_size"):
        config.resize_size = args.resize_size
    if hasattr(config, "pretrained"):
        config.pretrained = bool(args.pretrained)


def build_wrapper(model_name: str, args: argparse.Namespace) -> tuple[Any, str]:
    module_name, class_name, input_kind = MODEL_SPECS[model_name]
    module = importlib.import_module(module_name)
    wrapper_class = getattr(module, class_name)
    wrapper = wrapper_class()
    apply_config_overrides(wrapper.config, args)
    wrapper.device = resolve_device(args.device)
    return wrapper, input_kind


def build_optimizer(wrapper: Any, model: nn.Module) -> torch.optim.Optimizer:
    if hasattr(wrapper, "_build_optimizer"):
        return wrapper._build_optimizer(model)
    return torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.05)


def make_inputs(
    input_kind: str,
    batch_size: int,
    image_size: int,
    num_classes: int,
    device: torch.device,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    target = torch.randint(0, num_classes, (batch_size,), device=device)
    if input_kind == "image":
        return (torch.randn(batch_size, 3, image_size, image_size, device=device),), target
    if input_kind == "audio":
        return (torch.randn(batch_size, 3, image_size, image_size, device=device),), target
    if input_kind == "multimodal":
        image = torch.randn(batch_size, 3, image_size, image_size, device=device)
        audio = torch.randn(batch_size, 3, image_size, image_size, device=device)
        return (image, audio), target
    raise ValueError(f"Unsupported input kind: {input_kind}")


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    target: torch.Tensor,
    use_amp: bool,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    if use_amp:
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        with torch.cuda.amp.autocast(enabled=True):
            logits = model(*inputs)
            loss = criterion(logits, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        logits = model(*inputs)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
    return float(loss.detach().cpu())


def probe_one(model_name: str, batch_size: int, args: argparse.Namespace) -> dict[str, Any]:
    cleanup_cuda()
    device = resolve_device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this memory probe.")

    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    free_before, total = torch.cuda.mem_get_info(device)
    started = time.perf_counter()

    result: dict[str, Any] = {
        "model": model_name,
        "batch_size": batch_size,
        "status": "ok",
        "device": torch.cuda.get_device_name(device),
        "total_gib": total / 1024**3,
        "free_before_gib": free_before / 1024**3,
        "peak_allocated_gib": None,
        "peak_reserved_gib": None,
        "elapsed_sec": None,
        "image_size": None,
        "patch_size": None,
        "message": "",
    }

    try:
        wrapper, input_kind = build_wrapper(model_name, args)
        config = wrapper.config
        image_size = int(getattr(config, "image_size"))
        patch_size = int(getattr(config, "patch_size", 16))
        result["image_size"] = image_size
        result["patch_size"] = patch_size

        model = wrapper._build_model(
            num_classes=args.num_classes,
            use_pretrained=bool(args.pretrained),
        ).to(device)
        model.train()
        optimizer = build_optimizer(wrapper, model)
        criterion = nn.CrossEntropyLoss()
        inputs, target = make_inputs(
            input_kind=input_kind,
            batch_size=batch_size,
            image_size=image_size,
            num_classes=args.num_classes,
            device=device,
        )
        loss_value = train_step(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            inputs=inputs,
            target=target,
            use_amp=bool(args.amp),
        )
        torch.cuda.synchronize(device)
        result["loss"] = loss_value
    except RuntimeError as error:
        message = str(error).splitlines()[0]
        result["status"] = "oom" if "out of memory" in str(error).lower() else "runtime_error"
        result["message"] = message
    except Exception as error:
        result["status"] = "error"
        result["message"] = str(error).splitlines()[0]
    finally:
        result["elapsed_sec"] = time.perf_counter() - started
        if torch.cuda.is_available():
            result["peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 1024**3
            result["peak_reserved_gib"] = torch.cuda.max_memory_reserved(device) / 1024**3
        cleanup_cuda()

    return result


def print_result(result: dict[str, Any]) -> None:
    fields = [
        f"{result['model']}",
        f"batch={result['batch_size']}",
        f"status={result['status']}",
        f"image={result.get('image_size')}",
        f"patch={result.get('patch_size')}",
        f"free_before={result['free_before_gib']:.2f}GiB",
        f"peak_alloc={result['peak_allocated_gib']:.2f}GiB",
        f"peak_reserved={result['peak_reserved_gib']:.2f}GiB",
        f"elapsed={result['elapsed_sec']:.2f}s",
    ]
    if result.get("message"):
        fields.append(f"message={result['message']}")
    print("\t".join(fields), flush=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "batch_size",
        "status",
        "device",
        "image_size",
        "patch_size",
        "total_gib",
        "free_before_gib",
        "peak_allocated_gib",
        "peak_reserved_gib",
        "elapsed_sec",
        "loss",
        "message",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    models = parse_models(args.models)
    batch_sizes = parse_batch_sizes(args.batch_sizes)
    rows: list[dict[str, Any]] = []

    for model_name in models:
        saw_oom = False
        for batch_size in batch_sizes:
            if saw_oom and args.stop_on_oom:
                break
            result = probe_one(model_name, batch_size, args)
            print_result(result)
            rows.append(result)
            saw_oom = result["status"] == "oom"

    if args.csv is not None:
        write_csv(args.csv, rows)
        print(f"CSV written to: {args.csv}")
    if args.json is not None:
        write_json(args.json, rows)
        print(f"JSON written to: {args.json}")


if __name__ == "__main__":
    main()
