from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


MODEL_ORDER = (
    "ImageModel",
    "AudioModel",
    "LateFusionModel",
    "MidFusionModel",
    "ScoreFusionModel",
)
MODEL_LABELS = {
    "ImageModel": "Image",
    "AudioModel": "Audio",
    "LateFusionModel": "Late Fusion",
    "MidFusionModel": "Mid Fusion",
    "ScoreFusionModel": "Score Fusion",
}
COLORS = {
    "ImageModel": "#4C78A8",
    "AudioModel": "#F58518",
    "LateFusionModel": "#54A24B",
    "MidFusionModel": "#E45756",
    "ScoreFusionModel": "#B279A2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the five most recent completed training runs."
    )
    parser.add_argument(
        "--training-root",
        type=Path,
        default=Path("artifacts/training_runs"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/training_analysis/latest_5"),
    )
    parser.add_argument(
        "--dataset-name-contains",
        default="",
        help="Only include runs whose dataset path contains this text.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Only include runs with this image_size in config_snapshot.json.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of completed runs to analyze.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_latest_runs(
    training_root: Path,
    dataset_name_contains: str = "",
    image_size: int | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for run_root in training_root.iterdir():
        if not run_root.is_dir():
            continue
        summary_path = run_root / "logs" / "training_summary.json"
        history_path = run_root / "logs" / "training_history.json"
        metrics_path = run_root / "metrics" / "test_metrics.json"
        if not all(path.exists() for path in (summary_path, history_path, metrics_path)):
            continue

        summary = load_json(summary_path)
        if summary.get("status") != "completed":
            continue
        if dataset_name_contains and dataset_name_contains not in str(
            summary.get("dataset_root", "")
        ):
            continue
        config_path = run_root / "config_snapshot.json"
        config = load_json(config_path) if config_path.exists() else {}
        if image_size is not None and config.get("image_size") != image_size:
            continue
        candidates.append(
            {
                "root": run_root,
                "modified": summary_path.stat().st_mtime,
                "summary": summary,
                "history": load_json(history_path),
                "metrics": load_json(metrics_path),
                "config": config,
            }
        )

    latest = sorted(candidates, key=lambda item: item["modified"], reverse=True)[
        : max(1, limit)
    ]
    if not latest:
        raise ValueError("No completed training runs matched the requested filters.")
    return sorted(
        latest,
        key=lambda item: MODEL_ORDER.index(item["summary"]["model_name"]),
    )


def save_metrics_csv(runs: list[dict[str, Any]], output_root: Path) -> None:
    fieldnames = [
        "model",
        "run",
        "epochs",
        "train_samples",
        "val_samples",
        "test_samples",
        "loss",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "best_epoch",
        "best_val_loss",
        "best_val_accuracy",
        "final_val_loss",
        "final_val_accuracy",
    ]
    with (output_root / "model_metrics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            summary = run["summary"]
            metrics = run["metrics"]
            history = run["history"]
            best_item = min(history, key=lambda item: item["val"]["loss"])
            final_item = history[-1]
            writer.writerow(
                {
                    "model": summary["model_name"],
                    "run": run["root"].name,
                    "epochs": summary["epochs_completed"],
                    "train_samples": summary["train_samples"],
                    "val_samples": summary["val_samples"],
                    "test_samples": summary["test_samples"],
                    "loss": metrics["loss"],
                    "accuracy": metrics["accuracy"],
                    "macro_precision": metrics["macro_precision"],
                    "macro_recall": metrics["macro_recall"],
                    "macro_f1": metrics["macro_f1"],
                    "best_epoch": best_item["epoch"],
                    "best_val_loss": best_item["val"]["loss"],
                    "best_val_accuracy": best_item["val"]["accuracy"],
                    "final_val_loss": final_item["val"]["loss"],
                    "final_val_accuracy": final_item["val"]["accuracy"],
                }
            )


def save_class_f1_csv(runs: list[dict[str, Any]], output_root: Path) -> None:
    classes = runs[0]["metrics"]["confusion_matrix_labels"]
    with (output_root / "class_f1_scores.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", *classes])
        for run in runs:
            report = run["metrics"]["classification_report"]
            writer.writerow(
                [
                    run["summary"]["model_name"],
                    *[report[class_name]["f1-score"] for class_name in classes],
                ]
            )


def plot_performance_bars(
    runs: list[dict[str, Any]], output_root: Path
) -> None:
    labels = [MODEL_LABELS[run["summary"]["model_name"]] for run in runs]
    metric_specs = (
        ("accuracy", "Accuracy"),
        ("macro_precision", "Macro Precision"),
        ("macro_recall", "Macro Recall"),
        ("macro_f1", "Macro F1"),
    )
    x = np.arange(len(labels))
    width = 0.19
    fig, axis = plt.subplots(figsize=(13, 7))

    for index, (key, title) in enumerate(metric_specs):
        values = [run["metrics"][key] * 100.0 for run in runs]
        positions = x + (index - 1.5) * width
        bars = axis.bar(positions, values, width, label=title)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.25,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    axis.set_title("Model Performance Comparison")
    axis.set_ylabel("Score (%)")
    axis.set_xticks(x, labels)
    axis.set_ylim(85, 102)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(output_root / "model_performance_bar.png", dpi=180)
    plt.close(fig)


def plot_loss_bars(runs: list[dict[str, Any]], output_root: Path) -> None:
    labels = [MODEL_LABELS[run["summary"]["model_name"]] for run in runs]
    values = [run["metrics"]["loss"] for run in runs]
    colors = [COLORS[run["summary"]["model_name"]] for run in runs]
    fig, axis = plt.subplots(figsize=(10, 6))
    bars = axis.bar(labels, values, color=colors)
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(values) * 0.02,
            f"{value:.4f}",
            ha="center",
            va="bottom",
        )
    axis.set_title("Evaluation Loss Comparison")
    axis.set_ylabel("Cross-Entropy Loss")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_root / "model_loss_bar.png", dpi=180)
    plt.close(fig)


def plot_training_curves(
    runs: list[dict[str, Any]],
    output_root: Path,
    metric: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True)
    for run in runs:
        model_name = run["summary"]["model_name"]
        history = run["history"]
        epochs = [item["epoch"] for item in history]
        axes[0].plot(
            epochs,
            [item["train"][metric] for item in history],
            label=MODEL_LABELS[model_name],
            color=COLORS[model_name],
            linewidth=2,
        )
        axes[1].plot(
            epochs,
            [item["val"][metric] for item in history],
            label=MODEL_LABELS[model_name],
            color=COLORS[model_name],
            linewidth=2,
        )

    title = metric.capitalize()
    axes[0].set_title(f"Train {title}")
    axes[1].set_title(f"Validation {title}")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.set_ylabel(title)
        axis.grid(alpha=0.25)
        axis.legend()
    if metric == "loss":
        axes[0].set_yscale("log")
        axes[1].set_yscale("log")
    else:
        axes[0].set_ylim(0.85, 1.01)
        axes[1].set_ylim(0.75, 1.01)

    fig.suptitle(f"Training History: {title}", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_root / f"training_{metric}_curves.png", dpi=180)
    plt.close(fig)


def save_report(runs: list[dict[str, Any]], output_root: Path) -> None:
    ranked = sorted(runs, key=lambda run: run["metrics"]["macro_f1"], reverse=True)
    lines = [
        "# Latest Five Training Runs",
        "",
        "> Test is an exact copy of validation for this dataset. Evaluation metrics are "
        "useful for relative inspection only and are not independent test estimates.",
        "",
        "| Rank | Model | Accuracy | Macro F1 | Loss | Best epoch |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for rank, run in enumerate(ranked, start=1):
        history = run["history"]
        best_item = min(history, key=lambda item: item["val"]["loss"])
        metrics = run["metrics"]
        lines.append(
            f"| {rank} | {run['summary']['model_name']} | "
            f"{metrics['accuracy'] * 100:.2f}% | "
            f"{metrics['macro_f1'] * 100:.2f}% | "
            f"{metrics['loss']:.6f} | {best_item['epoch']} |"
        )

    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `model_performance_bar.png`: accuracy, precision, recall, and F1",
            "- `model_loss_bar.png`: evaluation loss",
            "- `training_loss_curves.png`: epoch-level train/validation loss",
            "- `training_accuracy_curves.png`: epoch-level train/validation accuracy",
            "- `model_metrics.csv`: model-level metrics",
            "- `class_f1_scores.csv`: per-class F1 scores",
        ]
    )
    (output_root / "report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    training_root = args.training_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    runs = collect_latest_runs(
        training_root,
        dataset_name_contains=args.dataset_name_contains,
        image_size=args.image_size,
        limit=args.limit,
    )
    save_metrics_csv(runs, output_root)
    save_class_f1_csv(runs, output_root)
    plot_performance_bars(runs, output_root)
    plot_loss_bars(runs, output_root)
    plot_training_curves(runs, output_root, "loss")
    plot_training_curves(runs, output_root, "accuracy")
    save_report(runs, output_root)

    print(f"Analysis written to: {output_root}")
    for run in runs:
        metrics = run["metrics"]
        print(
            f"{run['summary']['model_name']}: "
            f"accuracy={metrics['accuracy']:.6f}, "
            f"macro_f1={metrics['macro_f1']:.6f}, "
            f"loss={metrics['loss']:.6f}"
        )


if __name__ == "__main__":
    main()
