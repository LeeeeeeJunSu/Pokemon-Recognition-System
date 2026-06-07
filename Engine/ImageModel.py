from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:
    from .IModel import IModel, PathLike
    from .VisionTransformerFactory import build_vit_b_16
except ImportError:
    from IModel import IModel, PathLike
    from VisionTransformerFactory import build_vit_b_16


@dataclass
class _ImageModelConfig:
    model_name: str = "vit_b_16"
    pretrained: bool = False
    pretrained_weights: str = "IMAGENET1K_V1"
    image_size: int = 224
    patch_size: int = 16
    resize_size: int = 256
    batch_size: int = 16
    num_workers: int = 0
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 10
    dropout: float = 0.0
    early_stopping_patience: int = 5
    optimizer: str = "adamw"
    seed: int = 42
    device: str = "auto"
    top_k: int = 5
    best_model_name: str = "ImageModel_best.pt"
    last_model_name: str = "ImageModel_last.pt"

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "_ImageModelConfig":
        valid_keys = {field.name for field in fields(cls)}
        filtered = {key: value for key, value in values.items() if key in valid_keys}
        return cls(**filtered)


@dataclass
class _ImageSample:
    split: str
    class_name: str
    label_index: int
    sample_id: str
    image_path: Path
    meta_path: Path | None = None


class _ImageOnlyDataset(Dataset):
    def __init__(self, samples: list[_ImageSample], transform: transforms.Compose) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str, str, str]:
        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            image = ImageOps.exif_transpose(image)
            image = image.convert("RGB")
            tensor = self.transform(image)
        return (
            tensor,
            sample.label_index,
            sample.sample_id,
            sample.class_name,
            str(sample.image_path),
        )


class ImageModel(IModel):
    def __init__(self) -> None:
        self._project_root = Path(__file__).resolve().parents[1]
        self.config = self._load_config()
        self.device = self._resolve_device(self.config.device)
        self.model: nn.Module | None = None
        self.class_to_index: dict[str, int] = {}
        self.index_to_class: dict[int, str] = {}
        self.class_metadata: dict[str, dict[str, Any]] = {}

    def Train(self, DataSetPath: PathLike, ResultSavePath: PathLike) -> None:
        self.config = self._load_config()
        self.device = self._resolve_device(self.config.device)
        self._seed_everything(self.config.seed)

        dataset_root = Path(DataSetPath).expanduser().resolve()
        result_root = Path(ResultSavePath).expanduser().resolve()
        self._validate_dataset_root(dataset_root)
        result_dirs = self._prepare_result_directories(result_root)

        self.class_to_index, self.index_to_class, self.class_metadata = self._load_label_mapping(
            dataset_root
        )
        train_samples = self._collect_split_samples(dataset_root / "train", "train")
        val_samples = self._collect_split_samples(dataset_root / "val", "val")
        test_samples = self._collect_split_samples(dataset_root / "test", "test")

        if not train_samples:
            raise ValueError(f"No training samples found under: {dataset_root / 'train'}")
        if not test_samples:
            raise ValueError(f"No test samples found under: {dataset_root / 'test'}")

        train_loader = DataLoader(
            _ImageOnlyDataset(train_samples, self._build_train_transform()),
            **self._build_dataloader_kwargs(shuffle=True),
        )
        val_loader = None
        if val_samples:
            val_loader = DataLoader(
                _ImageOnlyDataset(val_samples, self._build_eval_transform()),
                **self._build_dataloader_kwargs(shuffle=False),
            )
        test_loader = DataLoader(
            _ImageOnlyDataset(test_samples, self._build_eval_transform()),
            **self._build_dataloader_kwargs(shuffle=False),
        )

        self.model = self._build_model(
            num_classes=len(self.class_to_index),
            use_pretrained=self.config.pretrained,
        ).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = self._build_optimizer(self.model)

        self._save_json(result_root / "config_snapshot.json", asdict(self.config))

        history: list[dict[str, Any]] = []
        best_monitor_loss = float("inf")
        epochs_without_improvement = 0
        best_checkpoint_path = result_dirs["checkpoints"] / self.config.best_model_name
        last_checkpoint_path = result_dirs["checkpoints"] / self.config.last_model_name
        history_path = result_dirs["logs"] / "training_history.json"
        summary_path = result_dirs["logs"] / "training_summary.json"
        train_summary = {
            "model_name": self.__class__.__name__,
            "dataset_root": str(dataset_root),
            "result_root": str(result_root),
            "num_classes": len(self.class_to_index),
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "test_samples": len(test_samples),
            "best_checkpoint": str(best_checkpoint_path),
            "last_checkpoint": str(last_checkpoint_path),
            "configured_epochs": self.config.epochs,
            "epochs_completed": 0,
            "status": "running",
        }
        self._save_json(history_path, history)
        self._save_json(summary_path, train_summary)

        for epoch in range(1, self.config.epochs + 1):
            train_metrics = self._run_epoch(train_loader, criterion, optimizer)
            val_metrics = (
                self._evaluate_classification_loader(val_loader, criterion)["summary"]
                if val_loader is not None
                else None
            )

            history_entry = {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
            }
            history.append(history_entry)

            monitor_loss = (
                val_metrics["loss"]
                if val_metrics is not None
                else train_metrics["loss"]
            )
            self._save_checkpoint(last_checkpoint_path)

            if monitor_loss < best_monitor_loss:
                best_monitor_loss = monitor_loss
                epochs_without_improvement = 0
                self._save_checkpoint(best_checkpoint_path)
            else:
                epochs_without_improvement += 1

            train_summary["epochs_completed"] = len(history)
            train_summary["status"] = "running"
            train_summary["best_monitor_loss"] = best_monitor_loss
            self._save_json(history_path, history)
            self._save_json(summary_path, train_summary)

            if (
                val_loader is not None
                and self.config.early_stopping_patience > 0
                and epochs_without_improvement >= self.config.early_stopping_patience
            ):
                break

        if best_checkpoint_path.exists():
            self.Load(best_checkpoint_path)
        else:
            self.Load(last_checkpoint_path)

        test_result = self._evaluate_classification_loader(test_loader, criterion)
        train_summary["epochs_completed"] = len(history)
        train_summary["status"] = "completed"
        self._save_json(history_path, history)
        self._save_json(summary_path, train_summary)
        self._save_json(result_dirs["metrics"] / "test_metrics.json", test_result["summary"])
        self._save_json(
            result_dirs["predictions"] / "test_predictions.json",
            test_result["predictions"],
        )

    def Load(self, NetworkFilePath: PathLike) -> None:
        checkpoint_path = Path(NetworkFilePath).expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Network file not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_config = checkpoint.get("config", {})
        self.config = _ImageModelConfig.from_mapping(checkpoint_config)
        self.device = self._resolve_device(self.config.device)

        self.class_to_index = {
            str(key): int(value)
            for key, value in checkpoint.get("class_to_index", {}).items()
        }
        self.index_to_class = {
            int(key): str(value)
            for key, value in checkpoint.get("index_to_class", {}).items()
        }
        self.class_metadata = {
            str(key): value for key, value in checkpoint.get("class_metadata", {}).items()
        }

        num_classes = int(checkpoint.get("num_classes", len(self.index_to_class)))
        self.model = self._build_model(num_classes=num_classes, use_pretrained=False).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def Inference(self, DataPath: PathLike, ResultSavePath: PathLike) -> None:
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call Load() or Train() before Inference().")

        data_path = Path(DataPath).expanduser().resolve()
        result_root = Path(ResultSavePath).expanduser().resolve()
        result_dirs = self._prepare_result_directories(result_root)

        inference_samples = self._collect_inference_samples(data_path)
        inference_loader = DataLoader(
            _ImageOnlyDataset(inference_samples, self._build_eval_transform()),
            **self._build_dataloader_kwargs(shuffle=False),
        )
        inference_result = self._predict_loader(inference_loader)

        summary = {
            "data_path": str(data_path),
            "result_root": str(result_root),
            "num_samples": len(inference_samples),
        }
        if inference_result["summary"] is not None:
            summary["metrics"] = inference_result["summary"]

        self._save_json(result_dirs["logs"] / "inference_summary.json", summary)
        self._save_json(
            result_dirs["predictions"] / "inference_predictions.json",
            inference_result["predictions"],
        )
        if inference_result["summary"] is not None:
            self._save_json(
                result_dirs["metrics"] / "inference_metrics.json",
                inference_result["summary"],
            )

    def _load_config(self) -> _ImageModelConfig:
        config_dir = self._project_root / "Config"
        candidates = [
            config_dir / "ImageModel.json",
            config_dir / "image_model.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                with candidate.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                if isinstance(payload.get("ImageModel"), dict):
                    payload = payload["ImageModel"]
                return _ImageModelConfig.from_mapping(payload)
        return _ImageModelConfig()

    def _resolve_device(self, configured_device: str) -> torch.device:
        if configured_device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(configured_device)

    def _seed_everything(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _validate_dataset_root(self, dataset_root: Path) -> None:
        if not dataset_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
        if not (dataset_root / "train").exists():
            raise FileNotFoundError(f"Missing train split: {dataset_root / 'train'}")
        if not (dataset_root / "test").exists():
            raise FileNotFoundError(f"Missing test split: {dataset_root / 'test'}")

    def _prepare_result_directories(self, result_root: Path) -> dict[str, Path]:
        directories = {
            "root": result_root,
            "logs": result_root / "logs",
            "metrics": result_root / "metrics",
            "predictions": result_root / "predictions",
            "checkpoints": result_root / "checkpoints",
        }
        for directory in directories.values():
            directory.mkdir(parents=True, exist_ok=True)
        return directories

    def _build_dataloader_kwargs(self, shuffle: bool) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "batch_size": self.config.batch_size,
            "shuffle": shuffle,
            "num_workers": self.config.num_workers,
            "pin_memory": self.device.type == "cuda",
        }
        if self.config.num_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = 2
        return kwargs

    def _move_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self.device, non_blocking=self.device.type == "cuda")

    def _load_label_mapping(
        self, dataset_root: Path
    ) -> tuple[dict[str, int], dict[int, str], dict[str, dict[str, Any]]]:
        labels_path = dataset_root / "labels.json"
        if labels_path.exists():
            with labels_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            class_to_index: dict[str, int] = {}
            class_metadata: dict[str, dict[str, Any]] = {}
            for class_name, metadata in payload.items():
                class_to_index[str(class_name)] = int(metadata.get("index", class_name))
                class_metadata[str(class_name)] = metadata
            index_to_class = {
                index: class_name for class_name, index in class_to_index.items()
            }
            return class_to_index, index_to_class, class_metadata

        discovered_classes: set[str] = set()
        for split_name in ("train", "val", "test"):
            split_root = dataset_root / split_name
            if not split_root.exists():
                continue
            for class_dir in split_root.iterdir():
                if class_dir.is_dir():
                    discovered_classes.add(class_dir.name)

        sorted_classes = sorted(discovered_classes, key=self._class_sort_key)
        if all(class_name.isdigit() for class_name in sorted_classes):
            class_to_index = {class_name: int(class_name) for class_name in sorted_classes}
        else:
            class_to_index = {
                class_name: index for index, class_name in enumerate(sorted_classes)
            }
        index_to_class = {index: class_name for class_name, index in class_to_index.items()}
        return class_to_index, index_to_class, {}

    def _collect_split_samples(self, split_root: Path, split_name: str) -> list[_ImageSample]:
        if not split_root.exists():
            return []

        samples: list[_ImageSample] = []
        for class_dir in sorted(
            (item for item in split_root.iterdir() if item.is_dir()),
            key=lambda path: self._class_sort_key(path.name),
        ):
            class_name = class_dir.name
            label_index = self.class_to_index.get(class_name, -1)
            for sample_dir in sorted(
                (item for item in class_dir.iterdir() if item.is_dir()),
                key=lambda path: path.name,
            ):
                image_path = sample_dir / "image.png"
                if not image_path.exists():
                    continue
                meta_path = sample_dir / "meta.json"
                samples.append(
                    _ImageSample(
                        split=split_name,
                        class_name=class_name,
                        label_index=label_index,
                        sample_id=sample_dir.name,
                        image_path=image_path,
                        meta_path=meta_path if meta_path.exists() else None,
                    )
                )
        return samples

    def _collect_inference_samples(self, data_path: Path) -> list[_ImageSample]:
        if not data_path.exists():
            raise FileNotFoundError(f"Inference data path not found: {data_path}")

        if data_path.is_file() and data_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            return [
                _ImageSample(
                    split="inference",
                    class_name="unknown",
                    label_index=-1,
                    sample_id=data_path.stem,
                    image_path=data_path,
                )
            ]

        if (data_path / "image.png").exists():
            return [
                _ImageSample(
                    split="inference",
                    class_name=data_path.parent.name,
                    label_index=self.class_to_index.get(data_path.parent.name, -1),
                    sample_id=data_path.name,
                    image_path=data_path / "image.png",
                    meta_path=(data_path / "meta.json") if (data_path / "meta.json").exists() else None,
                )
            ]

        split_samples: list[_ImageSample] = []
        has_named_splits = any((data_path / split_name).exists() for split_name in ("train", "val", "test"))
        if has_named_splits:
            for split_name in ("train", "val", "test"):
                split_root = data_path / split_name
                split_samples.extend(self._collect_split_samples(split_root, split_name))
            if split_samples:
                return split_samples

        recursive_samples: list[_ImageSample] = []
        for image_path in sorted(data_path.rglob("image.png")):
            sample_dir = image_path.parent
            class_name = sample_dir.parent.name if sample_dir.parent != data_path else "unknown"
            recursive_samples.append(
                _ImageSample(
                    split="inference",
                    class_name=class_name,
                    label_index=self.class_to_index.get(class_name, -1),
                    sample_id=sample_dir.name,
                    image_path=image_path,
                    meta_path=(sample_dir / "meta.json") if (sample_dir / "meta.json").exists() else None,
                )
            )

        if not recursive_samples:
            raise ValueError(f"No image samples found for inference under: {data_path}")
        return recursive_samples

    def _build_train_transform(self) -> transforms.Compose:
        return self._build_eval_transform()

    def _build_eval_transform(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize(self.config.resize_size),
                transforms.CenterCrop(self.config.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def _build_model(self, num_classes: int, use_pretrained: bool) -> nn.Module:
        if self.config.model_name != "vit_b_16":
            raise ValueError(f"Unsupported image model: {self.config.model_name}")

        model = build_vit_b_16(
            image_size=self.config.image_size,
            patch_size=self.config.patch_size,
            use_pretrained=use_pretrained,
            pretrained_weights=self.config.pretrained_weights,
        )
        in_features = model.heads.head.in_features
        model.heads.head = nn.Sequential(
            nn.Dropout(self.config.dropout),
            nn.Linear(in_features, num_classes),
        )
        return model

    def _build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        optimizer_name = self.config.optimizer.lower()
        if optimizer_name == "adamw":
            return torch.optim.AdamW(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        if optimizer_name == "sgd":
            return torch.optim.SGD(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9,
            )
        raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")

    def _run_epoch(
        self,
        data_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, float]:
        if self.model is None:
            raise RuntimeError("Model is not initialized.")

        self.model.train()
        running_loss = 0.0
        y_true: list[int] = []
        y_pred: list[int] = []

        for images, labels, *_ in data_loader:
            images = self._move_tensor(images)
            labels = self._move_tensor(labels)

            optimizer.zero_grad()
            logits = self.model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * images.size(0)
            predictions = torch.argmax(logits, dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())

        average_loss = running_loss / max(len(data_loader.dataset), 1)
        accuracy = accuracy_score(y_true, y_pred) if y_true else 0.0
        return {
            "loss": average_loss,
            "accuracy": float(accuracy),
        }

    def _evaluate_classification_loader(
        self,
        data_loader: DataLoader | None,
        criterion: nn.Module,
    ) -> dict[str, Any]:
        if data_loader is None:
            return {"summary": None, "predictions": []}
        if self.model is None:
            raise RuntimeError("Model is not initialized.")

        self.model.eval()
        running_loss = 0.0
        y_true: list[int] = []
        y_pred: list[int] = []
        predictions: list[dict[str, Any]] = []

        with torch.no_grad():
            for images, labels, sample_ids, class_names, image_paths in data_loader:
                images = self._move_tensor(images)
                labels = self._move_tensor(labels)

                logits = self.model(images)
                loss = criterion(logits, labels)
                probabilities = torch.softmax(logits, dim=1)
                top_k = min(self.config.top_k, probabilities.size(1))
                top_probabilities, top_indices = torch.topk(probabilities, k=top_k, dim=1)
                predicted_indices = top_indices[:, 0]

                running_loss += float(loss.item()) * images.size(0)
                batch_true = labels.cpu().tolist()
                batch_pred = predicted_indices.cpu().tolist()
                y_true.extend(batch_true)
                y_pred.extend(batch_pred)

                for index, sample_id in enumerate(sample_ids):
                    predictions.append(
                        self._build_prediction_record(
                            sample_id=sample_id,
                            image_path=image_paths[index],
                            input_class_name=class_names[index],
                            ground_truth_index=batch_true[index],
                            top_indices=top_indices[index].cpu().tolist(),
                            top_probabilities=top_probabilities[index].cpu().tolist(),
                        )
                    )

        summary = self._build_metrics_summary(
            y_true=y_true,
            y_pred=y_pred,
            average_loss=running_loss / max(len(data_loader.dataset), 1),
        )
        return {"summary": summary, "predictions": predictions}

    def _predict_loader(self, data_loader: DataLoader) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Model is not initialized.")

        self.model.eval()
        y_true: list[int] = []
        y_pred: list[int] = []
        predictions: list[dict[str, Any]] = []

        with torch.no_grad():
            for images, labels, sample_ids, class_names, image_paths in data_loader:
                images = self._move_tensor(images)
                logits = self.model(images)
                probabilities = torch.softmax(logits, dim=1)
                top_k = min(self.config.top_k, probabilities.size(1))
                top_probabilities, top_indices = torch.topk(probabilities, k=top_k, dim=1)

                batch_labels = labels.tolist()
                batch_pred = top_indices[:, 0].cpu().tolist()

                for index, sample_id in enumerate(sample_ids):
                    ground_truth_index = batch_labels[index]
                    if ground_truth_index >= 0:
                        y_true.append(ground_truth_index)
                        y_pred.append(batch_pred[index])

                    predictions.append(
                        self._build_prediction_record(
                            sample_id=sample_id,
                            image_path=image_paths[index],
                            input_class_name=class_names[index],
                            ground_truth_index=ground_truth_index,
                            top_indices=top_indices[index].cpu().tolist(),
                            top_probabilities=top_probabilities[index].cpu().tolist(),
                        )
                    )

        summary = None
        if y_true:
            summary = self._build_metrics_summary(y_true=y_true, y_pred=y_pred, average_loss=None)
        return {"summary": summary, "predictions": predictions}

    def _build_prediction_record(
        self,
        sample_id: str,
        image_path: str,
        input_class_name: str,
        ground_truth_index: int,
        top_indices: list[int],
        top_probabilities: list[float],
    ) -> dict[str, Any]:
        predicted_index = int(top_indices[0])
        record = {
            "sample_id": sample_id,
            "image_path": image_path,
            "input_class_name": input_class_name,
            "ground_truth_index": ground_truth_index if ground_truth_index >= 0 else None,
            "ground_truth_class_name": self.index_to_class.get(ground_truth_index)
            if ground_truth_index >= 0
            else None,
            "predicted_index": predicted_index,
            "predicted_class_name": self.index_to_class.get(predicted_index),
            "predicted_display_name": self._display_label(predicted_index),
            "confidence": float(top_probabilities[0]),
            "top_k": [],
        }

        for class_index, probability in zip(top_indices, top_probabilities):
            record["top_k"].append(
                {
                    "index": int(class_index),
                    "class_name": self.index_to_class.get(int(class_index)),
                    "display_name": self._display_label(int(class_index)),
                    "confidence": float(probability),
                }
            )
        return record

    def _build_metrics_summary(
        self,
        y_true: list[int],
        y_pred: list[int],
        average_loss: float | None,
    ) -> dict[str, Any]:
        labels = sorted(self.index_to_class.keys())
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
        display_labels = [self._display_label(label) for label in labels]
        summary = {
            "loss": average_loss,
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_precision": float(precision),
            "macro_recall": float(recall),
            "macro_f1": float(f1),
            "confusion_matrix_labels": display_labels,
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
            "classification_report": classification_report(
                y_true,
                y_pred,
                labels=labels,
                target_names=display_labels,
                output_dict=True,
                zero_division=0,
            ),
        }
        return summary

    def _display_label(self, label_index: int) -> str:
        class_name = self.index_to_class.get(label_index, str(label_index))
        metadata = self.class_metadata.get(class_name, {})
        return str(metadata.get("common_name", class_name))

    def _save_checkpoint(self, checkpoint_path: Path) -> None:
        if self.model is None:
            raise RuntimeError("Model is not initialized.")
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "class_to_index": self.class_to_index,
            "index_to_class": self.index_to_class,
            "class_metadata": self.class_metadata,
            "config": asdict(self.config),
            "num_classes": len(self.class_to_index),
        }
        torch.save(checkpoint, checkpoint_path)

    def _save_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def _class_sort_key(self, class_name: str) -> tuple[int, str]:
        if class_name.isdigit():
            return (0, f"{int(class_name):08d}")
        return (1, class_name)
