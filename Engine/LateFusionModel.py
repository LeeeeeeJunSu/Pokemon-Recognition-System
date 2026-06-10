from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from PIL import Image, ImageOps
from scipy.io import wavfile
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
class _LateFusionConfig:
    image_model_name: str = "vit_b_16"
    audio_model_name: str = "ast_vit_b_16"
    pretrained: bool = False
    pretrained_weights: str = "IMAGENET1K_V1"
    image_size: int = 224
    patch_size: int = 16
    resize_size: int = 256
    batch_size: int = 8
    num_workers: int = 0
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 10
    dropout: float = 0.0
    label_smoothing: float = 0.0
    train_augmentation: bool = False
    random_erasing_probability: float = 0.0
    freeze_backbone_epochs: int = 0
    backbone_lr_scale: float = 1.0
    modality_dropout: float = 0.0
    fusion_hidden_dim: int = 512
    fusion_dropout: float = 0.2
    early_stopping_patience: int = 5
    optimizer: str = "adamw"
    seed: int = 42
    device: str = "auto"
    top_k: int = 5
    best_model_name: str = "LateFusionModel_best.pt"
    last_model_name: str = "LateFusionModel_last.pt"
    sample_rate: int = 22050
    clip_duration_seconds: float = 10.0
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 128
    f_min: float = 20.0
    f_max: float = 11025.0

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "_LateFusionConfig":
        valid_keys = {field.name for field in fields(cls)}
        filtered = {key: value for key, value in values.items() if key in valid_keys}
        return cls(**filtered)


@dataclass
class _MultimodalSample:
    split: str
    class_name: str
    label_index: int
    sample_id: str
    image_path: Path
    audio_path: Path
    meta_path: Path | None = None


class _AudioFeatureExtractor:
    def __init__(self, config: _LateFusionConfig) -> None:
        self.config = config
        self.target_num_samples = int(config.sample_rate * config.clip_duration_seconds)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            f_min=config.f_min,
            f_max=config.f_max,
            center=True,
            power=2.0,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)

    def __call__(self, audio_path: Path) -> torch.Tensor:
        waveform = self._load_waveform(audio_path)
        waveform = self._prepare_waveform(waveform)
        spectrogram = self.mel_transform(waveform.unsqueeze(0))
        spectrogram = self.amplitude_to_db(spectrogram)

        spectrogram = spectrogram.squeeze(0)
        spectrogram = (spectrogram - spectrogram.mean()) / (spectrogram.std() + 1e-6)
        spectrogram = spectrogram.unsqueeze(0)
        spectrogram = F.interpolate(
            spectrogram.unsqueeze(0),
            size=(self.config.image_size, self.config.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        spectrogram = spectrogram.repeat(3, 1, 1)
        return spectrogram

    def _load_waveform(self, audio_path: Path) -> torch.Tensor:
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            waveform = waveform.mean(dim=0)
        except Exception:
            sample_rate, audio = wavfile.read(audio_path)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            if np.issubdtype(audio.dtype, np.integer):
                max_value = max(abs(np.iinfo(audio.dtype).min), np.iinfo(audio.dtype).max)
                waveform = torch.from_numpy(audio.astype(np.float32) / float(max_value))
            else:
                waveform = torch.from_numpy(audio.astype(np.float32))

        if sample_rate != self.config.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=sample_rate,
                new_freq=self.config.sample_rate,
            )
        return waveform

    def _prepare_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        waveform = waveform.flatten()

        if waveform.numel() > self.target_num_samples:
            start = (waveform.numel() - self.target_num_samples) // 2
            waveform = waveform[start : start + self.target_num_samples]
        elif waveform.numel() < self.target_num_samples:
            pad_amount = self.target_num_samples - waveform.numel()
            waveform = F.pad(waveform, (0, pad_amount))

        return waveform.clamp(-1.0, 1.0)


class _MultimodalDataset(Dataset):
    def __init__(
        self,
        samples: list[_MultimodalSample],
        image_transform: transforms.Compose,
        audio_processor: _AudioFeatureExtractor,
    ) -> None:
        self.samples = samples
        self.image_transform = image_transform
        self.audio_processor = audio_processor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, int, str, str, str, str]:
        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            image = ImageOps.exif_transpose(image)
            image = image.convert("RGB")
            image_tensor = self.image_transform(image)
        audio_tensor = self.audio_processor(sample.audio_path)
        return (
            image_tensor,
            audio_tensor,
            sample.label_index,
            sample.sample_id,
            sample.class_name,
            str(sample.image_path),
            str(sample.audio_path),
        )


class _LateFusionNetwork(nn.Module):
    def __init__(
        self,
        image_backbone: nn.Module,
        audio_backbone: nn.Module,
        feature_dim: int,
        num_classes: int,
        hidden_dim: int,
        dropout: float,
        modality_dropout: float,
    ) -> None:
        super().__init__()
        self.image_backbone = image_backbone
        self.audio_backbone = audio_backbone
        self.modality_dropout = float(max(0.0, min(1.0, modality_dropout)))
        self.fusion_head = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, image: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        image_features = self.image_backbone(image)
        audio_features = self.audio_backbone(audio)
        image_features, audio_features = self._apply_modality_dropout(
            image_features,
            audio_features,
        )
        fused = torch.cat([image_features, audio_features], dim=1)
        return self.fusion_head(fused)

    def _apply_modality_dropout(
        self,
        image_features: torch.Tensor,
        audio_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.training or self.modality_dropout <= 0.0:
            return image_features, audio_features
        choices = torch.rand(
            image_features.size(0),
            1,
            device=image_features.device,
            dtype=image_features.dtype,
        )
        half_probability = self.modality_dropout / 2.0
        image_mask = (choices >= half_probability).to(image_features.dtype)
        audio_mask = (choices < 1.0 - half_probability).to(audio_features.dtype)
        return image_features * image_mask, audio_features * audio_mask


class LateFusionModel(IModel):
    def __init__(self) -> None:
        self._project_root = Path(__file__).resolve().parents[1]
        self.config = self._load_config()
        self.device = self._resolve_device(self.config.device)
        self.model: _LateFusionNetwork | None = None
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
            _MultimodalDataset(
                train_samples,
                self._build_image_train_transform(),
                _AudioFeatureExtractor(self.config),
            ),
            **self._build_dataloader_kwargs(shuffle=True),
        )
        val_loader = None
        if val_samples:
            val_loader = DataLoader(
                _MultimodalDataset(
                    val_samples,
                    self._build_image_eval_transform(),
                    _AudioFeatureExtractor(self.config),
                ),
                **self._build_dataloader_kwargs(shuffle=False),
            )
        test_loader = DataLoader(
            _MultimodalDataset(
                test_samples,
                self._build_image_eval_transform(),
                _AudioFeatureExtractor(self.config),
            ),
            **self._build_dataloader_kwargs(shuffle=False),
        )

        self.model = self._build_model(
            num_classes=len(self.class_to_index),
            use_pretrained=self.config.pretrained,
        ).to(self.device)
        criterion = nn.CrossEntropyLoss(
            label_smoothing=max(0.0, min(1.0, float(self.config.label_smoothing)))
        )
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

        freeze_backbone_epochs = max(0, int(self.config.freeze_backbone_epochs))
        for epoch in range(1, self.config.epochs + 1):
            backbones_trainable = epoch > freeze_backbone_epochs
            self._set_backbones_trainable(backbones_trainable)
            train_metrics = self._run_epoch(
                train_loader,
                criterion,
                optimizer,
                freeze_backbones=not backbones_trainable,
            )
            val_metrics = (
                self._evaluate_classification_loader(val_loader, criterion)["summary"]
                if val_loader is not None
                else None
            )

            history_entry = {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "backbones_trainable": backbones_trainable,
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

        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_config = checkpoint.get("config", {})
        self.config = _LateFusionConfig.from_mapping(checkpoint_config)
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
            _MultimodalDataset(
                inference_samples,
                self._build_image_eval_transform(),
                _AudioFeatureExtractor(self.config),
            ),
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

    def _load_config(self) -> _LateFusionConfig:
        config_dir = self._project_root / "Config"
        candidates = [
            config_dir / "LateFusionModel.json",
            config_dir / "late_fusion_model.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                with candidate.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                if isinstance(payload.get("LateFusionModel"), dict):
                    payload = payload["LateFusionModel"]
                return _LateFusionConfig.from_mapping(payload)
        return _LateFusionConfig()

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

    def _collect_split_samples(self, split_root: Path, split_name: str) -> list[_MultimodalSample]:
        if not split_root.exists():
            return []

        samples: list[_MultimodalSample] = []
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
                audio_path = sample_dir / "audio.wav"
                if not image_path.exists() or not audio_path.exists():
                    continue
                meta_path = sample_dir / "meta.json"
                samples.append(
                    _MultimodalSample(
                        split=split_name,
                        class_name=class_name,
                        label_index=label_index,
                        sample_id=sample_dir.name,
                        image_path=image_path,
                        audio_path=audio_path,
                        meta_path=meta_path if meta_path.exists() else None,
                    )
                )
        return samples

    def _collect_inference_samples(self, data_path: Path) -> list[_MultimodalSample]:
        if not data_path.exists():
            raise FileNotFoundError(f"Inference data path not found: {data_path}")

        if (data_path / "image.png").exists() and (data_path / "audio.wav").exists():
            return [
                _MultimodalSample(
                    split="inference",
                    class_name=data_path.parent.name,
                    label_index=self.class_to_index.get(data_path.parent.name, -1),
                    sample_id=data_path.name,
                    image_path=data_path / "image.png",
                    audio_path=data_path / "audio.wav",
                    meta_path=(data_path / "meta.json") if (data_path / "meta.json").exists() else None,
                )
            ]

        split_samples: list[_MultimodalSample] = []
        has_named_splits = any((data_path / split_name).exists() for split_name in ("train", "val", "test"))
        if has_named_splits:
            for split_name in ("train", "val", "test"):
                split_root = data_path / split_name
                split_samples.extend(self._collect_split_samples(split_root, split_name))
            if split_samples:
                return split_samples

        recursive_samples: list[_MultimodalSample] = []
        for image_path in sorted(data_path.rglob("image.png")):
            sample_dir = image_path.parent
            audio_path = sample_dir / "audio.wav"
            if not audio_path.exists():
                continue
            class_name = sample_dir.parent.name if sample_dir.parent != data_path else "unknown"
            recursive_samples.append(
                _MultimodalSample(
                    split="inference",
                    class_name=class_name,
                    label_index=self.class_to_index.get(class_name, -1),
                    sample_id=sample_dir.name,
                    image_path=image_path,
                    audio_path=audio_path,
                    meta_path=(sample_dir / "meta.json") if (sample_dir / "meta.json").exists() else None,
                )
            )

        if not recursive_samples:
            raise ValueError(
                f"No paired image/audio samples found for inference under: {data_path}"
            )
        return recursive_samples

    def _build_image_train_transform(self) -> transforms.Compose:
        if not self.config.train_augmentation:
            return self._build_image_eval_transform()

        return transforms.Compose(
            [
                transforms.Resize(self.config.resize_size),
                transforms.RandomResizedCrop(
                    self.config.image_size,
                    scale=(0.78, 1.0),
                    ratio=(0.9, 1.1),
                ),
                transforms.RandomHorizontalFlip(p=0.15),
                transforms.ColorJitter(
                    brightness=0.12,
                    contrast=0.12,
                    saturation=0.08,
                    hue=0.02,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                transforms.RandomErasing(
                    p=max(0.0, min(1.0, float(self.config.random_erasing_probability))),
                    scale=(0.02, 0.08),
                    ratio=(0.3, 3.3),
                    value="random",
                ),
            ]
        )

    def _build_image_eval_transform(self) -> transforms.Compose:
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

    def _build_feature_backbone(self, use_pretrained: bool) -> tuple[nn.Module, int]:
        backbone = build_vit_b_16(
            image_size=self.config.image_size,
            patch_size=self.config.patch_size,
            use_pretrained=use_pretrained,
            pretrained_weights=self.config.pretrained_weights,
        )
        feature_dim = backbone.heads.head.in_features
        backbone.heads = nn.Identity()
        return backbone, feature_dim

    def _build_model(self, num_classes: int, use_pretrained: bool) -> _LateFusionNetwork:
        image_backbone, image_feature_dim = self._build_feature_backbone(use_pretrained)
        audio_backbone, audio_feature_dim = self._build_feature_backbone(use_pretrained)
        if image_feature_dim != audio_feature_dim:
            raise ValueError("Image and audio feature dimensions must match for late fusion.")
        return _LateFusionNetwork(
            image_backbone=image_backbone,
            audio_backbone=audio_backbone,
            feature_dim=image_feature_dim,
            num_classes=num_classes,
            hidden_dim=self.config.fusion_hidden_dim,
            dropout=self.config.fusion_dropout,
            modality_dropout=self.config.modality_dropout,
        )

    def _build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        optimizer_name = self.config.optimizer.lower()
        if not isinstance(model, _LateFusionNetwork):
            raise TypeError("LateFusionModel optimizer expects a _LateFusionNetwork.")
        backbone_params = (
            list(model.image_backbone.parameters()) + list(model.audio_backbone.parameters())
        )
        fusion_params = [
            parameter
            for name, parameter in model.named_parameters()
            if not name.startswith(("image_backbone.", "audio_backbone."))
        ]
        parameter_groups = [
            {
                "params": backbone_params,
                "lr": self.config.learning_rate
                * max(0.0, float(self.config.backbone_lr_scale)),
            },
            {
                "params": fusion_params,
                "lr": self.config.learning_rate,
            },
        ]
        if optimizer_name == "adamw":
            return torch.optim.AdamW(
                parameter_groups,
                weight_decay=self.config.weight_decay,
            )
        if optimizer_name == "sgd":
            return torch.optim.SGD(
                parameter_groups,
                weight_decay=self.config.weight_decay,
                momentum=0.9,
            )
        raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")

    def _set_backbones_trainable(self, trainable: bool) -> None:
        if self.model is None:
            return
        for backbone in (self.model.image_backbone, self.model.audio_backbone):
            for parameter in backbone.parameters():
                parameter.requires_grad = trainable

    def _set_backbones_eval_mode(self) -> None:
        if self.model is None:
            return
        self.model.image_backbone.eval()
        self.model.audio_backbone.eval()

    def _run_epoch(
        self,
        data_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        freeze_backbones: bool = False,
    ) -> dict[str, float]:
        if self.model is None:
            raise RuntimeError("Model is not initialized.")

        self.model.train()
        if freeze_backbones:
            self._set_backbones_eval_mode()
        running_loss = 0.0
        y_true: list[int] = []
        y_pred: list[int] = []

        for image_features, audio_features, labels, *_ in data_loader:
            image_features = self._move_tensor(image_features)
            audio_features = self._move_tensor(audio_features)
            labels = self._move_tensor(labels)

            optimizer.zero_grad()
            logits = self.model(image_features, audio_features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * image_features.size(0)
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
        condition_labels: list[str] = []
        predictions: list[dict[str, Any]] = []

        with torch.no_grad():
            for (
                image_features,
                audio_features,
                labels,
                sample_ids,
                class_names,
                image_paths,
                audio_paths,
            ) in data_loader:
                image_features = self._move_tensor(image_features)
                audio_features = self._move_tensor(audio_features)
                labels = self._move_tensor(labels)

                logits = self.model(image_features, audio_features)
                loss = criterion(logits, labels)
                probabilities = torch.softmax(logits, dim=1)
                top_k = min(self.config.top_k, probabilities.size(1))
                top_probabilities, top_indices = torch.topk(probabilities, k=top_k, dim=1)
                predicted_indices = top_indices[:, 0]

                running_loss += float(loss.item()) * image_features.size(0)
                batch_true = labels.cpu().tolist()
                batch_pred = predicted_indices.cpu().tolist()
                batch_conditions = [
                    self._condition_from_sample_id(str(sample_id)) for sample_id in sample_ids
                ]
                y_true.extend(batch_true)
                y_pred.extend(batch_pred)
                condition_labels.extend(batch_conditions)

                for index, sample_id in enumerate(sample_ids):
                    predictions.append(
                        self._build_prediction_record(
                            sample_id=sample_id,
                            image_path=image_paths[index],
                            audio_path=audio_paths[index],
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
            condition_labels=condition_labels,
        )
        return {"summary": summary, "predictions": predictions}

    def _predict_loader(self, data_loader: DataLoader) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Model is not initialized.")

        self.model.eval()
        y_true: list[int] = []
        y_pred: list[int] = []
        condition_labels: list[str] = []
        predictions: list[dict[str, Any]] = []

        with torch.no_grad():
            for (
                image_features,
                audio_features,
                labels,
                sample_ids,
                class_names,
                image_paths,
                audio_paths,
            ) in data_loader:
                image_features = self._move_tensor(image_features)
                audio_features = self._move_tensor(audio_features)
                logits = self.model(image_features, audio_features)
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
                        condition_labels.append(
                            self._condition_from_sample_id(str(sample_id))
                        )

                    predictions.append(
                        self._build_prediction_record(
                            sample_id=sample_id,
                            image_path=image_paths[index],
                            audio_path=audio_paths[index],
                            input_class_name=class_names[index],
                            ground_truth_index=ground_truth_index,
                            top_indices=top_indices[index].cpu().tolist(),
                            top_probabilities=top_probabilities[index].cpu().tolist(),
                        )
                    )

        summary = None
        if y_true:
            summary = self._build_metrics_summary(
                y_true=y_true,
                y_pred=y_pred,
                average_loss=None,
                condition_labels=condition_labels,
            )
        return {"summary": summary, "predictions": predictions}

    def _build_prediction_record(
        self,
        sample_id: str,
        image_path: str,
        audio_path: str,
        input_class_name: str,
        ground_truth_index: int,
        top_indices: list[int],
        top_probabilities: list[float],
    ) -> dict[str, Any]:
        predicted_index = int(top_indices[0])
        record = {
            "sample_id": sample_id,
            "image_path": image_path,
            "audio_path": audio_path,
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
        condition_labels: list[str] | None = None,
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
        if condition_labels:
            summary["condition_metrics"] = self._build_condition_metrics(
                y_true,
                y_pred,
                condition_labels,
            )
        return summary

    def _condition_from_sample_id(self, sample_id: str) -> str:
        wrapped = f"_{sample_id}_"
        for condition in (
            "clean_sanity",
            "audio_hard",
            "image_hard",
            "both_hard",
            "balanced",
            "train",
        ):
            if f"_{condition}_" in wrapped:
                return condition
        return "unknown"

    def _build_condition_metrics(
        self,
        y_true: list[int],
        y_pred: list[int],
        condition_labels: list[str],
    ) -> dict[str, dict[str, float | int]]:
        metrics: dict[str, dict[str, float | int]] = {}
        for condition in sorted(set(condition_labels)):
            indices = [
                index for index, label in enumerate(condition_labels) if label == condition
            ]
            if not indices:
                continue
            condition_true = [y_true[index] for index in indices]
            condition_pred = [y_pred[index] for index in indices]
            _, _, f1, _ = precision_recall_fscore_support(
                condition_true,
                condition_pred,
                average="macro",
                zero_division=0,
            )
            metrics[condition] = {
                "support": len(indices),
                "accuracy": float(accuracy_score(condition_true, condition_pred)),
                "macro_f1": float(f1),
            }
        return metrics

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
