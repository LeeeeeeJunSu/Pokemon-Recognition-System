from __future__ import annotations

import json
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import soundfile as sf
import torch
import torchaudio
from PIL import Image
from PySide6.QtCore import QThread, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QFont, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from Engine.AudioModel import AudioModel
from Engine.ImageModel import ImageModel
from Engine.LateFusionModel import LateFusionModel
from Engine.MidFusionModel import MidFusionModel
from Engine.ScoreFusionModel import ScoreFusionModel


MODEL_REGISTRY = {
    "ImageModel": ImageModel,
    "AudioModel": AudioModel,
    "LateFusionModel": LateFusionModel,
    "MidFusionModel": MidFusionModel,
    "ScoreFusionModel": ScoreFusionModel,
}

MODEL_INPUT_MODES = {
    "ImageModel": "image",
    "AudioModel": "audio",
    "LateFusionModel": "multimodal",
    "MidFusionModel": "multimodal",
    "ScoreFusionModel": "multimodal",
}


def _now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _slugify(value: str) -> str:
    cleaned = "".join(character if character.isalnum() else "-" for character in value.strip())
    parts = [part for part in cleaned.split("-") if part]
    return "-".join(parts) or "sample"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


def _read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _format_confidence(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return str(value)


def _format_optional_path(path: Path | None) -> str:
    return str(path) if path is not None else "-"


def _audio_info_text(audio_path: Path | None) -> str:
    if audio_path is None:
        return "오디오 입력 없음"
    try:
        info = sf.info(str(audio_path))
        duration = getattr(info, "duration", None)
        lines = [
            f"파일: {audio_path.name}",
            f"샘플레이트: {info.samplerate}Hz",
            f"채널: {info.channels}",
        ]
        if duration is not None:
            lines.append(f"길이: {duration:.2f}s")
        return "\n".join(lines)
    except Exception:
        return f"파일: {audio_path.name}\n메타데이터를 읽지 못했습니다."


@dataclass
class InferenceRun:
    run_id: str
    model_name: str
    checkpoint_path: Path
    result_root: Path
    image_path: Path | None
    audio_path: Path | None
    status: str = "ready"
    message: str = "대기 중"
    created_at: str = field(default_factory=_now_text)
    finished_at: str | None = None
    payload: dict[str, Any] | None = None

    @property
    def predicted_label(self) -> str:
        prediction = (self.payload or {}).get("prediction") or {}
        return str(prediction.get("predicted_display_name", "-"))

    @property
    def predicted_confidence(self) -> str:
        prediction = (self.payload or {}).get("prediction") or {}
        return _format_confidence(prediction.get("confidence"))


class InferenceWorker(QThread):
    progress_changed = Signal(str, str)
    finished_run = Signal(str, bool, object)

    def __init__(self, run: InferenceRun) -> None:
        super().__init__()
        self.run_info = run

    def run(self) -> None:
        try:
            self.progress_changed.emit(self.run_info.run_id, "입력 샘플을 준비하는 중입니다.")
            input_root = self._prepare_input_payload()

            request_payload = {
                "run_id": self.run_info.run_id,
                "model_name": self.run_info.model_name,
                "checkpoint_path": str(self.run_info.checkpoint_path),
                "image_path": _format_optional_path(self.run_info.image_path),
                "audio_path": _format_optional_path(self.run_info.audio_path),
                "input_root": str(input_root),
                "created_at": self.run_info.created_at,
            }
            _write_json(
                self.run_info.result_root / "logs" / "inference_request.json",
                request_payload,
            )

            self.progress_changed.emit(self.run_info.run_id, "체크포인트를 로드하는 중입니다.")
            model = MODEL_REGISTRY[self.run_info.model_name]()
            model.Load(self.run_info.checkpoint_path)

            self.progress_changed.emit(self.run_info.run_id, "파일 기반 추론을 실행하는 중입니다.")
            model.Inference(input_root, self.run_info.result_root)

            self.progress_changed.emit(self.run_info.run_id, "결과를 불러오는 중입니다.")
            payload = self._load_result_payload()
            self.finished_run.emit(self.run_info.run_id, True, payload)
        except Exception:
            error_text = traceback.format_exc()
            payload = {"error": error_text}
            _write_json(
                self.run_info.result_root / "logs" / "inference_error.json",
                payload,
            )
            self.finished_run.emit(self.run_info.run_id, False, payload)

    def _prepare_input_payload(self) -> Path:
        input_root = self.run_info.result_root / "input_payload" / "sample_000"
        input_root.mkdir(parents=True, exist_ok=True)

        input_mode = MODEL_INPUT_MODES[self.run_info.model_name]
        if input_mode in {"image", "multimodal"}:
            if self.run_info.image_path is None:
                raise ValueError("이미지 입력이 필요한 모델입니다.")
            self._prepare_image_file(self.run_info.image_path, input_root / "image.png")

        if input_mode in {"audio", "multimodal"}:
            if self.run_info.audio_path is None:
                raise ValueError("오디오 입력이 필요한 모델입니다.")
            self._prepare_audio_file(self.run_info.audio_path, input_root / "audio.wav")

        meta = {
            "mode": "file_test_gui",
            "model_name": self.run_info.model_name,
            "original_image_path": str(self.run_info.image_path) if self.run_info.image_path else None,
            "original_audio_path": str(self.run_info.audio_path) if self.run_info.audio_path else None,
            "prepared_at": _now_text(),
        }
        _write_json(input_root / "meta.json", meta)
        return input_root

    def _prepare_image_file(self, src: Path, dst: Path) -> None:
        with Image.open(src) as image:
            image = image.convert("RGB")
            image.save(dst, format="PNG")

    def _prepare_audio_file(self, src: Path, dst: Path) -> None:
        try:
            audio, sample_rate = sf.read(str(src), always_2d=True, dtype="float32")
            waveform = torch.from_numpy(audio.T.copy())
            torchaudio.save(str(dst), waveform, sample_rate)
            return
        except Exception:
            pass

        waveform, sample_rate = torchaudio.load(str(src))
        torchaudio.save(str(dst), waveform, sample_rate)

    def _load_result_payload(self) -> dict[str, Any]:
        summary = _read_json(self.run_info.result_root / "logs" / "inference_summary.json") or {}
        predictions = (
            _read_json(self.run_info.result_root / "predictions" / "inference_predictions.json")
            or []
        )
        metrics = _read_json(self.run_info.result_root / "metrics" / "inference_metrics.json")
        return {
            "summary": summary,
            "predictions": predictions,
            "prediction": predictions[0] if predictions else None,
            "metrics": metrics,
        }


class FileTestWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Pokemon Recognition File Tester")
        self.resize(1500, 930)

        self.default_result_base = PROJECT_ROOT / "artifacts" / "file_test_runs"
        self.active_worker: InferenceWorker | None = None
        self.current_run_id: str | None = None
        self.selected_run_id: str | None = None
        self.runs: list[InferenceRun] = []
        self.run_sequence = 0

        self._build_ui()
        self._apply_styles()
        self._update_input_mode()
        self._refresh_run_table()
        self._refresh_result_panel()
        self._append_log("파일 기반 테스트 앱이 준비되었습니다.")

    def _build_ui(self) -> None:
        central = QWidget(self)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(20, 20, 20, 20)
        root_layout.setSpacing(16)

        title_label = QLabel("File Test Studio")
        title_label.setObjectName("TitleLabel")
        subtitle_label = QLabel(
            "체크포인트와 입력 파일을 고르면, 엔진 규격에 맞는 임시 샘플을 만들어 바로 추론합니다."
        )
        subtitle_label.setObjectName("SubtitleLabel")
        root_layout.addWidget(title_label)
        root_layout.addWidget(subtitle_label)

        controls_card = QFrame()
        controls_card.setObjectName("Card")
        controls_layout = QGridLayout(controls_card)
        controls_layout.setHorizontalSpacing(12)
        controls_layout.setVerticalSpacing(12)

        self.model_combo = QComboBox()
        self.model_combo.addItems(MODEL_REGISTRY.keys())
        self.model_combo.currentTextChanged.connect(self._update_input_mode)

        self.checkpoint_edit = QLineEdit()
        checkpoint_button = QPushButton("체크포인트 선택")
        checkpoint_button.clicked.connect(self._choose_checkpoint)

        self.result_base_edit = QLineEdit(str(self.default_result_base))
        result_base_button = QPushButton("결과 위치")
        result_base_button.clicked.connect(self._choose_result_base)

        self.image_edit = QLineEdit()
        self.image_button = QPushButton("이미지 선택")
        self.image_button.clicked.connect(self._choose_image)

        self.audio_edit = QLineEdit()
        self.audio_button = QPushButton("오디오 선택")
        self.audio_button.clicked.connect(self._choose_audio)

        self.run_button = QPushButton("파일 기반 추론 실행")
        self.run_button.setObjectName("PrimaryButton")
        self.run_button.clicked.connect(self._start_inference)

        self.open_result_button = QPushButton("결과 폴더 열기")
        self.open_result_button.clicked.connect(self._open_selected_result_folder)

        controls_layout.addWidget(QLabel("모델 선택"), 0, 0)
        controls_layout.addWidget(self.model_combo, 0, 1)
        controls_layout.addWidget(self.run_button, 0, 2)
        controls_layout.addWidget(QLabel("체크포인트"), 1, 0)
        controls_layout.addWidget(self.checkpoint_edit, 1, 1)
        controls_layout.addWidget(checkpoint_button, 1, 2)
        controls_layout.addWidget(QLabel("결과 루트"), 2, 0)
        controls_layout.addWidget(self.result_base_edit, 2, 1)
        controls_layout.addWidget(result_base_button, 2, 2)
        controls_layout.addWidget(QLabel("이미지 파일"), 3, 0)
        controls_layout.addWidget(self.image_edit, 3, 1)
        controls_layout.addWidget(self.image_button, 3, 2)
        controls_layout.addWidget(QLabel("오디오 파일"), 4, 0)
        controls_layout.addWidget(self.audio_edit, 4, 1)
        controls_layout.addWidget(self.audio_button, 4, 2)
        controls_layout.addWidget(self.open_result_button, 5, 2)

        root_layout.addWidget(controls_card)

        status_card = QFrame()
        status_card.setObjectName("Card")
        status_layout = QGridLayout(status_card)
        status_layout.setHorizontalSpacing(18)
        status_layout.setVerticalSpacing(8)

        self.active_run_label = QLabel("실행 중인 작업 없음")
        self.status_label = QLabel("Idle")
        self.result_path_label = QLabel("-")
        self.result_path_label.setWordWrap(True)
        self.message_label = QLabel("입력을 선택해 추론을 시작하세요.")
        self.message_label.setWordWrap(True)

        status_layout.addWidget(QLabel("현재 실행"), 0, 0)
        status_layout.addWidget(self.active_run_label, 0, 1)
        status_layout.addWidget(QLabel("상태"), 0, 2)
        status_layout.addWidget(self.status_label, 0, 3)
        status_layout.addWidget(QLabel("결과 폴더"), 1, 0)
        status_layout.addWidget(self.result_path_label, 1, 1, 1, 3)
        status_layout.addWidget(QLabel("메시지"), 2, 0)
        status_layout.addWidget(self.message_label, 2, 1, 1, 3)

        root_layout.addWidget(status_card)

        splitter = QSplitter(Qt.Horizontal)

        left_panel = QFrame()
        left_panel.setObjectName("Card")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(16, 16, 16, 16)
        left_layout.setSpacing(12)

        preview_group = QGroupBox("입력 프리뷰")
        preview_layout = QVBoxLayout(preview_group)
        self.image_preview = QLabel("이미지 미리보기 없음")
        self.image_preview.setFixedSize(420, 300)
        self.image_preview.setAlignment(Qt.AlignCenter)
        self.image_preview.setObjectName("PreviewBox")
        self.audio_info_label = QLabel("오디오 입력 없음")
        self.audio_info_label.setWordWrap(True)
        preview_layout.addWidget(self.image_preview)
        preview_layout.addWidget(self.audio_info_label)
        left_layout.addWidget(preview_group)

        history_group = QGroupBox("최근 실행 기록")
        history_layout = QVBoxLayout(history_group)
        self.run_table = QTableWidget(0, 6)
        self.run_table.setHorizontalHeaderLabels(
            ["실행", "모델", "상태", "예측", "신뢰도", "결과 폴더"]
        )
        self.run_table.verticalHeader().setVisible(False)
        self.run_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.run_table.setSelectionMode(QTableWidget.SingleSelection)
        self.run_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.run_table.horizontalHeader().setStretchLastSection(True)
        self.run_table.itemSelectionChanged.connect(self._on_run_selection_changed)
        history_layout.addWidget(self.run_table)
        left_layout.addWidget(history_group)
        splitter.addWidget(left_panel)

        detail_tabs = QTabWidget()

        summary_tab = QWidget()
        summary_layout = QVBoxLayout(summary_tab)
        summary_group = QGroupBox("추론 요약")
        summary_form = QFormLayout(summary_group)
        self.summary_labels: dict[str, QLabel] = {}
        for key, title in (
            ("run_id", "실행 ID"),
            ("model_name", "모델"),
            ("checkpoint_path", "체크포인트"),
            ("status", "상태"),
            ("image_path", "입력 이미지"),
            ("audio_path", "입력 오디오"),
            ("result_root", "결과 폴더"),
            ("predicted_label", "예측 클래스"),
            ("confidence", "신뢰도"),
            ("image_branch", "이미지 브랜치"),
            ("audio_branch", "오디오 브랜치"),
            ("message", "메시지"),
        ):
            label = QLabel("-")
            label.setWordWrap(True)
            self.summary_labels[key] = label
            summary_form.addRow(title, label)
        summary_layout.addWidget(summary_group)
        summary_layout.addStretch(1)
        detail_tabs.addTab(summary_tab, "요약")

        topk_tab = QWidget()
        topk_layout = QVBoxLayout(topk_tab)
        self.topk_table = QTableWidget(0, 4)
        self.topk_table.setHorizontalHeaderLabels(["Rank", "Label", "Class", "Confidence"])
        self.topk_table.verticalHeader().setVisible(False)
        self.topk_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.topk_table.horizontalHeader().setStretchLastSection(True)
        topk_layout.addWidget(self.topk_table)
        detail_tabs.addTab(topk_tab, "Top-K")

        json_tab = QWidget()
        json_layout = QVBoxLayout(json_tab)
        self.json_view = QPlainTextEdit()
        self.json_view.setReadOnly(True)
        json_layout.addWidget(self.json_view)
        detail_tabs.addTab(json_tab, "결과 JSON")

        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        log_layout.addWidget(self.log_view)
        detail_tabs.addTab(log_tab, "이벤트 로그")

        splitter.addWidget(detail_tabs)
        splitter.setSizes([630, 830])
        root_layout.addWidget(splitter, 1)

        self.setCentralWidget(central)

    def _apply_styles(self) -> None:
        self.setFont(QFont("Segoe UI", 10))
        self.setStyleSheet(
            """
            QMainWindow {
                background: #f3eee4;
            }
            QLabel#TitleLabel {
                font-size: 28px;
                font-weight: 700;
                color: #19334d;
            }
            QLabel#SubtitleLabel {
                color: #5d6f80;
            }
            QFrame#Card {
                background: #fffdf7;
                border: 1px solid #ded4c3;
                border-radius: 18px;
            }
            QGroupBox {
                border: 1px solid #e2d7c6;
                border-radius: 14px;
                margin-top: 12px;
                padding-top: 10px;
                font-weight: 600;
                color: #294257;
                background: #fffaf1;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px 0 6px;
            }
            QLabel#PreviewBox {
                background: #f7f0e5;
                border: 1px dashed #c6b6a0;
                border-radius: 16px;
                color: #536273;
            }
            QLineEdit, QComboBox, QPlainTextEdit, QTableWidget {
                background: #fffefb;
                border: 1px solid #d6cab8;
                border-radius: 10px;
                padding: 6px;
                color: #243241;
            }
            QHeaderView::section {
                background: #e9ded0;
                color: #314456;
                border: none;
                border-right: 1px solid #d9c9b5;
                padding: 8px;
                font-weight: 600;
            }
            QTableWidget {
                gridline-color: #e7dccb;
                selection-background-color: #d7ebe8;
                selection-color: #183040;
            }
            QPushButton {
                background: #e6dccb;
                border: 1px solid #cfbea5;
                border-radius: 12px;
                padding: 8px 14px;
                color: #263646;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #dcd0bd;
            }
            QPushButton#PrimaryButton {
                background: #0e7c86;
                border-color: #0a6770;
                color: white;
            }
            QPushButton#PrimaryButton:hover {
                background: #0d6f78;
            }
            QTabWidget::pane {
                border: 1px solid #ddcfbc;
                border-radius: 12px;
                background: #fffdf7;
            }
            QTabBar::tab {
                background: #efe4d4;
                border: 1px solid #decdb7;
                border-bottom: none;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                padding: 8px 14px;
                margin-right: 4px;
                color: #354658;
            }
            QTabBar::tab:selected {
                background: #fffdf7;
                color: #183040;
            }
            """
        )

    def _choose_checkpoint(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "체크포인트 선택",
            str(PROJECT_ROOT / "artifacts"),
            "PyTorch Checkpoint (*.pt *.pth);;All Files (*.*)",
        )
        if file_path:
            self.checkpoint_edit.setText(file_path)

    def _choose_result_base(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "결과 루트 선택",
            self.result_base_edit.text() or str(self.default_result_base),
        )
        if directory:
            self.result_base_edit.setText(directory)

    def _choose_image(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "이미지 파일 선택",
            str(PROJECT_ROOT),
            "Image Files (*.png *.jpg *.jpeg *.bmp *.webp);;All Files (*.*)",
        )
        if file_path:
            self.image_edit.setText(file_path)
            self._update_preview_widgets()

    def _choose_audio(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "오디오 파일 선택",
            str(PROJECT_ROOT),
            "Audio Files (*.wav *.flac *.mp3 *.ogg *.m4a);;All Files (*.*)",
        )
        if file_path:
            self.audio_edit.setText(file_path)
            self._update_preview_widgets()

    def _update_input_mode(self) -> None:
        mode = MODEL_INPUT_MODES[self.model_combo.currentText()]
        needs_image = mode in {"image", "multimodal"}
        needs_audio = mode in {"audio", "multimodal"}

        self.image_edit.setEnabled(needs_image)
        self.image_button.setEnabled(needs_image)
        self.audio_edit.setEnabled(needs_audio)
        self.audio_button.setEnabled(needs_audio)

        if not needs_image:
            self.image_edit.clear()
        if not needs_audio:
            self.audio_edit.clear()
        self._update_preview_widgets()

    def _update_preview_widgets(self) -> None:
        image_path = Path(self.image_edit.text()).expanduser() if self.image_edit.text().strip() else None
        audio_path = Path(self.audio_edit.text()).expanduser() if self.audio_edit.text().strip() else None

        if image_path is not None and image_path.exists():
            pixmap = QPixmap(str(image_path))
            if pixmap.isNull():
                self.image_preview.setText("이미지 미리보기를 불러오지 못했습니다.")
            else:
                self.image_preview.setText("")
                self.image_preview.setPixmap(
                    pixmap.scaled(
                        self.image_preview.size(),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation,
                    )
                )
        else:
            self.image_preview.setPixmap(QPixmap())
            self.image_preview.setText("이미지 미리보기 없음")

        self.audio_info_label.setText(_audio_info_text(audio_path))

    def _start_inference(self) -> None:
        if self.active_worker is not None and self.active_worker.isRunning():
            QMessageBox.information(self, "실행 중", "현재 추론이 진행 중입니다. 완료 후 다시 시도해 주세요.")
            return

        model_name = self.model_combo.currentText()
        checkpoint_path = Path(self.checkpoint_edit.text()).expanduser() if self.checkpoint_edit.text().strip() else None
        image_path = Path(self.image_edit.text()).expanduser() if self.image_edit.text().strip() else None
        audio_path = Path(self.audio_edit.text()).expanduser() if self.audio_edit.text().strip() else None
        result_base = Path(self.result_base_edit.text()).expanduser() if self.result_base_edit.text().strip() else self.default_result_base

        if checkpoint_path is None or not checkpoint_path.exists():
            QMessageBox.warning(self, "체크포인트 오류", "유효한 체크포인트 파일을 선택해 주세요.")
            return

        mode = MODEL_INPUT_MODES[model_name]
        if mode in {"image", "multimodal"} and (image_path is None or not image_path.exists()):
            QMessageBox.warning(self, "입력 오류", "이 모델은 이미지 입력이 필요합니다.")
            return
        if mode in {"audio", "multimodal"} and (audio_path is None or not audio_path.exists()):
            QMessageBox.warning(self, "입력 오류", "이 모델은 오디오 입력이 필요합니다.")
            return

        self.run_sequence += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sample_slug = _slugify((image_path or audio_path or checkpoint_path).stem)
        result_root = (
            result_base.resolve()
            / f"{timestamp}_{self.run_sequence:03d}_{model_name}_{sample_slug}"
        )

        run = InferenceRun(
            run_id=f"RUN-{self.run_sequence:03d}",
            model_name=model_name,
            checkpoint_path=checkpoint_path.resolve(),
            result_root=result_root,
            image_path=image_path.resolve() if image_path else None,
            audio_path=audio_path.resolve() if audio_path else None,
            status="running",
            message="추론 준비 중",
        )
        self.runs.insert(0, run)
        self.current_run_id = run.run_id
        self.selected_run_id = run.run_id
        self._append_log(f"{run.run_id} | {run.model_name} | 추론 시작")
        self._refresh_run_table()
        self._refresh_status_card(run)
        self._refresh_result_panel()

        self.active_worker = InferenceWorker(run)
        self.active_worker.progress_changed.connect(self._on_worker_progress)
        self.active_worker.finished_run.connect(self._on_worker_finished)
        self.active_worker.start()

    def _on_worker_progress(self, run_id: str, message: str) -> None:
        run = self._find_run(run_id)
        if run is None:
            return
        run.status = "running"
        run.message = message
        if run_id == self.current_run_id:
            self._refresh_status_card(run)
        if run_id == self.selected_run_id:
            self._refresh_result_panel()
        self._append_log(f"{run.run_id} | {message}")

    def _on_worker_finished(self, run_id: str, success: bool, payload: object) -> None:
        run = self._find_run(run_id)
        if run is None:
            return

        run.finished_at = _now_text()
        run.status = "completed" if success else "failed"
        run.message = "추론이 완료되었습니다." if success else "추론이 실패했습니다."
        run.payload = payload if isinstance(payload, dict) else {"raw": payload}

        if not success:
            self._append_log(f"{run.run_id} | 실패")
            error_text = str((run.payload or {}).get("error", "알 수 없는 오류"))
            self._append_log(error_text)
        else:
            self._append_log(f"{run.run_id} | 완료 | {run.predicted_label}")

        if self.active_worker is not None:
            self.active_worker.deleteLater()
            self.active_worker = None
        self.current_run_id = None

        self._refresh_run_table()
        self._refresh_status_card(self._find_run(self.selected_run_id))
        self._refresh_result_panel()

    def _refresh_status_card(self, run: InferenceRun | None) -> None:
        if run is None:
            self.active_run_label.setText("실행 중인 작업 없음")
            self.status_label.setText("Idle")
            self.result_path_label.setText("-")
            self.message_label.setText("입력을 선택해 추론을 시작하세요.")
            return

        self.active_run_label.setText(f"{run.run_id} · {run.model_name}")
        self.status_label.setText(run.status)
        self.result_path_label.setText(str(run.result_root))
        self.message_label.setText(run.message)

    def _refresh_run_table(self) -> None:
        self.run_table.setRowCount(len(self.runs))
        for row, run in enumerate(self.runs):
            values = [
                run.run_id,
                run.model_name,
                run.status,
                run.predicted_label,
                run.predicted_confidence,
                str(run.result_root),
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 0:
                    item.setData(Qt.UserRole, run.run_id)
                if column in (0, 1, 2, 4):
                    item.setTextAlignment(Qt.AlignCenter)
                self.run_table.setItem(row, column, item)
        self.run_table.resizeColumnsToContents()

        if self.selected_run_id is not None:
            for row in range(self.run_table.rowCount()):
                item = self.run_table.item(row, 0)
                if item is not None and item.data(Qt.UserRole) == self.selected_run_id:
                    self.run_table.selectRow(row)
                    break

    def _refresh_result_panel(self) -> None:
        run = self._find_run(self.selected_run_id) or self._find_run(self.current_run_id)
        if run is None:
            for label in self.summary_labels.values():
                label.setText("-")
            self.topk_table.setRowCount(0)
            self.json_view.setPlainText("실행 결과가 없습니다.")
            return

        payload = run.payload or {}
        prediction = payload.get("prediction") or {}
        image_branch = prediction.get("image_branch") or {}
        audio_branch = prediction.get("audio_branch") or {}

        self.summary_labels["run_id"].setText(run.run_id)
        self.summary_labels["model_name"].setText(run.model_name)
        self.summary_labels["checkpoint_path"].setText(str(run.checkpoint_path))
        self.summary_labels["status"].setText(run.status)
        self.summary_labels["image_path"].setText(_format_optional_path(run.image_path))
        self.summary_labels["audio_path"].setText(_format_optional_path(run.audio_path))
        self.summary_labels["result_root"].setText(str(run.result_root))
        self.summary_labels["predicted_label"].setText(str(prediction.get("predicted_display_name", "-")))
        self.summary_labels["confidence"].setText(_format_confidence(prediction.get("confidence")))
        self.summary_labels["image_branch"].setText(
            self._branch_text(image_branch)
        )
        self.summary_labels["audio_branch"].setText(
            self._branch_text(audio_branch)
        )
        self.summary_labels["message"].setText(run.message)

        top_k = prediction.get("top_k") or []
        self.topk_table.setRowCount(len(top_k))
        for row, item in enumerate(top_k):
            values = [
                str(row + 1),
                str(item.get("display_name", "-")),
                str(item.get("class_name", "-")),
                _format_confidence(item.get("confidence")),
            ]
            for column, value in enumerate(values):
                table_item = QTableWidgetItem(value)
                table_item.setTextAlignment(Qt.AlignCenter)
                self.topk_table.setItem(row, column, table_item)
        self.topk_table.resizeColumnsToContents()

        self.json_view.setPlainText(json.dumps(payload, ensure_ascii=False, indent=2))

        # Selecting a historical run also restores the preview context.
        if run.image_path and run.image_path.exists():
            pixmap = QPixmap(str(run.image_path))
            if not pixmap.isNull():
                self.image_preview.setText("")
                self.image_preview.setPixmap(
                    pixmap.scaled(
                        self.image_preview.size(),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation,
                    )
                )
        elif MODEL_INPUT_MODES[run.model_name] in {"image", "multimodal"}:
            self.image_preview.setPixmap(QPixmap())
            self.image_preview.setText("이미지 미리보기 없음")
        self.audio_info_label.setText(_audio_info_text(run.audio_path))

    def _branch_text(self, branch: dict[str, Any]) -> str:
        if not branch:
            return "-"
        label = branch.get("predicted_display_name") or branch.get("predicted_class_name") or "-"
        confidence = _format_confidence(branch.get("confidence"))
        return f"{label} ({confidence})"

    def _append_log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_view.appendPlainText(f"[{timestamp}] {message}")

    def _on_run_selection_changed(self) -> None:
        row = self.run_table.currentRow()
        if row < 0:
            return
        item = self.run_table.item(row, 0)
        if item is None:
            return
        self.selected_run_id = item.data(Qt.UserRole)
        run = self._find_run(self.selected_run_id)
        self._refresh_status_card(run if run is not None else self._find_run(self.current_run_id))
        self._refresh_result_panel()

    def _open_selected_result_folder(self) -> None:
        run = self._find_run(self.selected_run_id) or self._find_run(self.current_run_id)
        if run is None:
            QMessageBox.information(self, "결과 없음", "먼저 추론을 실행해 주세요.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(run.result_root)))

    def _find_run(self, run_id: str | None) -> InferenceRun | None:
        if run_id is None:
            return None
        return next((run for run in self.runs if run.run_id == run_id), None)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self.active_worker is not None and self.active_worker.isRunning():
            response = QMessageBox.question(
                self,
                "추론 진행 중",
                "지금 닫으면 진행 중인 추론이 중단될 수 있습니다. 정말 닫을까요?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if response != QMessageBox.Yes:
                event.ignore()
                return
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    window = FileTestWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
