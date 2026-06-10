from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from PIL import Image
from PySide6.QtCore import QThread, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QFont, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

try:
    import cv2
except ImportError:  # pragma: no cover - handled in the GUI at runtime.
    cv2 = None

try:
    import sounddevice as sd
except ImportError:  # pragma: no cover - handled in the GUI at runtime.
    sd = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


def _timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


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


def _confidence_int(value: Any) -> int:
    try:
        return max(0, min(10000, int(float(value) * 10000)))
    except (TypeError, ValueError):
        return 0


def _guess_model_name_from_path(path: Path) -> str | None:
    lower_name = path.name.lower()
    for model_name in MODEL_REGISTRY:
        if model_name.lower() in lower_name:
            return model_name
    return None


@dataclass
class CheckpointOption:
    model_name: str
    checkpoint_path: Path
    run_root: Path
    checkpoint_kind: str
    display_text: str
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class RealtimeSessionConfig:
    model_name: str
    checkpoint_path: Path
    result_root: Path
    camera_index: int
    inference_interval_seconds: float
    audio_window_seconds: float
    preview_fps: float = 12.0


class AudioRingBuffer:
    def __init__(self, sample_rate: int, max_seconds: float) -> None:
        self.sample_rate = sample_rate
        self.max_samples = max(1, int(sample_rate * max_seconds))
        self._chunks: deque[np.ndarray] = deque()
        self._total_samples = 0
        self._lock = threading.Lock()
        self.last_status: str | None = None

    def callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        if status:
            self.last_status = str(status)
        mono = np.asarray(indata[:, 0], dtype=np.float32).copy()
        with self._lock:
            self._chunks.append(mono)
            self._total_samples += mono.shape[0]
            while self._total_samples > self.max_samples and self._chunks:
                removed = self._chunks.popleft()
                self._total_samples -= removed.shape[0]

    def latest(self, seconds: float) -> np.ndarray:
        target_samples = max(1, int(self.sample_rate * seconds))
        with self._lock:
            if not self._chunks:
                return np.zeros(target_samples, dtype=np.float32)
            audio = np.concatenate(list(self._chunks)).astype(np.float32, copy=False)

        if audio.shape[0] >= target_samples:
            return audio[-target_samples:]

        padded = np.zeros(target_samples, dtype=np.float32)
        padded[-audio.shape[0] :] = audio
        return padded

    def available_seconds(self) -> float:
        with self._lock:
            return self._total_samples / float(self.sample_rate)


class RealtimeInferenceWorker(QThread):
    status_changed = Signal(str)
    frame_ready = Signal(object)
    prediction_ready = Signal(object)
    failed = Signal(str)
    stopped = Signal()

    def __init__(self, session_config: RealtimeSessionConfig) -> None:
        super().__init__()
        self.session_config = session_config
        self._stop_event = threading.Event()

    def request_stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        camera = None
        audio_stream = None

        try:
            model_name = self.session_config.model_name
            input_mode = MODEL_INPUT_MODES[model_name]
            needs_image = input_mode in {"image", "multimodal"}
            needs_audio = input_mode in {"audio", "multimodal"}

            if needs_image and cv2 is None:
                raise RuntimeError("opencv-python 패키지가 필요합니다. `pip install opencv-python` 후 다시 실행하세요.")
            if needs_audio and sd is None:
                raise RuntimeError("sounddevice 패키지가 필요합니다. `pip install sounddevice` 후 다시 실행하세요.")

            self.status_changed.emit("모델을 로드하는 중입니다.")
            model = MODEL_REGISTRY[model_name]()
            model.Load(self.session_config.checkpoint_path)
            self._optimize_model_for_single_sample(model)

            sample_rate = int(getattr(model.config, "sample_rate", 22050))
            audio_window_seconds = self.session_config.audio_window_seconds
            audio_buffer: AudioRingBuffer | None = None

            self.session_config.result_root.mkdir(parents=True, exist_ok=True)
            _write_json(
                self.session_config.result_root / "logs" / "session_config.json",
                {
                    **asdict(self.session_config),
                    "checkpoint_path": str(self.session_config.checkpoint_path),
                    "result_root": str(self.session_config.result_root),
                    "started_at": _now_text(),
                    "input_mode": input_mode,
                    "sample_rate": sample_rate if needs_audio else None,
                },
            )

            if needs_image:
                self.status_changed.emit("카메라를 여는 중입니다.")
                camera = self._open_camera(self.session_config.camera_index)

            if needs_audio:
                self.status_changed.emit("마이크 입력을 여는 중입니다.")
                audio_buffer = AudioRingBuffer(
                    sample_rate=sample_rate,
                    max_seconds=max(audio_window_seconds * 2.0, audio_window_seconds + 1.0),
                )
                audio_stream = sd.InputStream(
                    samplerate=sample_rate,
                    channels=1,
                    dtype="float32",
                    callback=audio_buffer.callback,
                )
                audio_stream.start()

            latest_frame: np.ndarray | None = None
            last_preview_at = 0.0
            next_inference_at = time.monotonic()
            tick_index = 0
            self.status_changed.emit("실시간 추론을 시작했습니다.")

            while not self._stop_event.is_set():
                now = time.monotonic()

                if camera is not None:
                    ok, frame = camera.read()
                    if ok:
                        latest_frame = frame
                        if now - last_preview_at >= 1.0 / max(self.session_config.preview_fps, 1.0):
                            self.frame_ready.emit(self._encode_preview(frame))
                            last_preview_at = now

                if now >= next_inference_at:
                    if needs_image and latest_frame is None:
                        self.status_changed.emit("카메라 프레임을 기다리는 중입니다.")
                        time.sleep(0.03)
                        continue
                    if needs_audio and audio_buffer is None:
                        raise RuntimeError("마이크 버퍼가 초기화되지 않았습니다.")

                    tick_index += 1
                    started_at = time.perf_counter()
                    self.status_changed.emit(f"{tick_index}번째 추론을 실행하는 중입니다.")
                    payload = self._run_one_tick(
                        model=model,
                        tick_index=tick_index,
                        latest_frame=latest_frame,
                        audio_buffer=audio_buffer,
                        sample_rate=sample_rate,
                        audio_window_seconds=audio_window_seconds,
                    )
                    payload["latency_seconds"] = time.perf_counter() - started_at
                    payload["completed_at"] = _now_text()
                    self.prediction_ready.emit(payload)
                    next_inference_at = time.monotonic() + self.session_config.inference_interval_seconds

                time.sleep(0.01)

        except Exception:
            self.failed.emit(traceback.format_exc())
        finally:
            if audio_stream is not None:
                try:
                    audio_stream.stop()
                    audio_stream.close()
                except Exception:
                    pass
            if camera is not None:
                camera.release()
            self.stopped.emit()

    def _open_camera(self, camera_index: int) -> Any:
        assert cv2 is not None
        backends = [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF), ("ANY", 0)] if sys.platform.startswith("win") else [("ANY", 0)]
        camera_indices = self._candidate_camera_indices(camera_index)
        errors: list[str] = []

        for candidate_index in camera_indices:
            for backend_name, backend in backends:
                camera = cv2.VideoCapture(candidate_index, backend)
                if not camera.isOpened():
                    errors.append(f"index={candidate_index}, backend={backend_name}: open failed")
                    camera.release()
                    continue

                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                frame_ok = False
                frame = None
                for _ in range(5):
                    frame_ok, frame = camera.read()
                    if frame_ok:
                        break
                    time.sleep(0.05)

                if frame_ok and self._is_usable_camera_frame(frame):
                    self.status_changed.emit(f"카메라 index={candidate_index}, backend={backend_name}를 사용합니다.")
                    return camera

                reason = "black frame" if frame_ok else "frame read failed"
                errors.append(f"index={candidate_index}, backend={backend_name}: {reason}")
                camera.release()

        requested = "자동" if camera_index < 0 else str(camera_index)
        detail = "\n".join(errors[-12:])
        raise RuntimeError(f"카메라를 열 수 없습니다. 요청={requested}\n{detail}")

    def _candidate_camera_indices(self, camera_index: int) -> list[int]:
        fallback_indices = [1, 0, 2, 3, 4, 5, 6, 7]
        if camera_index < 0:
            return fallback_indices
        return [camera_index] + [index for index in fallback_indices if index != camera_index]

    def _is_usable_camera_frame(self, frame: Any) -> bool:
        if frame is None:
            return False
        array = np.asarray(frame)
        if array.size == 0:
            return False
        gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY) if array.ndim == 3 else array
        return not (float(gray.mean()) < 3.0 and float(gray.std()) < 3.0)

    def _optimize_model_for_single_sample(self, model: Any) -> None:
        if hasattr(model, "config"):
            if hasattr(model.config, "batch_size"):
                model.config.batch_size = 1
            if hasattr(model.config, "num_workers"):
                model.config.num_workers = 0

    def _run_one_tick(
        self,
        model: Any,
        tick_index: int,
        latest_frame: np.ndarray | None,
        audio_buffer: AudioRingBuffer | None,
        sample_rate: int,
        audio_window_seconds: float,
    ) -> dict[str, Any]:
        tick_root = self.session_config.result_root / "ticks" / f"tick_{tick_index:06d}"
        input_root = tick_root / "input_payload" / "sample_000"
        result_root = tick_root / "result"
        input_root.mkdir(parents=True, exist_ok=True)
        result_root.mkdir(parents=True, exist_ok=True)

        input_mode = MODEL_INPUT_MODES[self.session_config.model_name]
        needs_image = input_mode in {"image", "multimodal"}
        needs_audio = input_mode in {"audio", "multimodal"}

        image_path: Path | None = None
        audio_path: Path | None = None

        if needs_image:
            if latest_frame is None:
                raise RuntimeError("카메라 프레임이 없습니다.")
            image_path = input_root / "image.png"
            self._save_frame(latest_frame, image_path)

        if needs_audio:
            if audio_buffer is None:
                raise RuntimeError("마이크 버퍼가 없습니다.")
            audio_path = input_root / "audio.wav"
            audio = audio_buffer.latest(audio_window_seconds)
            sf.write(str(audio_path), audio, sample_rate, subtype="PCM_16")

        meta = {
            "mode": "realtime_inference_gui",
            "model_name": self.session_config.model_name,
            "tick_index": tick_index,
            "captured_at": _now_text(),
            "image_path": str(image_path) if image_path else None,
            "audio_path": str(audio_path) if audio_path else None,
            "audio_window_seconds": audio_window_seconds if needs_audio else None,
            "audio_available_seconds": audio_buffer.available_seconds() if audio_buffer else None,
        }
        _write_json(input_root / "meta.json", meta)

        model.Inference(input_root, result_root)

        predictions = _read_json(result_root / "predictions" / "inference_predictions.json") or []
        summary = _read_json(result_root / "logs" / "inference_summary.json") or {}
        prediction = predictions[0] if predictions else {}
        payload = {
            "tick_index": tick_index,
            "input_root": str(input_root),
            "result_root": str(result_root),
            "summary": summary,
            "prediction": prediction,
            "top_k": prediction.get("top_k", []),
            "image_branch": prediction.get("image_branch"),
            "audio_branch": prediction.get("audio_branch"),
            "meta": meta,
        }
        _write_json(tick_root / "realtime_prediction.json", payload)
        return payload

    def _save_frame(self, frame: np.ndarray, path: Path) -> None:
        assert cv2 is not None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        Image.fromarray(rgb).save(path, format="PNG")

    def _encode_preview(self, frame: np.ndarray) -> bytes:
        assert cv2 is not None
        preview = frame
        height, width = preview.shape[:2]
        if width > 960:
            scale = 960.0 / float(width)
            preview = cv2.resize(preview, (960, int(height * scale)))
        ok, encoded = cv2.imencode(".jpg", preview, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if not ok:
            return b""
        return encoded.tobytes()


class RealtimeInferenceWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Pokemon Recognition Realtime Inference")
        self.resize(1540, 940)

        self.default_result_base = PROJECT_ROOT / "artifacts" / "realtime_runs"
        self.worker: RealtimeInferenceWorker | None = None
        self.checkpoint_options: list[CheckpointOption] = []
        self.last_result_root: Path | None = None

        self._build_ui()
        self._apply_styles()
        self._refresh_checkpoints()
        self._update_mode_hint()
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._shutdown_worker_on_app_quit)
        self._append_log("실시간 추론 앱이 준비되었습니다.")

    def _build_ui(self) -> None:
        central = QWidget(self)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(20, 20, 20, 20)
        root_layout.setSpacing(16)

        title = QLabel("Realtime Inference Studio")
        title.setObjectName("TitleLabel")
        subtitle = QLabel(
            "학습 앱에서 만든 체크포인트를 고르고, 카메라 프레임과 마이크 버퍼를 계속 샘플링해 같은 엔진 추론 결과를 실시간으로 보여줍니다."
        )
        subtitle.setObjectName("SubtitleLabel")
        root_layout.addWidget(title)
        root_layout.addWidget(subtitle)

        controls = QFrame()
        controls.setObjectName("Card")
        controls_layout = QGridLayout(controls)
        controls_layout.setHorizontalSpacing(12)
        controls_layout.setVerticalSpacing(10)

        self.checkpoint_combo = QComboBox()
        self.checkpoint_combo.currentIndexChanged.connect(self._on_checkpoint_option_changed)
        refresh_button = QPushButton("새로고침")
        refresh_button.clicked.connect(self._refresh_checkpoints)

        self.model_combo = QComboBox()
        self.model_combo.addItems(MODEL_REGISTRY.keys())
        self.model_combo.currentTextChanged.connect(self._update_mode_hint)

        self.checkpoint_edit = QLineEdit()
        checkpoint_button = QPushButton("직접 선택")
        checkpoint_button.clicked.connect(self._choose_checkpoint)

        self.result_base_edit = QLineEdit(str(self.default_result_base))
        result_base_button = QPushButton("결과 위치")
        result_base_button.clicked.connect(self._choose_result_base)

        self.camera_index_spin = QSpinBox()
        self.camera_index_spin.setRange(-1, 16)
        self.camera_index_spin.setSpecialValueText("자동")
        self.camera_index_spin.setValue(-1)

        self.interval_spin = QDoubleSpinBox()
        self.interval_spin.setRange(0.2, 60.0)
        self.interval_spin.setSingleStep(0.5)
        self.interval_spin.setDecimals(1)
        self.interval_spin.setValue(2.0)
        self.interval_spin.setSuffix(" s")

        self.audio_window_spin = QDoubleSpinBox()
        self.audio_window_spin.setRange(0.5, 30.0)
        self.audio_window_spin.setSingleStep(0.5)
        self.audio_window_spin.setDecimals(1)
        self.audio_window_spin.setValue(3.0)
        self.audio_window_spin.setSuffix(" s")

        self.start_button = QPushButton("실시간 추론 시작")
        self.start_button.setObjectName("PrimaryButton")
        self.start_button.clicked.connect(self._start_realtime)
        self.stop_button = QPushButton("정지")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._stop_realtime)
        self.open_result_button = QPushButton("결과 폴더 열기")
        self.open_result_button.clicked.connect(self._open_result_folder)

        controls_layout.addWidget(QLabel("학습 체크포인트"), 0, 0)
        controls_layout.addWidget(self.checkpoint_combo, 0, 1, 1, 3)
        controls_layout.addWidget(refresh_button, 0, 4)
        controls_layout.addWidget(QLabel("모델 종류"), 1, 0)
        controls_layout.addWidget(self.model_combo, 1, 1)
        controls_layout.addWidget(QLabel("체크포인트 파일"), 1, 2)
        controls_layout.addWidget(self.checkpoint_edit, 1, 3)
        controls_layout.addWidget(checkpoint_button, 1, 4)
        controls_layout.addWidget(QLabel("결과 루트"), 2, 0)
        controls_layout.addWidget(self.result_base_edit, 2, 1, 1, 3)
        controls_layout.addWidget(result_base_button, 2, 4)
        controls_layout.addWidget(QLabel("카메라 번호"), 3, 0)
        controls_layout.addWidget(self.camera_index_spin, 3, 1)
        controls_layout.addWidget(QLabel("추론 간격"), 3, 2)
        controls_layout.addWidget(self.interval_spin, 3, 3)
        controls_layout.addWidget(self.start_button, 3, 4)
        controls_layout.addWidget(QLabel("오디오 윈도우"), 4, 0)
        controls_layout.addWidget(self.audio_window_spin, 4, 1)
        controls_layout.addWidget(self.stop_button, 4, 3)
        controls_layout.addWidget(self.open_result_button, 4, 4)

        root_layout.addWidget(controls)

        status = QFrame()
        status.setObjectName("Card")
        status_layout = QGridLayout(status)
        status_layout.setHorizontalSpacing(18)
        self.mode_hint_label = QLabel("-")
        self.status_label = QLabel("Idle")
        self.result_path_label = QLabel("-")
        self.result_path_label.setWordWrap(True)
        self.prediction_label = QLabel("-")
        self.prediction_label.setObjectName("PredictionLabel")
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 10000)
        self.confidence_bar.setValue(0)
        self.confidence_bar.setFormat("-")
        status_layout.addWidget(QLabel("입력 모드"), 0, 0)
        status_layout.addWidget(self.mode_hint_label, 0, 1)
        status_layout.addWidget(QLabel("상태"), 0, 2)
        status_layout.addWidget(self.status_label, 0, 3)
        status_layout.addWidget(QLabel("최근 예측"), 1, 0)
        status_layout.addWidget(self.prediction_label, 1, 1)
        status_layout.addWidget(QLabel("신뢰도"), 1, 2)
        status_layout.addWidget(self.confidence_bar, 1, 3)
        status_layout.addWidget(QLabel("결과 폴더"), 2, 0)
        status_layout.addWidget(self.result_path_label, 2, 1, 1, 3)
        root_layout.addWidget(status)

        splitter = QSplitter(Qt.Horizontal)
        left_panel = QFrame()
        left_panel.setObjectName("Card")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(16, 16, 16, 16)

        preview_group = QGroupBox("카메라 프리뷰")
        preview_layout = QVBoxLayout(preview_group)
        self.preview_label = QLabel("카메라 입력을 기다리는 중입니다.")
        self.preview_label.setObjectName("PreviewBox")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(560, 360)
        preview_layout.addWidget(self.preview_label)
        left_layout.addWidget(preview_group, 2)

        history_group = QGroupBox("최근 추론 기록")
        history_layout = QVBoxLayout(history_group)
        self.history_table = QTableWidget(0, 5)
        self.history_table.setHorizontalHeaderLabels(["Tick", "예측", "신뢰도", "지연", "완료 시각"])
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.history_table.horizontalHeader().setStretchLastSection(True)
        history_layout.addWidget(self.history_table)
        left_layout.addWidget(history_group, 1)
        splitter.addWidget(left_panel)

        right_panel = QFrame()
        right_panel.setObjectName("Card")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(16, 16, 16, 16)

        topk_group = QGroupBox("Top-K 시각화")
        topk_layout = QVBoxLayout(topk_group)
        self.topk_table = QTableWidget(0, 4)
        self.topk_table.setHorizontalHeaderLabels(["Rank", "Label", "Class", "Confidence"])
        self.topk_table.verticalHeader().setVisible(False)
        self.topk_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.topk_table.horizontalHeader().setStretchLastSection(True)
        topk_layout.addWidget(self.topk_table)
        self.branch_label = QLabel("-")
        self.branch_label.setWordWrap(True)
        topk_layout.addWidget(self.branch_label)
        right_layout.addWidget(topk_group, 1)

        log_group = QGroupBox("이벤트 로그")
        log_layout = QVBoxLayout(log_group)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        log_layout.addWidget(self.log_view)
        right_layout.addWidget(log_group, 1)
        splitter.addWidget(right_panel)
        splitter.setSizes([830, 620])

        root_layout.addWidget(splitter, 1)
        self.setCentralWidget(central)

    def _apply_styles(self) -> None:
        self.setFont(QFont("Malgun Gothic", 10))
        self.setStyleSheet(
            """
            QMainWindow {
                background: #edf3ee;
            }
            QLabel#TitleLabel {
                font-size: 30px;
                font-weight: 800;
                color: #16352b;
            }
            QLabel#SubtitleLabel {
                color: #53665f;
            }
            QLabel#PredictionLabel {
                font-size: 22px;
                font-weight: 800;
                color: #153328;
            }
            QFrame#Card {
                background: #fffdf5;
                border: 1px solid #d9d7c5;
                border-radius: 18px;
            }
            QGroupBox {
                border: 1px solid #d5d7c7;
                border-radius: 14px;
                margin-top: 12px;
                padding-top: 12px;
                color: #1d3a30;
                font-weight: 700;
                background: #fbfaf1;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px 0 6px;
            }
            QLabel#PreviewBox {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #13231f, stop:1 #31463e);
                border: 1px solid #829284;
                border-radius: 18px;
                color: #edf4e9;
                font-weight: 700;
            }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTableWidget {
                background: #fffefa;
                border: 1px solid #cbd0bf;
                border-radius: 10px;
                padding: 6px;
                color: #20332d;
            }
            QHeaderView::section {
                background: #dde8dc;
                color: #213b31;
                border: none;
                border-right: 1px solid #c5d0c4;
                padding: 8px;
                font-weight: 700;
            }
            QTableWidget {
                gridline-color: #e5e8dc;
                selection-background-color: #cde7d8;
                selection-color: #14261f;
            }
            QPushButton {
                background: #dfe9dc;
                border: 1px solid #bdcbb8;
                border-radius: 12px;
                padding: 8px 14px;
                color: #21362e;
                font-weight: 700;
            }
            QPushButton:hover {
                background: #d2e1ce;
            }
            QPushButton:disabled {
                background: #eceee8;
                color: #8b948b;
            }
            QPushButton#PrimaryButton {
                background: #237257;
                border-color: #1d614a;
                color: white;
            }
            QPushButton#PrimaryButton:hover {
                background: #1f654e;
            }
            QProgressBar {
                border: 1px solid #c4cdbc;
                border-radius: 10px;
                background: #eef2ea;
                height: 22px;
                color: #1a3229;
                font-weight: 700;
                text-align: center;
            }
            QProgressBar::chunk {
                border-radius: 9px;
                background: #d9912b;
            }
            """
        )

    def _refresh_checkpoints(self) -> None:
        selected_path = self.checkpoint_edit.text().strip()
        self.checkpoint_options = self._discover_training_checkpoints()
        self.checkpoint_combo.blockSignals(True)
        self.checkpoint_combo.clear()

        if not self.checkpoint_options:
            self.checkpoint_combo.addItem("학습 체크포인트를 찾지 못했습니다. 직접 선택을 사용하세요.", None)
        else:
            for option in self.checkpoint_options:
                self.checkpoint_combo.addItem(option.display_text, option)

        self.checkpoint_combo.blockSignals(False)

        if selected_path:
            for index, option in enumerate(self.checkpoint_options):
                if str(option.checkpoint_path) == selected_path:
                    self.checkpoint_combo.setCurrentIndex(index)
                    break
        else:
            self._on_checkpoint_option_changed(self.checkpoint_combo.currentIndex())

        self._append_log(f"체크포인트 {len(self.checkpoint_options)}개를 찾았습니다.")

    def _discover_training_checkpoints(self) -> list[CheckpointOption]:
        training_root = PROJECT_ROOT / "artifacts" / "training_runs"
        if not training_root.exists():
            return []

        options: list[CheckpointOption] = []
        for current_root, dir_names, file_names in os.walk(training_root, onerror=lambda error: None):
            if "training_summary.json" not in file_names:
                continue
            summary_path = Path(current_root) / "training_summary.json"
            summary = _read_json(summary_path)
            if not isinstance(summary, dict):
                continue
            model_name = str(summary.get("model_name", ""))
            if model_name not in MODEL_REGISTRY:
                continue
            local_run_root = summary_path.parent.parent if summary_path.parent.name == "logs" else summary_path.parent
            stored_run_root = Path(str(summary.get("result_root") or local_run_root)).expanduser()
            run_root = stored_run_root if stored_run_root.exists() else local_run_root
            for key, kind in (("best_checkpoint", "best"), ("last_checkpoint", "last")):
                checkpoint_value = summary.get(key)
                if not checkpoint_value:
                    continue
                checkpoint_value_path = Path(str(checkpoint_value)).expanduser()
                candidate_paths = [checkpoint_value_path]
                if not checkpoint_value_path.is_absolute():
                    candidate_paths.extend(
                        [
                            run_root / checkpoint_value_path,
                            local_run_root / checkpoint_value_path,
                        ]
                    )
                candidate_paths.extend(
                    [
                        run_root / "checkpoints" / checkpoint_value_path.name,
                        local_run_root / "checkpoints" / checkpoint_value_path.name,
                    ]
                )
                checkpoint_path = next((path for path in candidate_paths if path.exists()), None)
                if checkpoint_path is None:
                    continue
                options.append(
                    CheckpointOption(
                        model_name=model_name,
                        checkpoint_path=checkpoint_path.resolve(),
                        run_root=run_root.resolve(),
                        checkpoint_kind=kind,
                        display_text=self._format_checkpoint_option(
                            model_name=model_name,
                            run_root=run_root,
                            kind=kind,
                            summary=summary,
                        ),
                        summary=summary,
                    )
                )

        options.sort(key=lambda option: option.checkpoint_path.stat().st_mtime, reverse=True)
        return options

    def _format_checkpoint_option(
        self,
        model_name: str,
        run_root: Path,
        kind: str,
        summary: dict[str, Any],
    ) -> str:
        epochs = f"{summary.get('epochs_completed', '-')}/{summary.get('configured_epochs', '-')}"
        classes = summary.get("num_classes", "-")
        status = summary.get("status", "-")
        return f"{model_name} | {run_root.name} | {kind} | classes={classes} | epochs={epochs} | {status}"

    def _on_checkpoint_option_changed(self, index: int) -> None:
        if index < 0:
            return
        option = self.checkpoint_combo.itemData(index)
        if not isinstance(option, CheckpointOption):
            return
        self.model_combo.setCurrentText(option.model_name)
        self.checkpoint_edit.setText(str(option.checkpoint_path))

    def _choose_checkpoint(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "체크포인트 선택",
            str(PROJECT_ROOT / "artifacts" / "training_runs"),
            "PyTorch Checkpoint (*.pt *.pth);;All Files (*.*)",
        )
        if not file_path:
            return
        checkpoint_path = Path(file_path).expanduser().resolve()
        self.checkpoint_edit.setText(str(checkpoint_path))
        guessed_model_name = _guess_model_name_from_path(checkpoint_path)
        if guessed_model_name:
            self.model_combo.setCurrentText(guessed_model_name)

    def _choose_result_base(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "결과 루트 선택",
            self.result_base_edit.text() or str(self.default_result_base),
        )
        if directory:
            self.result_base_edit.setText(directory)

    def _update_mode_hint(self) -> None:
        mode = MODEL_INPUT_MODES[self.model_combo.currentText()]
        if mode == "image":
            text = "카메라만 사용"
            self.audio_window_spin.setEnabled(False)
        elif mode == "audio":
            text = "마이크만 사용"
            self.audio_window_spin.setEnabled(True)
        else:
            text = "카메라 + 마이크 사용"
            self.audio_window_spin.setEnabled(True)
        self.mode_hint_label.setText(text)

    def _start_realtime(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self, "실행 중", "이미 실시간 추론이 실행 중입니다.")
            return

        model_name = self.model_combo.currentText()
        checkpoint_path = Path(self.checkpoint_edit.text()).expanduser() if self.checkpoint_edit.text().strip() else None
        if checkpoint_path is None or not checkpoint_path.exists():
            QMessageBox.warning(self, "체크포인트 오류", "유효한 체크포인트 파일을 선택하세요.")
            return

        result_base = Path(self.result_base_edit.text()).expanduser() if self.result_base_edit.text().strip() else self.default_result_base
        result_root = result_base.resolve() / f"{_timestamp_slug()}_{model_name}_realtime"
        self.last_result_root = result_root

        session_config = RealtimeSessionConfig(
            model_name=model_name,
            checkpoint_path=checkpoint_path.resolve(),
            result_root=result_root,
            camera_index=self.camera_index_spin.value(),
            inference_interval_seconds=float(self.interval_spin.value()),
            audio_window_seconds=float(self.audio_window_spin.value()),
        )

        self._set_running_state(True)
        self.history_table.setRowCount(0)
        self.topk_table.setRowCount(0)
        self.prediction_label.setText("-")
        self.confidence_bar.setValue(0)
        self.confidence_bar.setFormat("-")
        self.result_path_label.setText(str(result_root))
        self.status_label.setText("Starting")
        self._append_log(f"{model_name} 실시간 추론을 시작합니다.")

        self.worker = RealtimeInferenceWorker(session_config)
        self.worker.status_changed.connect(self._on_status_changed)
        self.worker.frame_ready.connect(self._on_frame_ready)
        self.worker.prediction_ready.connect(self._on_prediction_ready)
        self.worker.failed.connect(self._on_worker_failed)
        self.worker.stopped.connect(self._on_worker_stopped)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.start()

    def _stop_realtime(self) -> None:
        if self.worker is not None:
            self.status_label.setText("Stopping")
            self._append_log("실시간 추론 정지를 요청했습니다.")
            self.worker.request_stop()

    def _set_running_state(self, running: bool) -> None:
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.checkpoint_combo.setEnabled(not running)
        self.model_combo.setEnabled(not running)
        self.checkpoint_edit.setEnabled(not running)
        self.camera_index_spin.setEnabled(not running)
        self.interval_spin.setEnabled(not running)
        self.audio_window_spin.setEnabled(not running and MODEL_INPUT_MODES[self.model_combo.currentText()] != "image")

    def _on_status_changed(self, message: str) -> None:
        self.status_label.setText(message)
        self._append_log(message)

    def _on_frame_ready(self, frame_bytes: object) -> None:
        if not isinstance(frame_bytes, (bytes, bytearray)) or not frame_bytes:
            return
        pixmap = QPixmap()
        if pixmap.loadFromData(bytes(frame_bytes)):
            self.preview_label.setText("")
            self.preview_label.setPixmap(
                pixmap.scaled(
                    self.preview_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )

    def _on_prediction_ready(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        prediction = payload.get("prediction") or {}
        label = str(prediction.get("predicted_display_name", "-"))
        confidence = prediction.get("confidence")
        confidence_text = _format_confidence(confidence)
        latency = float(payload.get("latency_seconds", 0.0))

        self.prediction_label.setText(label)
        self.confidence_bar.setValue(_confidence_int(confidence))
        self.confidence_bar.setFormat(confidence_text)
        self.status_label.setText(f"최근 추론 완료 ({latency:.2f}s)")
        self._append_log(f"tick {payload.get('tick_index')} | {label} | {confidence_text} | {latency:.2f}s")

        self._refresh_topk(payload)
        self._prepend_history_row(payload, label, confidence_text, latency)

    def _refresh_topk(self, payload: dict[str, Any]) -> None:
        top_k = payload.get("top_k") or []
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

        branch_lines: list[str] = []
        image_branch = payload.get("image_branch")
        audio_branch = payload.get("audio_branch")
        if isinstance(image_branch, dict):
            branch_lines.append(
                f"이미지 branch: {image_branch.get('predicted_display_name', '-')} ({_format_confidence(image_branch.get('confidence'))})"
            )
        if isinstance(audio_branch, dict):
            branch_lines.append(
                f"오디오 branch: {audio_branch.get('predicted_display_name', '-')} ({_format_confidence(audio_branch.get('confidence'))})"
            )
        self.branch_label.setText("\n".join(branch_lines) if branch_lines else "-")

    def _prepend_history_row(
        self,
        payload: dict[str, Any],
        label: str,
        confidence_text: str,
        latency: float,
    ) -> None:
        self.history_table.insertRow(0)
        values = [
            str(payload.get("tick_index", "-")),
            label,
            confidence_text,
            f"{latency:.2f}s",
            str(payload.get("completed_at", "-")),
        ]
        for column, value in enumerate(values):
            item = QTableWidgetItem(value)
            if column != 1:
                item.setTextAlignment(Qt.AlignCenter)
            self.history_table.setItem(0, column, item)
        while self.history_table.rowCount() > 80:
            self.history_table.removeRow(self.history_table.rowCount() - 1)
        self.history_table.resizeColumnsToContents()

    def _on_worker_failed(self, error_text: str) -> None:
        self.status_label.setText("Failed")
        self._append_log(error_text)
        QMessageBox.critical(self, "실시간 추론 오류", error_text)

    def _on_worker_stopped(self) -> None:
        self._set_running_state(False)
        self.status_label.setText("Idle")
        self._append_log("실시간 추론이 종료되었습니다.")

    def _on_worker_finished(self) -> None:
        worker = self.sender()
        if isinstance(worker, RealtimeInferenceWorker):
            worker.deleteLater()
            if self.worker is worker:
                self.worker = None

    def _shutdown_worker(self, timeout_ms: int | None = None) -> bool:
        if self.worker is None:
            return True

        self.worker.request_stop()
        if not self.worker.isRunning():
            return True

        if timeout_ms is None:
            self.worker.wait()
            return True
        return bool(self.worker.wait(timeout_ms))

    def _shutdown_worker_on_app_quit(self) -> None:
        self._shutdown_worker()

    def _open_result_folder(self) -> None:
        result_root = self.last_result_root
        if result_root is None:
            result_text = self.result_path_label.text().strip()
            result_root = Path(result_text) if result_text and result_text != "-" else None
        if result_root is None:
            QMessageBox.information(self, "결과 없음", "아직 열린 결과 폴더가 없습니다.")
            return
        result_root.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(result_root)))

    def _append_log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_view.appendPlainText(f"[{timestamp}] {message}")

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self.worker is not None and self.worker.isRunning():
            response = QMessageBox.question(
                self,
                "실시간 추론 실행 중",
                "지금 닫으면 카메라/마이크 스트림과 추론 작업을 중지합니다. 종료할까요?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if response != QMessageBox.Yes:
                event.ignore()
                return
            if not self._shutdown_worker(timeout_ms=3000):
                QMessageBox.warning(
                    self,
                    "Realtime inference is stopping",
                    "The inference worker is still shutting down. Try closing again after it finishes.",
                )
                event.ignore()
                return
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    window = RealtimeInferenceWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
