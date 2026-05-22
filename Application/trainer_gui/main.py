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

from PySide6.QtCore import QThread, QTimer, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QFont
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
    QProgressBar,
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


def _now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _slugify(value: str) -> str:
    cleaned = "".join(character if character.isalnum() else "-" for character in value.strip())
    parts = [part for part in cleaned.split("-") if part]
    return "-".join(parts) or "dataset"


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


def _format_float(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


@dataclass
class TrainingJob:
    job_id: str
    model_name: str
    dataset_root: Path
    result_root: Path
    status: str = "queued"
    message: str = "대기 중"
    created_at: str = field(default_factory=_now_text)
    started_at: str | None = None
    finished_at: str | None = None
    configured_epochs: int = 0
    epochs_completed: int = 0

    @property
    def logs_dir(self) -> Path:
        return self.result_root / "logs"

    @property
    def state_path(self) -> Path:
        return self.logs_dir / "training_state.json"

    @property
    def summary_path(self) -> Path:
        return self.logs_dir / "training_summary.json"

    @property
    def history_path(self) -> Path:
        return self.logs_dir / "training_history.json"

    @property
    def metrics_path(self) -> Path:
        return self.result_root / "metrics" / "test_metrics.json"


class TrainingWorker(QThread):
    finished_job = Signal(str, bool, str)
    state_changed = Signal(str, str, str)

    def __init__(self, job: TrainingJob) -> None:
        super().__init__()
        self.job = job

    def run(self) -> None:
        model_class = MODEL_REGISTRY[self.job.model_name]
        self.job.started_at = _now_text()
        self._write_state("running", "학습을 시작했습니다.")
        self.state_changed.emit(self.job.job_id, "running", "학습을 시작했습니다.")

        try:
            model = model_class()
            model.Train(self.job.dataset_root, self.job.result_root)
        except Exception:
            self.job.finished_at = _now_text()
            error_text = traceback.format_exc()
            self._write_state("failed", "학습 중 오류가 발생했습니다.", error=error_text)
            self.finished_job.emit(self.job.job_id, False, error_text)
            return

        self.job.finished_at = _now_text()
        self._write_state("completed", "학습이 완료되었습니다.")
        self.finished_job.emit(self.job.job_id, True, "학습이 완료되었습니다.")

    def _write_state(self, status: str, message: str, error: str | None = None) -> None:
        payload = {
            "job_id": self.job.job_id,
            "model_name": self.job.model_name,
            "dataset_root": str(self.job.dataset_root),
            "result_root": str(self.job.result_root),
            "status": status,
            "message": message,
            "created_at": self.job.created_at,
            "started_at": self.job.started_at,
            "finished_at": self.job.finished_at,
            "updated_at": _now_text(),
        }
        if error:
            payload["error"] = error
        _write_json(self.job.state_path, payload)


class TrainerWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Pokemon Recognition Trainer")
        self.resize(1520, 940)

        self.jobs: list[TrainingJob] = []
        self.active_worker: TrainingWorker | None = None
        self.active_job_id: str | None = None
        self.selected_job_id: str | None = None
        self.job_sequence = 0
        self.default_result_base = PROJECT_ROOT / "artifacts" / "training_runs"

        self._build_ui()
        self._apply_styles()

        self.monitor_timer = QTimer(self)
        self.monitor_timer.setInterval(1000)
        self.monitor_timer.timeout.connect(self._poll_result_directories)
        self.monitor_timer.start()

        self._refresh_queue_table()
        self._refresh_active_summary()
        self._refresh_detail_panel()
        self._append_log("학습 앱이 준비되었습니다.")

    def _build_ui(self) -> None:
        central = QWidget(self)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)

        title_label = QLabel("Training Queue Console")
        title_label.setObjectName("TitleLabel")
        subtitle_label = QLabel(
            "데이터셋을 고르고 모델을 큐에 넣으면, 결과 폴더를 읽어가며 진행 상태를 계속 갱신합니다."
        )
        subtitle_label.setObjectName("SubtitleLabel")
        main_layout.addWidget(title_label)
        main_layout.addWidget(subtitle_label)

        control_card = QFrame()
        control_card.setObjectName("Card")
        control_layout = QGridLayout(control_card)
        control_layout.setHorizontalSpacing(12)
        control_layout.setVerticalSpacing(12)

        self.dataset_path_edit = QLineEdit(str(PROJECT_ROOT / "Data" / "SSW60" / "processed"))
        dataset_button = QPushButton("폴더 선택")
        dataset_button.clicked.connect(self._choose_dataset_directory)

        self.result_base_edit = QLineEdit(str(self.default_result_base))
        result_base_button = QPushButton("저장 위치")
        result_base_button.clicked.connect(self._choose_result_base_directory)

        self.model_combo = QComboBox()
        self.model_combo.addItems(MODEL_REGISTRY.keys())

        self.queue_button = QPushButton("학습 대기열 추가")
        self.queue_button.setObjectName("PrimaryButton")
        self.queue_button.clicked.connect(self._enqueue_job)

        control_layout.addWidget(QLabel("데이터셋 폴더"), 0, 0)
        control_layout.addWidget(self.dataset_path_edit, 0, 1)
        control_layout.addWidget(dataset_button, 0, 2)
        control_layout.addWidget(QLabel("결과 루트"), 1, 0)
        control_layout.addWidget(self.result_base_edit, 1, 1)
        control_layout.addWidget(result_base_button, 1, 2)
        control_layout.addWidget(QLabel("모델 선택"), 2, 0)
        control_layout.addWidget(self.model_combo, 2, 1)
        control_layout.addWidget(self.queue_button, 2, 2)

        main_layout.addWidget(control_card)

        status_card = QFrame()
        status_card.setObjectName("Card")
        status_layout = QGridLayout(status_card)
        status_layout.setHorizontalSpacing(16)
        status_layout.setVerticalSpacing(8)

        self.active_job_label = QLabel("활성 작업 없음")
        self.active_status_label = QLabel("Idle")
        self.queue_depth_label = QLabel("대기열 0건")
        self.progress_text_label = QLabel("진행률 정보 없음")
        self.result_folder_label = QLabel("-")
        self.result_folder_label.setWordWrap(True)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)

        self.open_result_button = QPushButton("결과 폴더 열기")
        self.open_result_button.clicked.connect(self._open_selected_result_folder)
        self.open_dataset_button = QPushButton("데이터셋 폴더 열기")
        self.open_dataset_button.clicked.connect(self._open_selected_dataset_folder)

        status_layout.addWidget(QLabel("활성 작업"), 0, 0)
        status_layout.addWidget(self.active_job_label, 0, 1)
        status_layout.addWidget(QLabel("상태"), 0, 2)
        status_layout.addWidget(self.active_status_label, 0, 3)
        status_layout.addWidget(QLabel("큐"), 0, 4)
        status_layout.addWidget(self.queue_depth_label, 0, 5)
        status_layout.addWidget(QLabel("결과 폴더"), 1, 0)
        status_layout.addWidget(self.result_folder_label, 1, 1, 1, 5)
        status_layout.addWidget(self.progress_bar, 2, 0, 1, 6)
        status_layout.addWidget(self.progress_text_label, 3, 0, 1, 4)
        status_layout.addWidget(self.open_dataset_button, 3, 4)
        status_layout.addWidget(self.open_result_button, 3, 5)

        main_layout.addWidget(status_card)

        splitter = QSplitter(Qt.Horizontal)

        queue_panel = QFrame()
        queue_panel.setObjectName("Card")
        queue_layout = QVBoxLayout(queue_panel)
        queue_layout.setContentsMargins(16, 16, 16, 16)
        queue_layout.setSpacing(10)
        queue_layout.addWidget(QLabel("학습 대기열"))

        self.queue_table = QTableWidget(0, 6)
        self.queue_table.setHorizontalHeaderLabels(
            ["작업", "모델", "상태", "에폭", "데이터셋", "결과 폴더"]
        )
        self.queue_table.verticalHeader().setVisible(False)
        self.queue_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.queue_table.setSelectionMode(QTableWidget.SingleSelection)
        self.queue_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.queue_table.horizontalHeader().setStretchLastSection(True)
        self.queue_table.itemSelectionChanged.connect(self._on_queue_selection_changed)
        queue_layout.addWidget(self.queue_table)
        splitter.addWidget(queue_panel)

        detail_tabs = QTabWidget()
        detail_tabs.setObjectName("CardTabs")

        overview_tab = QWidget()
        overview_layout = QVBoxLayout(overview_tab)
        overview_group = QGroupBox("작업 개요")
        overview_form = QFormLayout(overview_group)
        self.detail_labels: dict[str, QLabel] = {}
        for key, label in (
            ("job_id", "작업 ID"),
            ("model_name", "모델"),
            ("status", "상태"),
            ("dataset_root", "데이터셋"),
            ("result_root", "결과 폴더"),
            ("created_at", "생성 시각"),
            ("started_at", "시작 시각"),
            ("finished_at", "종료 시각"),
            ("epochs", "에폭"),
            ("samples", "샘플 수"),
            ("checkpoints", "체크포인트"),
            ("message", "메시지"),
        ):
            value_label = QLabel("-")
            value_label.setWordWrap(True)
            self.detail_labels[key] = value_label
            overview_form.addRow(label, value_label)
        overview_layout.addWidget(overview_group)
        overview_layout.addStretch(1)
        detail_tabs.addTab(overview_tab, "개요")

        history_tab = QWidget()
        history_layout = QVBoxLayout(history_tab)
        self.history_table = QTableWidget(0, 5)
        self.history_table.setHorizontalHeaderLabels(
            ["Epoch", "Train Loss", "Train Acc", "Val Loss", "Val Acc"]
        )
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.history_table.horizontalHeader().setStretchLastSection(True)
        history_layout.addWidget(self.history_table)
        detail_tabs.addTab(history_tab, "학습 이력")

        metrics_tab = QWidget()
        metrics_layout = QVBoxLayout(metrics_tab)
        self.metrics_view = QPlainTextEdit()
        self.metrics_view.setReadOnly(True)
        metrics_layout.addWidget(self.metrics_view)
        detail_tabs.addTab(metrics_tab, "평가 지표")

        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        self.event_log_view = QPlainTextEdit()
        self.event_log_view.setReadOnly(True)
        log_layout.addWidget(self.event_log_view)
        detail_tabs.addTab(log_tab, "이벤트 로그")

        splitter.addWidget(detail_tabs)
        splitter.setSizes([760, 740])
        main_layout.addWidget(splitter, 1)

        self.setCentralWidget(central)

    def _apply_styles(self) -> None:
        self.setFont(QFont("Segoe UI", 10))
        self.setStyleSheet(
            """
            QMainWindow {
                background: #f4f0e7;
            }
            QLabel#TitleLabel {
                font-size: 28px;
                font-weight: 700;
                color: #1f2d3d;
            }
            QLabel#SubtitleLabel {
                color: #5d6a78;
                margin-bottom: 4px;
            }
            QFrame#Card, QTabWidget#CardTabs {
                background: #fffdf9;
                border: 1px solid #e3d8c6;
                border-radius: 18px;
            }
            QGroupBox {
                border: 1px solid #eadfcd;
                border-radius: 14px;
                margin-top: 12px;
                padding-top: 10px;
                font-weight: 600;
                color: #2f3e4d;
                background: #fffaf2;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px 0 6px;
            }
            QLineEdit, QComboBox, QTableWidget, QPlainTextEdit, QTabWidget::pane {
                background: #fffefb;
                border: 1px solid #d9ccb9;
                border-radius: 10px;
                padding: 6px;
                color: #243241;
            }
            QHeaderView::section {
                background: #f0e8dc;
                color: #364351;
                border: none;
                border-right: 1px solid #dccdb8;
                padding: 8px;
                font-weight: 600;
            }
            QTableWidget {
                gridline-color: #eadfcd;
                selection-background-color: #ffe0b2;
                selection-color: #1f2d3d;
            }
            QPushButton {
                background: #f0e5d4;
                border: 1px solid #d7c4ab;
                border-radius: 12px;
                padding: 8px 14px;
                color: #2d3948;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #ead9c1;
            }
            QPushButton#PrimaryButton {
                background: #d86f1e;
                color: white;
                border-color: #bf5e14;
            }
            QPushButton#PrimaryButton:hover {
                background: #c96418;
            }
            QProgressBar {
                border: 1px solid #d8ccb9;
                border-radius: 10px;
                background: #f7f2ea;
                height: 18px;
            }
            QProgressBar::chunk {
                border-radius: 9px;
                background: #2c9c8c;
            }
            QTabBar::tab {
                background: #f3eadf;
                border: 1px solid #dfd0bb;
                border-bottom: none;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                padding: 8px 14px;
                margin-right: 4px;
                color: #3a4652;
            }
            QTabBar::tab:selected {
                background: #fffdf9;
                color: #1f2d3d;
            }
            """
        )

    def _choose_dataset_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "데이터셋 폴더 선택",
            self.dataset_path_edit.text() or str(PROJECT_ROOT),
        )
        if directory:
            self.dataset_path_edit.setText(directory)

    def _choose_result_base_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "결과 저장 루트 선택",
            self.result_base_edit.text() or str(self.default_result_base),
        )
        if directory:
            self.result_base_edit.setText(directory)

    def _enqueue_job(self) -> None:
        dataset_root = Path(self.dataset_path_edit.text()).expanduser()
        result_base = Path(self.result_base_edit.text()).expanduser()
        model_name = self.model_combo.currentText()

        if not dataset_root.exists():
            QMessageBox.warning(self, "데이터셋 오류", "선택한 데이터셋 폴더가 존재하지 않습니다.")
            return
        if not (dataset_root / "train").exists() or not (dataset_root / "test").exists():
            QMessageBox.warning(
                self,
                "데이터셋 오류",
                "데이터셋 폴더 아래에 최소한 train/test split이 필요합니다.",
            )
            return

        self.job_sequence += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_slug = _slugify(dataset_root.name)
        result_root = (
            result_base.resolve()
            / f"{timestamp}_{self.job_sequence:03d}_{model_name}_{dataset_slug}"
        )

        job = TrainingJob(
            job_id=f"JOB-{self.job_sequence:03d}",
            model_name=model_name,
            dataset_root=dataset_root.resolve(),
            result_root=result_root,
        )
        self.jobs.append(job)
        self.selected_job_id = job.job_id
        self._write_job_state(job, "queued", "대기열에 추가되었습니다.")
        self._append_log(f"{job.job_id} | {job.model_name} | 대기열 추가 | {job.dataset_root}")
        self._refresh_queue_table()
        self._refresh_active_summary()
        self._refresh_detail_panel()
        self._start_next_job_if_idle()

    def _start_next_job_if_idle(self) -> None:
        if self.active_worker is not None and self.active_worker.isRunning():
            return

        next_job = next((job for job in self.jobs if job.status == "queued"), None)
        if next_job is None:
            self.active_worker = None
            self.active_job_id = None
            self._refresh_active_summary()
            return

        self.active_job_id = next_job.job_id
        self.active_worker = TrainingWorker(next_job)
        self.active_worker.state_changed.connect(self._on_worker_state_changed)
        self.active_worker.finished_job.connect(self._on_worker_finished)
        self.active_worker.start()

    def _on_worker_state_changed(self, job_id: str, status: str, message: str) -> None:
        job = self._find_job(job_id)
        if job is None:
            return
        job.status = status
        job.message = message
        job.started_at = job.started_at or _now_text()
        self._append_log(f"{job.job_id} | {job.model_name} | {message}")
        self._refresh_queue_table()
        self._refresh_active_summary()
        self._refresh_detail_panel()

    def _on_worker_finished(self, job_id: str, success: bool, message: str) -> None:
        job = self._find_job(job_id)
        if job is not None:
            job.status = "completed" if success else "failed"
            job.message = "학습이 완료되었습니다." if success else "학습이 실패했습니다."
            job.finished_at = _now_text()

        summary_line = "완료" if success else "실패"
        self._append_log(f"{job_id} | 학습 {summary_line}")
        if not success:
            self._append_log(message)

        if self.active_worker is not None:
            self.active_worker.deleteLater()
            self.active_worker = None
        self.active_job_id = None
        self._refresh_queue_table()
        self._refresh_active_summary()
        self._refresh_detail_panel()
        self._start_next_job_if_idle()

    def _poll_result_directories(self) -> None:
        for job in self.jobs:
            state = _read_json(job.state_path)
            summary = _read_json(job.summary_path)

            if state is not None:
                job.status = str(state.get("status", job.status))
                job.message = str(state.get("message", job.message))
                job.started_at = state.get("started_at") or job.started_at
                job.finished_at = state.get("finished_at") or job.finished_at

            if summary is not None:
                job.configured_epochs = int(summary.get("configured_epochs", job.configured_epochs or 0))
                job.epochs_completed = int(summary.get("epochs_completed", job.epochs_completed or 0))
                if state is None and "status" in summary:
                    job.status = str(summary["status"])

        self._refresh_queue_table()
        self._refresh_active_summary()
        self._refresh_detail_panel()

    def _refresh_queue_table(self) -> None:
        selected_job_id = self.selected_job_id
        self.queue_table.setRowCount(len(self.jobs))

        for row, job in enumerate(self.jobs):
            epochs_text = "-"
            if job.configured_epochs > 0:
                epochs_text = f"{job.epochs_completed} / {job.configured_epochs}"
            elif job.epochs_completed > 0:
                epochs_text = str(job.epochs_completed)

            values = [
                job.job_id,
                job.model_name,
                job.status,
                epochs_text,
                str(job.dataset_root),
                str(job.result_root),
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 0:
                    item.setData(Qt.UserRole, job.job_id)
                if column in (0, 1, 2, 3):
                    item.setTextAlignment(Qt.AlignCenter)
                self.queue_table.setItem(row, column, item)

        self.queue_table.resizeColumnsToContents()
        if selected_job_id is not None:
            for row in range(self.queue_table.rowCount()):
                item = self.queue_table.item(row, 0)
                if item is not None and item.data(Qt.UserRole) == selected_job_id:
                    self.queue_table.selectRow(row)
                    break

    def _refresh_active_summary(self) -> None:
        active_job = self._find_job(self.active_job_id) if self.active_job_id else None
        queued_count = len([job for job in self.jobs if job.status == "queued"])
        self.queue_depth_label.setText(f"대기열 {queued_count}건")

        if active_job is None:
            self.active_job_label.setText("활성 작업 없음")
            self.active_status_label.setText("Idle")
            self.progress_text_label.setText("진행 중인 학습이 없습니다.")
            self.result_folder_label.setText("-")
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(0)
            return

        self.active_job_label.setText(f"{active_job.job_id} · {active_job.model_name}")
        self.active_status_label.setText(active_job.status)
        self.progress_text_label.setText(active_job.message or "학습 진행 중")
        self.result_folder_label.setText(str(active_job.result_root))

        if active_job.configured_epochs > 0:
            self.progress_bar.setRange(0, active_job.configured_epochs)
            self.progress_bar.setValue(min(active_job.epochs_completed, active_job.configured_epochs))
            self.progress_bar.setFormat(f"{active_job.epochs_completed} / {active_job.configured_epochs}")
            self.progress_bar.setTextVisible(True)
        else:
            self.progress_bar.setRange(0, 0)
            self.progress_bar.setTextVisible(False)

    def _refresh_detail_panel(self) -> None:
        job = self._find_job(self.selected_job_id) or self._find_job(self.active_job_id)
        if job is None:
            for label in self.detail_labels.values():
                label.setText("-")
            self.history_table.setRowCount(0)
            self.metrics_view.setPlainText("선택된 작업이 없습니다.")
            return

        summary = _read_json(job.summary_path) or {}
        state = _read_json(job.state_path) or {}
        history = _read_json(job.history_path) or []
        metrics = _read_json(job.metrics_path)

        samples_text = (
            f"train={summary.get('train_samples', '-')}, "
            f"val={summary.get('val_samples', '-')}, "
            f"test={summary.get('test_samples', '-')}"
        )
        checkpoints_text = (
            f"best={summary.get('best_checkpoint', '-')}\n"
            f"last={summary.get('last_checkpoint', '-')}"
        )
        epochs_text = (
            f"{summary.get('epochs_completed', job.epochs_completed)} / "
            f"{summary.get('configured_epochs', job.configured_epochs or '-')}"
        )

        self.detail_labels["job_id"].setText(job.job_id)
        self.detail_labels["model_name"].setText(job.model_name)
        self.detail_labels["status"].setText(str(state.get("status", job.status)))
        self.detail_labels["dataset_root"].setText(str(job.dataset_root))
        self.detail_labels["result_root"].setText(str(job.result_root))
        self.detail_labels["created_at"].setText(job.created_at)
        self.detail_labels["started_at"].setText(job.started_at or "-")
        self.detail_labels["finished_at"].setText(job.finished_at or "-")
        self.detail_labels["epochs"].setText(epochs_text)
        self.detail_labels["samples"].setText(samples_text)
        self.detail_labels["checkpoints"].setText(checkpoints_text)
        self.detail_labels["message"].setText(str(state.get("message", job.message)))

        self.history_table.setRowCount(len(history))
        for row, item in enumerate(history):
            train_metrics = item.get("train", {}) or {}
            val_metrics = item.get("val", {}) or {}
            row_values = [
                str(item.get("epoch", row + 1)),
                _format_float(train_metrics.get("loss")),
                _format_float(train_metrics.get("accuracy")),
                _format_float(val_metrics.get("loss")),
                _format_float(val_metrics.get("accuracy")),
            ]
            for column, value in enumerate(row_values):
                table_item = QTableWidgetItem(value)
                table_item.setTextAlignment(Qt.AlignCenter)
                self.history_table.setItem(row, column, table_item)
        self.history_table.resizeColumnsToContents()

        if metrics is None:
            metrics_payload = {
                "state": state,
                "summary": summary,
                "latest_epoch": history[-1] if history else None,
            }
            self.metrics_view.setPlainText(
                json.dumps(metrics_payload, ensure_ascii=False, indent=2)
            )
        else:
            self.metrics_view.setPlainText(json.dumps(metrics, ensure_ascii=False, indent=2))

    def _on_queue_selection_changed(self) -> None:
        current_row = self.queue_table.currentRow()
        if current_row < 0:
            return
        item = self.queue_table.item(current_row, 0)
        if item is None:
            return
        self.selected_job_id = item.data(Qt.UserRole)
        self._refresh_detail_panel()

    def _write_job_state(self, job: TrainingJob, status: str, message: str, error: str | None = None) -> None:
        payload = {
            "job_id": job.job_id,
            "model_name": job.model_name,
            "dataset_root": str(job.dataset_root),
            "result_root": str(job.result_root),
            "status": status,
            "message": message,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "updated_at": _now_text(),
        }
        if error:
            payload["error"] = error
        _write_json(job.state_path, payload)

    def _find_job(self, job_id: str | None) -> TrainingJob | None:
        if job_id is None:
            return None
        return next((job for job in self.jobs if job.job_id == job_id), None)

    def _append_log(self, message: str) -> None:
        timestamped = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
        self.event_log_view.appendPlainText(timestamped)

    def _open_selected_result_folder(self) -> None:
        job = self._find_job(self.selected_job_id) or self._find_job(self.active_job_id)
        if job is None:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(job.result_root)))

    def _open_selected_dataset_folder(self) -> None:
        job = self._find_job(self.selected_job_id) or self._find_job(self.active_job_id)
        if job is None:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(job.dataset_root)))

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self.active_worker is not None and self.active_worker.isRunning():
            response = QMessageBox.question(
                self,
                "학습 진행 중",
                "지금 종료하면 진행 중인 학습이 중단될 수 있습니다. 정말 닫을까요?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if response != QMessageBox.Yes:
                event.ignore()
                return
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    window = TrainerWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
