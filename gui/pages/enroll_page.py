from pathlib import Path

from PyQt5.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from gui.widgets.audio_quality_card import AudioQualityCard
from gui.widgets.recording_controls import RecordingControls


class EnrollPage(QWidget):
    def __init__(self):
        super().__init__()

        title = QLabel("授权注册")
        title.setObjectName("PageTitle")
        hint = QLabel(
            "采集多条同一人的音频样本，生成当前活动声纹档案。"
            "导入手机录音通常比电脑内置麦克风更稳定。"
        )
        hint.setWordWrap(True)
        hint.setObjectName("PageHint")

        self.display_name_edit = QLineEdit("授权用户")
        self.display_name_edit.setPlaceholderText("例如：张三 / 管理员")
        self.progress_label = QLabel("0 / 3")
        self.progress_label.setObjectName("MetricValue")
        self.status_label = QLabel("等待开始注册")
        self.status_label.setWordWrap(True)

        self.recording_controls = RecordingControls()
        self.quality_card = AudioQualityCard()
        self.reset_button = QPushButton("重置本轮注册")

        self.samples_list = QListWidget()

        form = QFormLayout()
        form.setSpacing(12)
        form.addRow("授权人名称", self.display_name_edit)
        form.addRow("当前进度", self.progress_label)
        form.addRow("运行状态", self.status_label)

        task_box = QGroupBox("注册任务")
        task_layout = QVBoxLayout(task_box)
        task_layout.addLayout(form)
        task_layout.addWidget(self.recording_controls)
        task_layout.addWidget(self.quality_card)
        task_layout.addWidget(self.reset_button)

        note = QLabel(
            "建议在安静环境下采集，单次样本不宜过短。"
            "若只是烟测，可把注册样本数临时设为 1。"
        )
        note.setWordWrap(True)
        note.setObjectName("SectionNote")

        sample_box = QGroupBox("已采集样本")
        sample_layout = QVBoxLayout(sample_box)
        sample_layout.addWidget(self.samples_list)

        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.addWidget(title)
        layout.addWidget(hint)
        layout.addWidget(task_box)
        layout.addWidget(note)
        layout.addWidget(sample_box)
        layout.addStretch(1)

    def display_name(self):
        text = self.display_name_edit.text().strip()
        return text or "授权用户"

    def set_progress(self, current, total):
        self.progress_label.setText(f"{current} / {total}")

    def add_sample(self, audio_path, quality_info=None):
        name = Path(audio_path).name
        if quality_info:
            dur = quality_info.get("duration_s", 0)
            grade = quality_info.get("quality_grade", "?")
            name = f"{name} ({dur:.1f}s, {grade})"
        self.samples_list.addItem(name)
        self.quality_card.set_quality(quality_info)

    def clear_samples(self):
        self.samples_list.clear()
        self.quality_card.clear()

    def set_recording(self, recording):
        self.recording_controls.set_recording(recording)
        self.reset_button.setEnabled(not recording)

    def set_level(self, level):
        self.recording_controls.set_level(level)

    def set_status(self, text, ok=None):
        color = "#334155"
        if ok is True:
            color = "#067647"
        elif ok is False:
            color = "#b42318"
        self.status_label.setStyleSheet(f"color: {color}; font-weight: 600;")
        self.status_label.setText(text)
