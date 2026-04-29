from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class RecordingControls(QWidget):
    start_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    import_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.start_button = QPushButton("开始录音")
        self.stop_button = QPushButton("停止录音")
        self.import_button = QPushButton("导入音频")
        self.stop_button.setEnabled(False)

        self.start_button.clicked.connect(self.start_clicked)
        self.stop_button.clicked.connect(self.stop_clicked)
        self.import_button.clicked.connect(self.import_clicked)

        self.level_bar = QProgressBar()
        self.level_bar.setRange(0, 100)
        self.level_bar.setTextVisible(False)
        self.level_bar.setFixedHeight(14)

        self._timer_label = QLabel("00:00")
        self._timer_label.setObjectName("MutedValue")
        self._timer_label.setAlignment(Qt.AlignCenter)
        self._timer_label.setFixedWidth(60)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        btn_row.addWidget(self.start_button)
        btn_row.addWidget(self.stop_button)
        btn_row.addWidget(self.import_button)
        btn_row.addStretch(1)

        level_row = QHBoxLayout()
        level_row.setSpacing(8)
        level_row.addWidget(QLabel("电平"))
        level_row.addWidget(self.level_bar, 1)
        level_row.addWidget(self._timer_label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addLayout(btn_row)
        layout.addLayout(level_row)

    def set_recording(self, recording):
        self.start_button.setEnabled(not recording)
        self.stop_button.setEnabled(recording)
        self.import_button.setEnabled(not recording)

    def set_level(self, level):
        self.level_bar.setValue(int(level * 100))

    def set_elapsed(self, seconds):
        m, s = divmod(int(seconds), 60)
        self._timer_label.setText(f"{m:02d}:{s:02d}")

    def reset_elapsed(self):
        self._timer_label.setText("00:00")

    def set_controls_enabled(self, enabled):
        self.start_button.setEnabled(enabled)
        self.import_button.setEnabled(enabled)
        self.stop_button.setEnabled(False)
