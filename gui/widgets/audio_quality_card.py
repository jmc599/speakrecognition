from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget


_GRADE_COLORS = {
    "A": ("#ecfdf3", "#067647"),
    "B": ("#f0fdf4", "#15803d"),
    "C": ("#fefce8", "#a16207"),
    "D": ("#fef2f2", "#b91c1c"),
}


class AudioQualityCard(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("AudioQualityCard")

        self._grade_label = QLabel("-")
        self._grade_label.setAlignment(Qt.AlignCenter)
        self._grade_label.setFixedSize(36, 36)
        self._grade_label.setStyleSheet(
            "font-size:16pt;font-weight:700;border-radius:8px;"
            "background:#f5f7fa;color:#334155;"
        )

        self._detail_label = QLabel("")
        self._detail_label.setWordWrap(True)
        self._detail_label.setStyleSheet("color:#475569;font-size:9pt;")

        self._warn_label = QLabel("")
        self._warn_label.setWordWrap(True)
        self._warn_label.setStyleSheet("color:#a16207;font-size:9pt;font-weight:600;")
        self._warn_label.hide()

        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(2)
        right.addWidget(self._detail_label)
        right.addWidget(self._warn_label)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(10)
        layout.addWidget(self._grade_label)
        layout.addLayout(right, 1)

    def set_quality(self, quality_info):
        if quality_info is None:
            self._grade_label.setText("-")
            self._grade_label.setStyleSheet(
                "font-size:16pt;font-weight:700;border-radius:8px;"
                "background:#f5f7fa;color:#334155;"
            )
            self._detail_label.setText("")
            self._warn_label.hide()
            return

        grade = quality_info.get("quality_grade", "?")
        bg, fg = _GRADE_COLORS.get(grade, ("#f5f7fa", "#334155"))
        self._grade_label.setText(grade)
        self._grade_label.setStyleSheet(
            f"font-size:16pt;font-weight:700;border-radius:8px;"
            f"background:{bg};color:{fg};"
        )

        dur = quality_info.get("duration_s", 0)
        snr = quality_info.get("snr_estimate_db", 0)
        rms = quality_info.get("rms_db", 0)
        clip = quality_info.get("clipping_ratio", 0)

        parts = [
            f"{dur:.1f}s",
            f"SNR {snr:.1f}dB",
            f"RMS {rms:.1f}dB",
        ]
        if clip > 0.001:
            parts.append(f"clip {clip:.2%}")
        self._detail_label.setText("  |  ".join(parts))

        warnings = []
        if dur < 1.0:
            warnings.append("时长不足 1 秒，无法使用")
        elif dur < 2.0:
            warnings.append("时长较短，建议 3 秒以上")
        if snr < 10.0:
            warnings.append("信噪比偏低，建议安静环境")
        if clip > 0.01:
            warnings.append("存在削波，建议降低音量")

        if warnings:
            self._warn_label.setText(" · ".join(warnings))
            self._warn_label.show()
        else:
            self._warn_label.hide()

    def clear(self):
        self.set_quality(None)
