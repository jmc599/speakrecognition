from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)


class _Card(QGroupBox):
    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        self._layout = QVBoxLayout(self)
        self._layout.setSpacing(6)

    def add_row(self, key, default="-"):
        label = QLabel(default)
        label.setWordWrap(True)
        label.setObjectName("MutedValue")
        self._layout.addWidget(QLabel(key))
        self._layout.addWidget(label)
        return label


class DashboardPage(QWidget):
    def __init__(self):
        super().__init__()

        title = QLabel("仪表盘")
        title.setObjectName("PageTitle")
        hint = QLabel("系统状态总览。所有数据均为只读展示，操作请前往对应页面。")
        hint.setWordWrap(True)
        hint.setObjectName("PageHint")

        self._backend_card = _Card("后端状态")
        self._backend_mode = self._backend_card.add_row("推理后端")
        self._backend_threshold = self._backend_card.add_row("当前阈值")
        self._backend_threshold_src = self._backend_card.add_row("阈值来源")

        self._profile_card = _Card("活动档案")
        self._profile_name = self._profile_card.add_row("授权人")
        self._profile_samples = self._profile_card.add_row("样本数")
        self._profile_created = self._profile_card.add_row("注册时间")

        self._verify_card = _Card("最近验证")
        self._verify_score = self._verify_card.add_row("分数")
        self._verify_decision = self._verify_card.add_row("结论")
        self._verify_time = self._verify_card.add_row("时间")

        self._quality_card = _Card("输入质量")
        self._quality_grade = self._quality_card.add_row("等级")
        self._quality_detail = self._quality_card.add_row("详情")

        grid = QGridLayout()
        grid.setSpacing(14)
        grid.addWidget(self._backend_card, 0, 0)
        grid.addWidget(self._profile_card, 0, 1)
        grid.addWidget(self._verify_card, 1, 0)
        grid.addWidget(self._quality_card, 1, 1)

        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.addWidget(title)
        layout.addWidget(hint)
        layout.addLayout(grid)
        layout.addStretch(1)

    def refresh(self, config, profile, last_entry, last_quality):
        backend = config.get("backend_mode", "local_onnx")
        labels = {
            "remote_rknn": "Remote RKNN",
            "wespeaker_onnx": "WeSpeaker ONNX",
            "local_pytorch": "Local PyTorch",
            "local_onnx": "Local ONNX",
        }
        self._backend_mode.setText(labels.get(backend, backend))

        th = config.get("threshold")
        self._backend_threshold.setText(f"{th:.6f}" if th is not None else "未设置")
        self._backend_threshold_src.setText(config.get("threshold_source", "unset"))

        if profile:
            self._profile_name.setText(profile.get("display_name", "-"))
            self._profile_samples.setText(str(profile.get("num_samples", "-")))
            self._profile_created.setText(profile.get("created_at", "-"))
        else:
            self._profile_name.setText("未注册")
            self._profile_samples.setText("-")
            self._profile_created.setText("-")

        if last_entry:
            score = last_entry.get("score", "")
            self._verify_score.setText(f"{float(score):.6f}" if score != "" else "-")
            self._verify_decision.setText(str(last_entry.get("decision", "-")))
            self._verify_time.setText(str(last_entry.get("created_at", "-")))
        else:
            self._verify_score.setText("-")
            self._verify_decision.setText("-")
            self._verify_time.setText("-")

        if last_quality:
            self._quality_grade.setText(last_quality.get("quality_grade", "-"))
            dur = last_quality.get("duration_s", 0)
            snr = last_quality.get("snr_estimate_db", 0)
            self._quality_detail.setText(f"{dur:.1f}s  |  SNR {snr:.1f}dB")
        else:
            self._quality_grade.setText("-")
            self._quality_detail.setText("-")
