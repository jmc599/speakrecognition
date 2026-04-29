from PyQt5.QtWidgets import (
    QGroupBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from gui.widgets.audio_quality_card import AudioQualityCard
from gui.widgets.recording_controls import RecordingControls
from gui.widgets.score_gauge import ScoreGauge


class VerifyPage(QWidget):
    def __init__(self):
        super().__init__()

        title = QLabel("身份验证")
        title.setObjectName("PageTitle")
        hint = QLabel(
            "使用当前活动声纹档案进行同人 / 异人判定。"
            "Remote RKNN 模式下，分数与结论均由板端直接返回。"
        )
        hint.setWordWrap(True)
        hint.setObjectName("PageHint")

        self.profile_label = QLabel("未加载活动档案")
        self.profile_label.setObjectName("MetricValue")

        self.recording_controls = RecordingControls()

        self.quality_card = AudioQualityCard()

        self.status_label = QLabel("等待验证")
        self.status_label.setWordWrap(True)

        self.score_gauge = ScoreGauge()

        input_box = QGroupBox("输入")
        input_layout = QVBoxLayout(input_box)
        input_layout.setSpacing(8)
        input_layout.addWidget(self.recording_controls)
        input_layout.addWidget(self.quality_card)
        input_layout.addWidget(self.status_label)

        result_box = QGroupBox("验证结果")
        result_layout = QVBoxLayout(result_box)
        result_layout.addWidget(self.score_gauge)

        self.passphrase_expected_label = QLabel("-")
        self.passphrase_recognized_label = QLabel("-")
        self.passphrase_match_label = QLabel("Disabled")
        self.speaker_result_label = QLabel("-")
        self.final_result_label = QLabel("-")
        for label in (
            self.passphrase_expected_label,
            self.passphrase_recognized_label,
            self.passphrase_match_label,
            self.speaker_result_label,
            self.final_result_label,
        ):
            label.setWordWrap(True)

        passphrase_box = QGroupBox("Passphrase")
        passphrase_layout = QVBoxLayout(passphrase_box)
        passphrase_layout.addWidget(QLabel("Expected"))
        passphrase_layout.addWidget(self.passphrase_expected_label)
        passphrase_layout.addWidget(QLabel("Recognized"))
        passphrase_layout.addWidget(self.passphrase_recognized_label)
        passphrase_layout.addWidget(QLabel("Passphrase Match"))
        passphrase_layout.addWidget(self.passphrase_match_label)
        passphrase_layout.addWidget(QLabel("Speaker Result"))
        passphrase_layout.addWidget(self.speaker_result_label)
        passphrase_layout.addWidget(QLabel("Final Result"))
        passphrase_layout.addWidget(self.final_result_label)

        note = QLabel(
            "若导入音频的表现明显优于电脑麦克风录音，优先使用手机录音导入进行正式展示。"
        )
        note.setWordWrap(True)
        note.setObjectName("SectionNote")

        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.addWidget(title)
        layout.addWidget(hint)
        layout.addWidget(QLabel("当前授权档案"))
        layout.addWidget(self.profile_label)
        layout.addWidget(input_box)
        layout.addWidget(result_box)
        layout.addWidget(passphrase_box)
        layout.addWidget(note)
        layout.addStretch(1)

    def set_profile_name(self, display_name):
        self.profile_label.setText(display_name or "未加载活动档案")

    def set_recording(self, recording):
        self.recording_controls.set_recording(recording)

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

    def set_quality(self, quality_info):
        self.quality_card.set_quality(quality_info)

    def set_result(self, score, threshold, decision, metadata=None):
        self.score_gauge.set_result(score, threshold, decision)

    def set_passphrase_result(
        self,
        *,
        enabled,
        expected_text="",
        recognized_text="",
        matched=None,
        speaker_passed=None,
        final_passed=None,
    ):
        if not enabled:
            self.passphrase_expected_label.setText("Disabled")
            self.passphrase_recognized_label.setText("-")
            self.passphrase_match_label.setText("Disabled")
            self.speaker_result_label.setText("-")
            self.final_result_label.setText("-")
            return

        self.passphrase_expected_label.setText(expected_text or "-")
        self.passphrase_recognized_label.setText(recognized_text or "-")
        self.passphrase_match_label.setText("Match" if matched else "Mismatch")
        self.speaker_result_label.setText("Pass" if speaker_passed else "Reject")
        self.final_result_label.setText("Accept" if final_passed else "Reject")

    def clear_result(self):
        self.score_gauge.clear()
        self.quality_card.clear()
        self.set_passphrase_result(enabled=False)

    def set_controls_enabled(self, enabled):
        self.recording_controls.set_controls_enabled(enabled)
