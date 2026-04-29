from PyQt5.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class ProfilePage(QWidget):
    def __init__(self):
        super().__init__()

        title = QLabel("活动档案")
        title.setObjectName("PageTitle")
        hint = QLabel(
            "当前界面只维护一个活动声纹档案。"
            "Remote RKNN 模式下，本地档案与板端 active_profile.npy 会同步更新。"
        )
        hint.setWordWrap(True)
        hint.setObjectName("PageHint")

        self.name_label = QLabel("未注册")
        self.name_label.setObjectName("MetricValue")
        self.created_label = QLabel("-")
        self.created_label.setObjectName("MutedValue")
        self.samples_label = QLabel("-")
        self.samples_label.setObjectName("MutedValue")
        self.path_label = QLabel("-")
        self.path_label.setWordWrap(True)
        self.path_label.setObjectName("MutedValue")
        self.backend_label = QLabel("-")
        self.backend_label.setObjectName("MutedValue")
        self.fingerprint_label = QLabel("-")
        self.fingerprint_label.setObjectName("MutedValue")
        self.match_label = QLabel("-")
        self.match_label.setObjectName("MutedValue")

        self.delete_button = QPushButton("删除当前档案")
        self.reenroll_button = QPushButton("重新注册")

        form = QFormLayout()
        form.setSpacing(12)
        form.addRow("授权人名称", self.name_label)
        form.addRow("注册时间", self.created_label)
        form.addRow("样本数量", self.samples_label)
        form.addRow("后端模式", self.backend_label)
        form.addRow("模型指纹", self.fingerprint_label)
        form.addRow("指纹匹配", self.match_label)
        form.addRow("Embedding 路径", self.path_label)

        box = QGroupBox("当前授权档案")
        box.setLayout(form)

        actions = QHBoxLayout()
        actions.setSpacing(10)
        actions.addWidget(self.delete_button)
        actions.addWidget(self.reenroll_button)
        actions.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.addWidget(title)
        layout.addWidget(hint)
        layout.addWidget(box)
        layout.addLayout(actions)
        layout.addStretch(1)

    def set_profile(self, profile):
        if profile is None:
            self.name_label.setText("未注册")
            self.created_label.setText("-")
            self.samples_label.setText("-")
            self.path_label.setText("-")
            self.backend_label.setText("-")
            self.fingerprint_label.setText("-")
            self.match_label.setText("-")
            self.match_label.setStyleSheet("")
            return

        self.name_label.setText(profile.get("display_name", "授权用户"))
        self.created_label.setText(profile.get("created_at", "-"))
        self.samples_label.setText(str(profile.get("num_samples", "-")))
        self.path_label.setText(profile.get("embedding_path", "-"))
        self.backend_label.setText(profile.get("backend_mode", "-"))

        fp = profile.get("model_fingerprint", "-")
        if fp and len(str(fp)) > 16:
            fp = str(fp)[:16] + "..."
        self.fingerprint_label.setText(str(fp))

        if profile.get("model_fingerprint_mismatch"):
            self.match_label.setText("不匹配 — 需要重新注册")
            self.match_label.setStyleSheet("color:#b42318;font-weight:600;")
        else:
            self.match_label.setText("匹配")
            self.match_label.setStyleSheet("color:#067647;font-weight:600;")
