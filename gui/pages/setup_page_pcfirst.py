from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from gui.services.config_store import (
    DEFAULT_THRESHOLD_POLICY,
    THRESHOLD_SOURCE_MANUAL,
    THRESHOLD_SOURCE_UNSET,
)
from gui.services.passphrase_match import MATCH_MODE_CONTAINS, MATCH_MODE_EXACT
from gui.widgets import NoWheelComboBox, NoWheelDoubleSpinBox, NoWheelSpinBox

_BACKEND_INDEX = {
    "local_onnx": 0,
    "local_pytorch": 1,
    "wespeaker_onnx": 2,
    "remote_rknn": 3,
}


class SetupPage(QWidget):
    def __init__(self):
        super().__init__()

        title = QLabel("系统设置")
        title.setObjectName("PageTitle")
        hint = QLabel(
            "配置推理后端、板端连接与验证阈值。"
            "切换后端时只显示对应配置项。"
        )
        hint.setWordWrap(True)
        hint.setObjectName("PageHint")

        # -- general --
        self.backend_combo = NoWheelComboBox()
        self.backend_combo.addItem("本地 ONNX", "local_onnx")
        self.backend_combo.addItem("本地 PyTorch", "local_pytorch")
        self.backend_combo.addItem("本地 WeSpeaker ONNX", "wespeaker_onnx")
        self.backend_combo.addItem("远端 RKNN", "remote_rknn")

        self.device_combo = NoWheelComboBox()

        general_form = QFormLayout()
        general_form.setSpacing(12)
        general_form.addRow("推理后端", self.backend_combo)
        general_form.addRow("输入设备", self.device_combo)
        general_box = QGroupBox("运行模式")
        general_box.setLayout(general_form)

        # -- backend-specific stacked pages --
        self.local_model_edit = QLineEdit()
        self.local_model_edit.setPlaceholderText("例如：checkpoints/resnet_v7....onnx")

        self.local_pytorch_edit = QLineEdit()
        self.local_pytorch_edit.setPlaceholderText("例如：checkpoints/resnet_v7....pth")

        self.wespeaker_model_edit = QLineEdit()
        self.wespeaker_model_edit.setPlaceholderText("例如：checkpoints/wespeaker....onnx")

        self.host_edit = QLineEdit()
        self.host_edit.setPlaceholderText("192.168.50.2")
        self.port_spin = NoWheelSpinBox()
        self.port_spin.setRange(1, 65535)
        self.user_edit = QLineEdit()
        self.user_edit.setPlaceholderText("root")
        self.password_edit = QLineEdit()
        self.password_edit.setEchoMode(QLineEdit.Password)
        self.password_edit.setPlaceholderText("如需自动连接可保存密码")
        self.model_edit = QLineEdit()
        self.model_edit.setPlaceholderText("/root/models/resnet_v7_vox2ft_s1_latest_fixed_rt160_nonorm_v232.rknn")
        self.script_edit = QLineEdit()
        self.script_edit.setPlaceholderText("/root/models/rknn_remote_runner.py")
        self.workdir_edit = QLineEdit()
        self.workdir_edit.setPlaceholderText("/root/models/gui_jobs")
        self.python_edit = QLineEdit()
        self.python_edit.setPlaceholderText("python3")
        self.transport_combo = NoWheelComboBox()
        self.transport_combo.addItem("HTTP first (fallback SSH)", "http_first")
        self.transport_combo.addItem("SSH only", "ssh_only")
        self.http_base_url_edit = QLineEdit()
        self.http_base_url_edit.setPlaceholderText("http://192.168.50.2:8765")

        # stack page 0: local_onnx
        onnx_page = QWidget()
        onnx_form = QFormLayout(onnx_page)
        onnx_form.setSpacing(12)
        onnx_form.addRow("本地 ONNX 模型", self.local_model_edit)

        # stack page 1: local_pytorch
        pytorch_page = QWidget()
        pytorch_form = QFormLayout(pytorch_page)
        pytorch_form.setSpacing(12)
        pytorch_form.addRow("本地 PyTorch 检查点", self.local_pytorch_edit)

        # stack page 2: wespeaker_onnx
        wespeaker_page = QWidget()
        wespeaker_form = QFormLayout(wespeaker_page)
        wespeaker_form.setSpacing(12)
        wespeaker_form.addRow("WeSpeaker ONNX 模型", self.wespeaker_model_edit)

        # stack page 3: remote_rknn
        remote_page = QWidget()
        remote_form = QFormLayout(remote_page)
        remote_form.setSpacing(12)
        remote_form.addRow("板端 IP", self.host_edit)
        remote_form.addRow("SSH 端口", self.port_spin)
        remote_form.addRow("用户名", self.user_edit)
        remote_form.addRow("密码", self.password_edit)
        remote_form.addRow("远端 RKNN 模型", self.model_edit)
        remote_form.addRow("Remote Runner Script", self.script_edit)
        remote_form.addRow("Remote Workdir", self.workdir_edit)
        remote_form.addRow("Remote Python", self.python_edit)
        remote_form.addRow("Remote Transport", self.transport_combo)
        remote_form.addRow("Remote HTTP Base URL", self.http_base_url_edit)

        self._backend_stack = QStackedWidget()
        self._backend_stack.addWidget(onnx_page)
        self._backend_stack.addWidget(pytorch_page)
        self._backend_stack.addWidget(wespeaker_page)
        self._backend_stack.addWidget(remote_page)

        backend_box = QGroupBox("后端配置")
        backend_layout = QVBoxLayout(backend_box)
        backend_layout.addWidget(self._backend_stack)

        # PLACEHOLDER_THRESHOLD_AND_ACTIONS

        # -- threshold & enroll --
        self.threshold_spin = NoWheelDoubleSpinBox()
        self.threshold_spin.setDecimals(6)
        self.threshold_spin.setRange(-1.0, 1.0)
        self.threshold_spin.setSingleStep(0.01)
        self.enroll_spin = NoWheelSpinBox()
        self.enroll_spin.setRange(1, 10)
        self.wespeaker_calibrated_label = QLabel("否")
        self.wespeaker_calibrated_label.setObjectName("MutedValue")

        verify_form = QFormLayout()
        verify_form.setSpacing(12)
        verify_form.addRow("当前阈值", self.threshold_spin)
        verify_form.addRow("注册样本数", self.enroll_spin)
        verify_form.addRow("WeSpeaker 已标定", self.wespeaker_calibrated_label)
        verify_box = QGroupBox("验证参数")
        verify_box.setLayout(verify_form)

        self.passphrase_enabled_check = QCheckBox("Enable passphrase verification")
        self.passphrase_edit = QLineEdit()
        self.passphrase_edit.setPlaceholderText("Enter the expected passphrase")
        self.passphrase_mode_combo = NoWheelComboBox()
        self.passphrase_mode_combo.addItem("Normalized exact", MATCH_MODE_EXACT)
        self.passphrase_mode_combo.addItem("Normalized contains", MATCH_MODE_CONTAINS)
        self.asr_model_edit = QLineEdit()
        self.asr_model_edit.setPlaceholderText("iic/SenseVoiceSmall")
        self.asr_language_edit = QLineEdit()
        self.asr_language_edit.setPlaceholderText("zh")

        passphrase_form = QFormLayout()
        passphrase_form.setSpacing(12)
        passphrase_form.addRow(self.passphrase_enabled_check)
        passphrase_form.addRow("Expected Passphrase", self.passphrase_edit)
        passphrase_form.addRow("Passphrase Match", self.passphrase_mode_combo)
        passphrase_form.addRow("ASR Model", self.asr_model_edit)
        passphrase_form.addRow("ASR Language", self.asr_language_edit)
        passphrase_box = QGroupBox("Passphrase Verification")
        passphrase_box.setLayout(passphrase_form)

        note = QLabel("如果只是做 GUI 烟测，可以把注册样本数临时设为 1；正式演示再恢复为 3。")
        note.setWordWrap(True)
        note.setObjectName("SectionNote")

        # -- actions --
        self.save_button = QPushButton("保存配置")
        self.test_button = QPushButton("测试后端")
        self.status_label = QLabel("尚未测试")
        self.status_label.setWordWrap(True)
        self.status_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        actions = QHBoxLayout()
        actions.setSpacing(10)
        actions.addWidget(self.save_button)
        actions.addWidget(self.test_button)
        actions.addStretch(1)

        status_box = QGroupBox("后端状态")
        status_layout = QVBoxLayout(status_box)
        status_layout.addWidget(self.status_label)

        # -- main layout --
        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.addWidget(title)
        layout.addWidget(hint)
        layout.addWidget(general_box)
        layout.addWidget(backend_box)
        layout.addWidget(verify_box)
        layout.addWidget(passphrase_box)
        layout.addWidget(note)
        layout.addLayout(actions)
        layout.addWidget(status_box)
        layout.addStretch(1)

        # -- per-backend threshold state --
        self._backend_thresholds = {k: None for k in _BACKEND_INDEX}
        self._backend_threshold_unset = {k: True for k in _BACKEND_INDEX}
        self._backend_threshold_policies = {k: DEFAULT_THRESHOLD_POLICY for k in _BACKEND_INDEX}
        self._backend_threshold_sources = {k: THRESHOLD_SOURCE_UNSET for k in _BACKEND_INDEX}
        self._backend_threshold_calibrated = {k: False for k in _BACKEND_INDEX}

        # -- signals --
        self.backend_combo.currentIndexChanged.connect(self._on_backend_changed)
        self.threshold_spin.valueChanged.connect(self._on_threshold_changed)

    # -- public API (unchanged contract) --

    def set_devices(self, device_names):
        current_index = self.current_device_index()
        current_name = self.current_device_name()
        self.device_combo.clear()
        self.device_combo.addItem("", {"index": None, "name": ""})
        for device in device_names:
            if isinstance(device, dict):
                label = device.get("label", device.get("name", ""))
                self.device_combo.addItem(label, device)
            else:
                self.device_combo.addItem(str(device), {"index": None, "name": str(device)})

        target_row = -1
        if current_index is not None:
            target_row = self._find_device_row_by_index(current_index)
        if target_row < 0 and current_name:
            target_row = self._find_device_row_by_name(current_name)
        if target_row >= 0:
            self.device_combo.setCurrentIndex(target_row)

    def set_config(self, config):
        backend_mode = config.get("backend_mode", "local_onnx")
        index = self.backend_combo.findData(backend_mode)
        if index >= 0:
            self.backend_combo.setCurrentIndex(index)

        self._set_backend_threshold_state("local_onnx",
            threshold=config.get("local_onnx_threshold"),
            threshold_policy=config.get("local_onnx_threshold_policy", DEFAULT_THRESHOLD_POLICY),
            threshold_source=config.get("local_onnx_threshold_source", THRESHOLD_SOURCE_UNSET),
            calibrated=config.get("local_onnx_threshold_calibrated", False))
        self._set_backend_threshold_state("local_pytorch",
            threshold=config.get("local_pytorch_threshold"),
            threshold_policy=config.get("local_pytorch_threshold_policy", DEFAULT_THRESHOLD_POLICY),
            threshold_source=config.get("local_pytorch_threshold_source", THRESHOLD_SOURCE_UNSET),
            calibrated=config.get("local_pytorch_threshold_calibrated", False))
        self._set_backend_threshold_state("wespeaker_onnx",
            threshold=config.get("wespeaker_onnx_threshold"),
            threshold_policy=config.get("wespeaker_onnx_threshold_policy", DEFAULT_THRESHOLD_POLICY),
            threshold_source=config.get("wespeaker_onnx_threshold_source", THRESHOLD_SOURCE_UNSET),
            calibrated=config.get("wespeaker_threshold_calibrated", False))
        self._set_backend_threshold_state("remote_rknn",
            threshold=config.get("remote_rknn_threshold"),
            threshold_policy=config.get("remote_rknn_threshold_policy", DEFAULT_THRESHOLD_POLICY),
            threshold_source=config.get("remote_rknn_threshold_source", THRESHOLD_SOURCE_UNSET),
            calibrated=config.get("remote_rknn_threshold_calibrated", False))

        self.host_edit.setText(config.get("board_host", ""))
        self.port_spin.setValue(int(config.get("board_port", 22)))
        self.user_edit.setText(config.get("board_username", "root"))
        self.password_edit.setText(config.get("board_password", ""))
        self.local_model_edit.setText(config.get("local_onnx_model_path", ""))
        self.local_pytorch_edit.setText(config.get("local_pytorch_checkpoint_path", ""))
        self.wespeaker_model_edit.setText(config.get("wespeaker_onnx_model_path", ""))
        self.model_edit.setText(config.get("remote_model_path", ""))
        self.script_edit.setText(config.get("remote_runner_script", config.get("remote_embed_script", "")))
        self.workdir_edit.setText(config.get("remote_workdir", ""))
        self.python_edit.setText(config.get("remote_python", "python3"))
        transport_index = self.transport_combo.findData(
            config.get("remote_transport_preference", "http_first")
        )
        if transport_index >= 0:
            self.transport_combo.setCurrentIndex(transport_index)
        self.http_base_url_edit.setText(config.get("remote_http_base_url", ""))
        self.enroll_spin.setValue(int(config.get("enroll_samples", 3)))
        self.passphrase_enabled_check.setChecked(
            bool(config.get("passphrase_verification_enabled", False))
        )
        self.passphrase_edit.setText(config.get("expected_passphrase", ""))
        passphrase_mode_index = self.passphrase_mode_combo.findData(
            config.get("passphrase_match_mode", MATCH_MODE_EXACT)
        )
        if passphrase_mode_index >= 0:
            self.passphrase_mode_combo.setCurrentIndex(passphrase_mode_index)
        self.asr_model_edit.setText(config.get("asr_model_name", "iic/SenseVoiceSmall"))
        self.asr_language_edit.setText(config.get("asr_language", "zh"))
        self.wespeaker_calibrated_label.setText(
            "是" if self._backend_threshold_calibrated["wespeaker_onnx"] else "否"
        )
        self._sync_threshold_for_backend()

        device_index = config.get("device_index")
        device_name = config.get("device_name", "")
        row = self._find_device_row_by_index(device_index)
        if row < 0 and device_name:
            row = self._find_device_row_by_name(device_name)
        if row >= 0:
            self.device_combo.setCurrentIndex(row)

    # PLACEHOLDER_GET_CONFIG

    def get_config(self):
        return {
            "backend_mode": self.backend_combo.currentData(),
            "device_name": self.current_device_name(),
            "device_index": self.current_device_index(),
            "local_onnx_model_path": self.local_model_edit.text().strip(),
            "local_pytorch_checkpoint_path": self.local_pytorch_edit.text().strip(),
            "wespeaker_onnx_model_path": self.wespeaker_model_edit.text().strip(),
            "board_host": self.host_edit.text().strip(),
            "board_port": self.port_spin.value(),
            "board_username": self.user_edit.text().strip(),
            "board_password": self.password_edit.text(),
            "remote_model_path": self.model_edit.text().strip(),
            "remote_runner_script": self.script_edit.text().strip(),
            "remote_embed_script": self.script_edit.text().strip(),
            "remote_workdir": self.workdir_edit.text().strip(),
            "remote_python": self.python_edit.text().strip(),
            "remote_transport_preference": self.transport_combo.currentData(),
            "remote_http_base_url": self.http_base_url_edit.text().strip(),
            "threshold": self._current_backend_threshold_value(),
            "local_onnx_threshold": self._backend_thresholds["local_onnx"],
            "local_pytorch_threshold": self._backend_thresholds["local_pytorch"],
            "wespeaker_onnx_threshold": self._backend_thresholds["wespeaker_onnx"],
            "remote_rknn_threshold": self._backend_thresholds["remote_rknn"],
            "local_onnx_threshold_policy": self._backend_threshold_policies["local_onnx"],
            "local_pytorch_threshold_policy": self._backend_threshold_policies["local_pytorch"],
            "wespeaker_onnx_threshold_policy": self._backend_threshold_policies["wespeaker_onnx"],
            "remote_rknn_threshold_policy": self._backend_threshold_policies["remote_rknn"],
            "local_onnx_threshold_source": self._backend_threshold_sources["local_onnx"],
            "local_pytorch_threshold_source": self._backend_threshold_sources["local_pytorch"],
            "wespeaker_onnx_threshold_source": self._backend_threshold_sources["wespeaker_onnx"],
            "remote_rknn_threshold_source": self._backend_threshold_sources["remote_rknn"],
            "local_onnx_threshold_calibrated": self._backend_threshold_calibrated["local_onnx"],
            "local_pytorch_threshold_calibrated": self._backend_threshold_calibrated["local_pytorch"],
            "remote_rknn_threshold_calibrated": self._backend_threshold_calibrated["remote_rknn"],
            "wespeaker_threshold_calibrated": self._backend_threshold_calibrated["wespeaker_onnx"],
            "enroll_samples": self.enroll_spin.value(),
            "passphrase_verification_enabled": self.passphrase_enabled_check.isChecked(),
            "expected_passphrase": self.passphrase_edit.text().strip(),
            "asr_backend": "sensevoice_onnx",
            "asr_model_name": self.asr_model_edit.text().strip() or "iic/SenseVoiceSmall",
            "asr_language": self.asr_language_edit.text().strip() or "zh",
            "passphrase_match_mode": self.passphrase_mode_combo.currentData(),
        }

    def set_status(self, text, ok=None):
        color = "#334155"
        if ok is True:
            color = "#067647"
        elif ok is False:
            color = "#b42318"
        self.status_label.setStyleSheet(f"color: {color}; font-weight: 600;")
        self.status_label.setText(text)

    # -- internal --

    def _on_backend_changed(self):
        backend_mode = self.backend_combo.currentData()
        idx = _BACKEND_INDEX.get(backend_mode, 0)
        self._backend_stack.setCurrentIndex(idx)
        self._sync_threshold_for_backend()

    def _sync_threshold_for_backend(self):
        backend_mode = self.backend_combo.currentData()
        if backend_mode not in self._backend_thresholds:
            return
        self.threshold_spin.blockSignals(True)
        value = self._backend_thresholds[backend_mode]
        self.threshold_spin.setValue(0.0 if value is None else float(value))
        self.threshold_spin.blockSignals(False)

    def _on_threshold_changed(self, value):
        backend_mode = self.backend_combo.currentData()
        if backend_mode not in self._backend_thresholds:
            return
        self._backend_thresholds[backend_mode] = float(value)
        self._backend_threshold_unset[backend_mode] = False
        self._backend_threshold_sources[backend_mode] = THRESHOLD_SOURCE_MANUAL
        self._backend_threshold_calibrated[backend_mode] = False
        if backend_mode == "wespeaker_onnx":
            self.wespeaker_calibrated_label.setText("否")

    def _set_backend_threshold_state(self, backend_mode, *, threshold, threshold_policy, threshold_source, calibrated):
        value = None if threshold in (None, "") else float(threshold)
        self._backend_thresholds[backend_mode] = value
        self._backend_threshold_unset[backend_mode] = value is None
        self._backend_threshold_policies[backend_mode] = str(threshold_policy or DEFAULT_THRESHOLD_POLICY)
        self._backend_threshold_sources[backend_mode] = str(threshold_source or THRESHOLD_SOURCE_UNSET)
        self._backend_threshold_calibrated[backend_mode] = bool(calibrated)

    def _current_backend_threshold_value(self):
        backend_mode = self.backend_combo.currentData()
        if backend_mode not in self._backend_thresholds:
            return None
        if self._backend_threshold_unset.get(backend_mode, False):
            return None
        return self._backend_thresholds[backend_mode]

    def current_device_name(self):
        data = self.device_combo.currentData()
        if isinstance(data, dict):
            return data.get("name", "")
        return self.device_combo.currentText()

    def current_device_index(self):
        data = self.device_combo.currentData()
        if isinstance(data, dict):
            return data.get("index")
        return None

    def _find_device_row_by_index(self, target_index):
        if target_index in ("", None):
            return -1
        for row in range(self.device_combo.count()):
            data = self.device_combo.itemData(row)
            if isinstance(data, dict) and data.get("index") == target_index:
                return row
        return -1

    def _find_device_row_by_name(self, target_name):
        for row in range(self.device_combo.count()):
            data = self.device_combo.itemData(row)
            if isinstance(data, dict) and data.get("name") == target_name:
                return row
        return -1
