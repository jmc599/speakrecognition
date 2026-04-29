import json
from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QStackedWidget,
    QStatusBar,
    QWidget,
)

from gui.pages.dashboard_page import DashboardPage
from gui.pages.enroll_page import EnrollPage
from gui.pages.history_page import HistoryPage
from gui.pages.profile_page import ProfilePage
from gui.pages.setup_page_pcfirst import SetupPage
from gui.pages.verify_page import VerifyPage
from gui.services.audio_pipeline import normalize_audio_file
from gui.services.audio_quality import analyze_audio_quality
from gui.services.asr_service import AsrService
from gui.services.board_client import BoardClient
from gui.services.config_store import ConfigStore, clear_backend_threshold
from gui.services.feature_builder import build_feature_artifacts
from gui.services.history_store import HistoryStore
from gui.services.model_fingerprint import resolve_backend_model_fingerprint
from gui.services.passphrase_match import match_passphrase
from gui.services.onnx_local_verifier import OnnxLocalVerifier
from gui.services.paths import local_tmp_dir
from gui.services.profile_store import ProfileStore, build_merged_embedding
from gui.services.pytorch_local_verifier import PytorchLocalVerifier
from gui.services.recorder import Recorder
from gui.services.wespeaker_onnx_verifier import WespeakerOnnxVerifier
from gui.services.worker import TaskThread
from gui.widgets.sidebar import SidebarNav


def cosine_score(emb_a, emb_b):
    emb_a = np.asarray(emb_a, dtype=np.float32).reshape(1, -1)
    emb_b = np.asarray(emb_b, dtype=np.float32).reshape(1, -1)
    return float(np.sum(emb_a * emb_b, axis=1)[0])


_PAGE_LABELS = ["仪表盘", "注册", "验证", "历史", "档案", "设置"]
_IDX_DASHBOARD = 0
_IDX_ENROLL = 1
_IDX_VERIFY = 2
_IDX_HISTORY = 3
_IDX_PROFILE = 4
_IDX_SETUP = 5


class SpeakerIdentityTerminalWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Speaker Identity Terminal")
        self.resize(1140, 820)

        self.config_store = ConfigStore()
        self.config = self.config_store.load()
        self.profile_store = ProfileStore()
        self.history_store = HistoryStore()
        self.recorder = Recorder(
            device_name=self.config.get("device_name", ""),
            device_index=self.config.get("device_index"),
        )
        self._remote_board_client = None
        self._asr_service = None
        self._task_thread = None
        self._record_mode = None
        self._pending_enroll_embeddings = []
        self._recording_seconds = 0
        self._last_quality = None

        self.dashboard_page = DashboardPage()
        self.enroll_page = EnrollPage()
        self.verify_page = VerifyPage()
        self.history_page = HistoryPage()
        self.profile_page = ProfilePage()
        self.setup_page = SetupPage()

        self.sidebar = SidebarNav(_PAGE_LABELS)
        self.stack = QStackedWidget()
        for page in (
            self.dashboard_page,
            self.enroll_page,
            self.verify_page,
            self.history_page,
            self.profile_page,
            self.setup_page,
        ):
            self.stack.addWidget(self._wrap_scroll(page))

        container = QWidget()
        hlayout = QHBoxLayout(container)
        hlayout.setContentsMargins(0, 0, 0, 0)
        hlayout.setSpacing(0)
        hlayout.addWidget(self.sidebar)
        hlayout.addWidget(self.stack, 1)
        self.setCentralWidget(container)
        self.setStatusBar(QStatusBar())

        self._record_timer = QTimer(self)
        self._record_timer.setInterval(1000)
        self._record_timer.timeout.connect(self._on_record_tick)

        self._bind_signals()
        self._load_devices()
        self.setup_page.set_config(self.config)
        self._reset_enroll_session()
        self._refresh_profile()
        self._refresh_history()
        self._sync_backend_status()
        self.sidebar.select(_IDX_DASHBOARD)

    # -- layout helpers --

    def _wrap_scroll(self, widget):
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.NoFrame)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        area.setWidget(widget)
        return area

    # PLACEHOLDER_BIND_SIGNALS

    def _bind_signals(self):
        self.sidebar.page_selected.connect(self.stack.setCurrentIndex)
        self.recorder.level_changed.connect(self._on_level_changed)

        self.setup_page.save_button.clicked.connect(self._save_config)
        self.setup_page.test_button.clicked.connect(self._test_connection)

        self.enroll_page.recording_controls.start_clicked.connect(self._start_enroll_recording)
        self.enroll_page.recording_controls.stop_clicked.connect(self._stop_enroll_recording)
        self.enroll_page.recording_controls.import_clicked.connect(self._import_enroll_audio)
        self.enroll_page.reset_button.clicked.connect(self._reset_enroll_session)

        self.verify_page.recording_controls.start_clicked.connect(self._start_verify_recording)
        self.verify_page.recording_controls.stop_clicked.connect(self._stop_verify_recording)
        self.verify_page.recording_controls.import_clicked.connect(self._import_verify_audio)

        self.history_page.refresh_button.clicked.connect(self._refresh_history)
        self.history_page.export_button.clicked.connect(self._export_history)
        self.history_page.delete_button.clicked.connect(self._delete_selected_history)
        self.history_page.clear_button.clicked.connect(self._clear_history)

        self.profile_page.delete_button.clicked.connect(self._delete_active_profile)
        self.profile_page.reenroll_button.clicked.connect(self._go_to_enroll_page)

    # -- device enumeration --

    def _load_devices(self):
        try:
            devices = Recorder.list_input_devices()
        except Exception as exc:
            devices = []
            self.statusBar().showMessage(f"Failed to enumerate audio devices: {exc}")
        self.setup_page.set_devices(devices)

    # -- backend helpers --

    def _backend_client(self):
        backend_mode = self.config.get("backend_mode", "local_onnx")
        if backend_mode == "remote_rknn":
            if self._remote_board_client is None:
                self._remote_board_client = BoardClient(self.config)
            else:
                self._remote_board_client.config = self.config
            return self._remote_board_client
        if backend_mode == "wespeaker_onnx":
            return WespeakerOnnxVerifier(self.config)
        if backend_mode == "local_pytorch":
            return PytorchLocalVerifier(self.config)
        return OnnxLocalVerifier(self.config)

    def _invalidate_remote_board_client(self):
        if self._remote_board_client is not None:
            self._remote_board_client.close()
            self._remote_board_client = None

    def _asr_client(self):
        if self._asr_service is None:
            self._asr_service = AsrService(self.config)
        else:
            self._asr_service.config = self.config
        return self._asr_service

    def _invalidate_asr_service(self):
        self._asr_service = None

    @staticmethod
    def _asr_config_changed(old_config, new_config):
        watched_keys = (
            "passphrase_verification_enabled",
            "expected_passphrase",
            "asr_backend",
            "asr_model_name",
            "asr_language",
            "asr_cache_dir",
            "passphrase_match_mode",
        )
        return any(old_config.get(key) != new_config.get(key) for key in watched_keys)

    def _backend_label(self):
        backend_mode = self.config.get("backend_mode", "local_onnx")
        labels = {
            "remote_rknn": "Remote RKNN",
            "wespeaker_onnx": "Local WeSpeaker ONNX",
            "local_pytorch": "Local PyTorch",
        }
        return labels.get(backend_mode, "Local ONNX")

    def _sync_backend_status(self):
        label = self._backend_label()
        self.statusBar().showMessage(f"Current backend: {label}")
        self.sidebar.backend_indicator.set_state("ok", label)
        profile = self._get_active_profile()
        if profile:
            self.sidebar.profile_indicator.set_state(
                "ok", profile.get("display_name", "已注册")
            )
        else:
            self.sidebar.profile_indicator.set_state("off", "未注册")
        th = self.config.get("threshold")
        src = self.config.get("threshold_source", "unset")
        if th is not None:
            self.sidebar.threshold_indicator.set_state("ok", f"阈值 {th:.4f} ({src})")
        else:
            self.sidebar.threshold_indicator.set_state("warn", "阈值未设置")

    def _current_backend_mode(self):
        return self.config.get("backend_mode", "local_onnx")

    def _current_model_fingerprint(self):
        return resolve_backend_model_fingerprint(self.config)

    def _get_active_profile(self, allow_model_mismatch=False):
        return self.profile_store.get_active_profile(
            backend_mode=self._current_backend_mode(),
            model_fingerprint=self._current_model_fingerprint(),
            allow_model_mismatch=allow_model_mismatch,
        )

    # PLACEHOLDER_THRESHOLD_CHECKS

    def _require_wespeaker_threshold_calibrated(self):
        if self._current_backend_mode() != "wespeaker_onnx":
            return True
        if bool(self.config.get("wespeaker_threshold_calibrated", False)):
            return True
        self.verify_page.set_status("Calibrate threshold first", ok=False)
        QMessageBox.warning(
            self, "Threshold not calibrated",
            "WeSpeaker backend has not completed profile threshold calibration yet.",
        )
        return False

    def _require_active_threshold(self):
        if self.config.get("threshold") is not None:
            return True
        self.verify_page.set_status("Please calibrate or set a threshold first", ok=False)
        QMessageBox.warning(
            self, "Threshold Missing",
            f"{self._backend_label()} does not have an active threshold yet.",
        )
        return False

    def _ensure_profile_ready_for_verify(self):
        profile = self._get_active_profile(allow_model_mismatch=True)
        if profile is None:
            QMessageBox.warning(self, "No active profile", "Complete enrollment first.")
            return None
        if profile.get("model_fingerprint_mismatch"):
            self.verify_page.set_status("Model changed. Re-enroll required.", ok=False)
            QMessageBox.warning(
                self, "Model changed",
                "The active profile was created by a different model. Re-enroll first.",
            )
            return None
        if profile.get("embedding") is None:
            self.verify_page.set_status("Profile embedding missing", ok=False)
            QMessageBox.warning(
                self, "Profile missing",
                "The active profile embedding is missing. Re-enroll first.",
            )
            return None
        return profile

    def _backend_model_changed(self, old_config, new_config, backend_mode):
        old_cfg = dict(old_config)
        old_cfg["backend_mode"] = backend_mode
        new_cfg = dict(new_config)
        new_cfg["backend_mode"] = backend_mode
        return resolve_backend_model_fingerprint(old_cfg) != resolve_backend_model_fingerprint(new_cfg)

    def _reset_stale_thresholds(self, config):
        for backend_mode in ("local_onnx", "local_pytorch", "wespeaker_onnx", "remote_rknn"):
            if self._backend_model_changed(self.config, config, backend_mode):
                clear_backend_threshold(config, backend_mode)
                if backend_mode == "wespeaker_onnx":
                    config["wespeaker_threshold_calibrated"] = False

    # -- config save / test --

    def _save_config(self):
        config = self.setup_page.get_config()
        backend_mode = config["backend_mode"]

        if backend_mode == "local_onnx" and not config["local_onnx_model_path"]:
            QMessageBox.warning(self, "Incomplete config", "Local ONNX model path is required.")
            return False
        if backend_mode == "local_pytorch" and not config["local_pytorch_checkpoint_path"]:
            QMessageBox.warning(self, "Incomplete config", "Local PyTorch checkpoint path is required.")
            return False
        if backend_mode == "wespeaker_onnx" and not config["wespeaker_onnx_model_path"]:
            QMessageBox.warning(self, "Incomplete config", "WeSpeaker ONNX model path is required.")
            return False
        if backend_mode == "remote_rknn":
            if not config["board_host"]:
                QMessageBox.warning(self, "Incomplete config", "Board host is required.")
                return False
            if not config["board_username"]:
                QMessageBox.warning(self, "Incomplete config", "SSH username is required.")
                return False
            if not config["remote_model_path"] or not config["remote_runner_script"]:
                QMessageBox.warning(self, "Incomplete config", "Remote model path and runner script are required.")
                return False
        if config.get("passphrase_verification_enabled") and not config.get("expected_passphrase"):
            QMessageBox.warning(
                self,
                "Incomplete config",
                "Expected passphrase is required when passphrase verification is enabled.",
            )
            return False

        self._reset_stale_thresholds(config)
        self._invalidate_remote_board_client()
        if self._asr_config_changed(self.config, config):
            self._invalidate_asr_service()
        self.config = self.config_store.save(config)
        self.recorder.set_device(
            device_name=self.config.get("device_name", ""),
            device_index=self.config.get("device_index"),
        )
        self.setup_page.set_config(self.config)
        self.setup_page.set_status(
            f"Configuration saved. Current backend: {self._backend_label()}", ok=True,
        )
        self._sync_backend_status()
        self._refresh_profile()
        self._refresh_dashboard()
        return True

    def _test_connection(self):
        if not self._save_config():
            return

        def task():
            return self._backend_client().test_connection()

        self.setup_page.set_status("Testing backend...", ok=None)
        self._run_task(task, self._on_connection_success, self._on_connection_failed)

    def _on_connection_success(self, result):
        mode = result.get("mode")
        if mode == "local_onnx":
            text = f"Local ONNX ready\nModel: {result['model_path']}"
        elif mode == "local_pytorch":
            text = f"Local PyTorch ready\nCheckpoint: {result['checkpoint_path']}"
        elif mode == "wespeaker_onnx":
            text = (
                f"WeSpeaker ONNX ready\n"
                f"Model: {result['model_path']}\n"
                f"Input: {result['input_name']}\n"
                f"Probe norm: {result['probe_embedding_norm']:.6f}\n"
                f"Internal L2: {'yes' if result.get('model_has_internal_l2') else 'no'}"
            )
        else:
            transport = result.get("transport", "ssh")
            transport_line = (
                f"Transport: HTTP ({result.get('http_base_url', '')})"
                if transport == "http"
                else "Transport: SSH runner"
            )
            framework_line = ""
            if transport == "http" and result.get("service_framework"):
                framework_line = f"\nFramework: {result['service_framework']}"
            fingerprint_line = ""
            if result.get("model_fingerprint"):
                fingerprint_line = f"\nModel fingerprint: {result['model_fingerprint']}"
            fallback_line = ""
            if result.get("transport_fallback_reason"):
                fallback_line = f"\nHTTP fallback reason: {result['transport_fallback_reason']}"
            text = (
                f"Remote RKNN ready\n"
                f"Host: {result['board_host']}\n"
                f"{transport_line}\n"
                f"Python: {result['python_version']}\n"
                f"Model: {result['model_path']}\n"
                f"Runner: {result['runner_script']}{framework_line}{fingerprint_line}{fallback_line}"
            )
        self.setup_page.set_status(text, ok=True)
        self.statusBar().showMessage("Backend test passed", 3000)
        self.sidebar.backend_indicator.set_state("ok", self._backend_label())

    def _on_connection_failed(self, message):
        self.setup_page.set_status(message, ok=False)
        QMessageBox.critical(self, "Backend test failed", message)

    # PLACEHOLDER_ENROLL_FLOW

    # -- enroll flow --

    def _start_enroll_recording(self):
        self._start_recording("enroll")

    def _stop_enroll_recording(self):
        audio_path = self._stop_recording()
        if audio_path:
            self._process_enroll_audio(audio_path)

    def _import_enroll_audio(self):
        audio_path = self._choose_audio_file()
        if audio_path:
            self._process_enroll_audio(audio_path)

    def _process_enroll_audio(self, audio_path):
        self.enroll_page.set_status(f"Processing enrollment audio ({self._backend_label()})...", ok=None)
        self.statusBar().showMessage("Processing enrollment audio")

        def task():
            result = self._build_backend_artifacts(audio_path, prefix="enroll")
            try:
                result["quality"] = analyze_audio_quality(result["normalized_audio_path"])
            except Exception:
                result["quality"] = None
            return result

        self._run_task(task, self._on_enroll_processed, self._on_task_failed)

    def _on_enroll_processed(self, result):
        quality = result.get("quality")
        if quality and quality.get("duration_s", 0) < 1.0:
            self.enroll_page.set_status("音频不足 1 秒，已拒绝", ok=False)
            return

        self._pending_enroll_embeddings.append(result["embedding"])
        current = len(self._pending_enroll_embeddings)
        total = int(self.config["enroll_samples"])

        self.enroll_page.add_sample(result["normalized_audio_path"], quality)
        self.enroll_page.set_progress(current, total)

        if current < total:
            self.enroll_page.set_status(f"Captured {current}/{total} samples", ok=True)
            return

        display_name = self.enroll_page.display_name()
        embeddings = np.concatenate(self._pending_enroll_embeddings, axis=0)

        if self._current_backend_mode() == "remote_rknn":
            self.enroll_page.set_status("Publishing remote active profile...", ok=None)

            def task():
                return self._save_remote_active_profile(display_name, embeddings, total)

            self._run_task(task, self._on_remote_enroll_saved, self._on_remote_enroll_failed)
            return

        profile = self.profile_store.save_active_profile(
            display_name=display_name,
            embeddings=embeddings,
            num_samples=total,
            backend_mode=self._current_backend_mode(),
            model_fingerprint=self._current_model_fingerprint(),
        )
        self._reset_enroll_session(reset_status=False)
        self._refresh_profile(profile)
        self.verify_page.set_status("Ready for verification", ok=None)
        self.enroll_page.set_status("Enrollment completed", ok=True)
        self._refresh_dashboard()
        QMessageBox.information(self, "Enrollment complete", f"{display_name} is now active.")

    # -- verify flow --

    def _start_verify_recording(self):
        if not self._require_wespeaker_threshold_calibrated():
            return
        if not self._require_active_threshold():
            return
        if self._ensure_profile_ready_for_verify() is None:
            return
        self._start_recording("verify")

    def _stop_verify_recording(self):
        audio_path = self._stop_recording()
        if audio_path:
            self._process_verify_audio(audio_path)

    def _import_verify_audio(self):
        if not self._require_wespeaker_threshold_calibrated():
            return
        if not self._require_active_threshold():
            return
        if self._ensure_profile_ready_for_verify() is None:
            return
        audio_path = self._choose_audio_file()
        if audio_path:
            self._process_verify_audio(audio_path)

    def _process_verify_audio(self, audio_path):
        if not self._require_wespeaker_threshold_calibrated():
            return
        if not self._require_active_threshold():
            return
        profile = self._ensure_profile_ready_for_verify()
        if profile is None:
            return

        self.verify_page.clear_result()
        self.verify_page.set_status(f"Verifying audio ({self._backend_label()})...", ok=None)
        self.statusBar().showMessage("Running verification")

        def task():
            if self._current_backend_mode() == "remote_rknn":
                normalized_audio_path = normalize_audio_file(audio_path, prefix="verify")
                verify_result = self._backend_client().verify_audio(
                    normalized_audio_path,
                    threshold=float(self.config["threshold"]),
                    prefix="verify",
                )
                result = {
                    "source_audio_path": str(audio_path),
                    "normalized_audio_path": str(normalized_audio_path),
                    "backend_mode": self._current_backend_mode(),
                    **verify_result,
                }
            else:
                artifacts = build_feature_artifacts(
                    audio_path, prefix="verify",
                    backend_mode=self._current_backend_mode(),
                )
                backend_result = self._backend_client().embed_feature(
                    artifacts["feature_path"], prefix="verify",
                )
                artifacts.update(backend_result)
                result = artifacts
            try:
                result["quality"] = analyze_audio_quality(result["normalized_audio_path"])
            except Exception:
                result["quality"] = None
            if self.config.get("passphrase_verification_enabled"):
                asr_result = self._asr_client().transcribe(result["normalized_audio_path"])
                match_result = match_passphrase(
                    asr_result.get("text", ""),
                    self.config.get("expected_passphrase", ""),
                    mode=self.config.get("passphrase_match_mode", "normalized_exact"),
                )
                result["passphrase"] = {
                    **asr_result,
                    **match_result,
                    "enabled": True,
                    "expected_text": self.config.get("expected_passphrase", ""),
                }
            return result

        self._run_task(task, self._on_verify_processed, self._on_task_failed)

    # PLACEHOLDER_ON_VERIFY_PROCESSED

    def _on_verify_processed(self, result):
        profile = self._get_active_profile(allow_model_mismatch=True)
        if (
            profile is None
            or profile.get("model_fingerprint_mismatch")
            or profile.get("embedding") is None
        ):
            self.verify_page.set_status("Verification failed", ok=False)
            QMessageBox.critical(self, "Verification failed", "The active profile is missing or out of date.")
            return

        if self._current_backend_mode() == "remote_rknn":
            score = float(result["score"])
            threshold = float(result["threshold"])
            passed = bool(result["passed"])
            decision = str(result["decision"])
        else:
            score = cosine_score(result["embedding"], profile["embedding"])
            threshold = float(self.config["threshold"])
            passed = score >= threshold
            decision = "accept" if passed else "reject"

        quality = result.get("quality")
        self._last_quality = quality
        passphrase_enabled = bool(self.config.get("passphrase_verification_enabled", False))
        passphrase_result = result.get("passphrase") if passphrase_enabled else None
        passphrase_matched = bool(passphrase_result.get("matched")) if passphrase_result else True
        final_passed = bool(passed and passphrase_matched)
        final_decision = "accept" if final_passed else "reject"

        self.verify_page.set_quality(quality)
        self.verify_page.set_result(score, threshold, final_decision)
        self.verify_page.set_passphrase_result(
            enabled=passphrase_enabled,
            expected_text=(passphrase_result or {}).get("expected_text", ""),
            recognized_text=(passphrase_result or {}).get("text", ""),
            matched=passphrase_matched if passphrase_enabled else None,
            speaker_passed=passed if passphrase_enabled else None,
            final_passed=final_passed if passphrase_enabled else None,
        )
        self.verify_page.set_status(
            "Verification passed" if final_passed else "Verification rejected",
            ok=final_passed,
        )

        notes = [
            f"backend={self._current_backend_mode()}",
            f"fingerprint={self._current_model_fingerprint()}",
        ]
        if self._current_backend_mode() == "remote_rknn":
            notes.append(f"remote_decision={decision}")
            notes.append(f"num_eval={int(result.get('num_eval', 5))}")
            notes.append(f"preprocess_for_inference={bool(result.get('preprocess_for_inference', True))}")
        if passphrase_enabled:
            notes.append(
                "passphrase="
                + json.dumps(
                    {
                        "enabled": True,
                        "expected_text": passphrase_result.get("expected_text", ""),
                        "recognized_text": passphrase_result.get("text", ""),
                        "matched": bool(passphrase_result.get("matched", False)),
                        "speaker_decision": decision,
                        "final_decision": final_decision,
                        "asr_backend": passphrase_result.get("backend", ""),
                        "asr_model_name": passphrase_result.get("model_name", ""),
                    },
                    ensure_ascii=False,
                )
            )
        self.history_store.add_entry(
            score=score,
            threshold=threshold,
            decision=final_decision,
            profile_id=profile["profile_id"],
            audio_path=result["normalized_audio_path"],
            notes=";".join(notes),
        )
        self._refresh_history()
        self._refresh_dashboard()

    # -- recording helpers --

    def _start_recording(self, mode):
        if self._task_thread is not None:
            QMessageBox.warning(self, "Busy", "A background task is still running.")
            return
        try:
            self.recorder.start_recording(prefix=mode)
        except Exception as exc:
            QMessageBox.critical(self, "Recording failed", str(exc))
            return

        self._record_mode = mode
        self._recording_seconds = 0
        controls = self._active_recording_controls()
        if controls:
            controls.set_recording(True)
            controls.reset_elapsed()
        if mode == "enroll":
            self.enroll_page.set_status("Recording...", ok=None)
        else:
            self.verify_page.set_status("Recording...", ok=None)
        self.statusBar().showMessage("Recording")
        self._record_timer.start()

    def _stop_recording(self):
        self._record_timer.stop()
        if self._record_mode is None:
            return None

        mode = self._record_mode
        self._record_mode = None
        try:
            audio_path = self.recorder.stop_recording()
        except Exception as exc:
            QMessageBox.critical(self, "Recording failed", str(exc))
            audio_path = None

        controls = self.enroll_page.recording_controls if mode == "enroll" else self.verify_page.recording_controls
        controls.set_recording(False)
        controls.set_level(0.0)
        self.statusBar().showMessage("Recording stopped", 2000)
        return audio_path

    def _on_record_tick(self):
        self._recording_seconds += 1
        controls = self._active_recording_controls()
        if controls:
            controls.set_elapsed(self._recording_seconds)

    def _active_recording_controls(self):
        if self._record_mode == "enroll":
            return self.enroll_page.recording_controls
        if self._record_mode == "verify":
            return self.verify_page.recording_controls
        return None

    def _on_level_changed(self, level):
        controls = self._active_recording_controls()
        if controls:
            controls.set_level(level)

    # -- task thread --

    def _run_task(self, target, on_success, on_failed):
        if self._task_thread is not None:
            QMessageBox.warning(self, "Busy", "A background task is still running.")
            return
        self.setCursor(Qt.WaitCursor)
        thread = TaskThread(target)
        self._task_thread = thread

        def handle_success(result):
            on_success(result)

        def handle_failed(message):
            on_failed(message)

        thread.succeeded.connect(handle_success)
        thread.failed.connect(handle_failed)
        thread.finished.connect(lambda: self._clear_task(thread))
        thread.start()

    def _clear_task(self, thread=None):
        active_thread = self._task_thread if thread is None else thread
        if active_thread is None:
            return
        if self._task_thread is active_thread:
            self.unsetCursor()
            self._task_thread = None
        active_thread.deleteLater()

    def _on_task_failed(self, message):
        self.enroll_page.set_status("Task failed", ok=False)
        self.verify_page.set_status("Task failed", ok=False)
        self.statusBar().showMessage("Task failed", 5000)
        QMessageBox.critical(self, "Task failed", message)

    # PLACEHOLDER_REFRESH_AND_HELPERS

    # -- refresh helpers --

    def _refresh_profile(self, profile=None):
        if profile is None:
            profile = self._get_active_profile()
        self.profile_page.set_profile(profile)
        if profile is None:
            self.verify_page.set_profile_name("No active profile")
            self.verify_page.recording_controls.set_controls_enabled(False)
        else:
            self.verify_page.set_profile_name(profile.get("display_name", "Authorized user"))
            self.verify_page.recording_controls.set_controls_enabled(True)
        self._sync_backend_status()

    def _refresh_history(self):
        entries = self.history_store.list_entries()
        self.history_page.set_entries(entries)

    def _refresh_dashboard(self):
        profile = self._get_active_profile()
        entries = self.history_store.list_entries()
        last_entry = entries[0] if entries else None
        self.dashboard_page.refresh(self.config, profile, last_entry, self._last_quality)

    # -- history actions --

    def _export_history(self):
        output_path, _ = QFileDialog.getSaveFileName(
            self, "Export verification history",
            str(Path.cwd() / "speaker_history.csv"), "CSV Files (*.csv)",
        )
        if not output_path:
            return
        self.history_store.export_csv(output_path)
        QMessageBox.information(self, "Export complete", f"History exported to:\n{output_path}")

    def _delete_selected_history(self):
        entry_ids = self.history_page.selected_entry_ids()
        if not entry_ids:
            QMessageBox.information(self, "Nothing selected", "Select one or more history rows first.")
            return
        self.history_store.delete_entries(entry_ids)
        self._refresh_history()

    def _clear_history(self):
        reply = QMessageBox.question(self, "Clear history", "Delete all verification history?")
        if reply != QMessageBox.Yes:
            return
        self.history_store.clear()
        self._refresh_history()

    # -- profile actions --

    def _delete_active_profile(self):
        profile = self._get_active_profile(allow_model_mismatch=True)
        if profile is None:
            QMessageBox.information(self, "No active profile", "There is no active profile to delete.")
            return
        reply = QMessageBox.question(self, "Delete active profile", "Delete the current active profile?")
        if reply != QMessageBox.Yes:
            return

        if self._current_backend_mode() == "remote_rknn":
            self.enroll_page.set_status("Deleting remote active profile...", ok=None)

            def task():
                return self._delete_remote_active_profile(profile)

            self._run_task(task, self._on_remote_profile_deleted, self._on_task_failed)
            return

        self.profile_store.delete_active_profile(backend_mode=self._current_backend_mode())
        self._refresh_profile()
        self.verify_page.clear_result()
        self.enroll_page.set_status("Active profile deleted", ok=True)
        self._refresh_dashboard()

    def _go_to_enroll_page(self):
        self._reset_enroll_session()
        self.sidebar.select(_IDX_ENROLL)
        self.stack.setCurrentIndex(_IDX_ENROLL)

    def _reset_enroll_session(self, reset_status=True):
        self._pending_enroll_embeddings = []
        self.enroll_page.clear_samples()
        self.enroll_page.set_progress(0, int(self.config["enroll_samples"]))
        if reset_status:
            self.enroll_page.set_status("Ready to enroll", ok=None)

    def _choose_audio_file(self):
        audio_path, _ = QFileDialog.getOpenFileName(
            self, "Choose audio file", "",
            "Audio Files (*.wav *.flac *.mp3 *.m4a *.ogg *.opus)",
        )
        return audio_path or None

    # -- backend artifact building --

    def _build_backend_artifacts(self, audio_path, prefix):
        if self._current_backend_mode() == "remote_rknn":
            normalized_audio_path = normalize_audio_file(audio_path, prefix=prefix)
            backend_result = self._backend_client().embed_audio(normalized_audio_path, prefix=prefix)
            return {
                "source_audio_path": str(audio_path),
                "normalized_audio_path": str(normalized_audio_path),
                "backend_mode": self._current_backend_mode(),
                **backend_result,
            }
        artifacts = build_feature_artifacts(
            audio_path, prefix=prefix, backend_mode=self._current_backend_mode(),
        )
        backend_result = self._backend_client().embed_feature(artifacts["feature_path"], prefix=prefix)
        artifacts.update(backend_result)
        return artifacts

    def _write_temp_embedding(self, embedding, prefix):
        output_path = Path(local_tmp_dir()) / f"{prefix}_{self._current_backend_mode()}_active.npy"
        np.save(output_path, np.asarray(embedding, dtype=np.float32))
        return output_path

    # -- remote profile sync --

    def _save_remote_active_profile(self, display_name, embeddings, num_samples):
        merged = build_merged_embedding(embeddings)
        temp_embedding_path = self._write_temp_embedding(merged, "active_profile")
        previous_profile = self._get_active_profile(allow_model_mismatch=True)
        rollback_path = None
        if previous_profile is not None and previous_profile.get("embedding") is not None:
            rollback_path = self._write_temp_embedding(previous_profile["embedding"], "rollback_profile")
        client = self._backend_client()
        try:
            client.push_active_profile(temp_embedding_path)
            try:
                profile = self.profile_store.save_active_profile(
                    display_name=display_name, embeddings=merged,
                    num_samples=num_samples,
                    backend_mode=self._current_backend_mode(),
                    model_fingerprint=self._current_model_fingerprint(),
                )
            except Exception as exc:
                try:
                    if rollback_path is not None:
                        client.push_active_profile(rollback_path)
                    else:
                        client.clear_active_profile()
                except Exception as rollback_exc:
                    raise RuntimeError(f"{exc}\n\nRemote rollback failed:\n{rollback_exc}") from exc
                raise
            return profile
        finally:
            temp_embedding_path.unlink(missing_ok=True)
            if rollback_path is not None:
                rollback_path.unlink(missing_ok=True)

    def _on_remote_enroll_saved(self, profile):
        self._reset_enroll_session(reset_status=False)
        self._refresh_profile(profile)
        self.enroll_page.set_status("Enrollment completed and remote profile published", ok=True)
        self.verify_page.set_status("Ready for verification", ok=None)
        self._refresh_dashboard()
        QMessageBox.information(
            self, "Enrollment complete",
            f"{profile.get('display_name', 'Authorized user')} is now active.",
        )

    def _on_remote_enroll_failed(self, message):
        self.enroll_page.set_status("Remote enrollment publish failed", ok=False)
        self.verify_page.set_status("Remote enrollment publish failed", ok=False)
        self.statusBar().showMessage("Enrollment publish failed", 5000)
        QMessageBox.critical(self, "Enrollment publish failed", message)

    def _delete_remote_active_profile(self, profile):
        client = self._backend_client()
        rollback_path = self._write_temp_embedding(profile["embedding"], "rollback_profile")
        try:
            client.clear_active_profile()
            try:
                deleted = self.profile_store.delete_active_profile(backend_mode=self._current_backend_mode())
            except Exception as exc:
                try:
                    client.push_active_profile(rollback_path)
                except Exception as rollback_exc:
                    raise RuntimeError(f"{exc}\n\nRemote rollback failed:\n{rollback_exc}") from exc
                raise
            if not deleted:
                try:
                    client.push_active_profile(rollback_path)
                except Exception as rollback_exc:
                    raise RuntimeError(f"Local active profile delete failed.\n\nRemote rollback failed:\n{rollback_exc}")
                raise RuntimeError("Local active profile delete failed.")
            return True
        finally:
            rollback_path.unlink(missing_ok=True)

    def _on_remote_profile_deleted(self, _result):
        self._refresh_profile()
        self.verify_page.clear_result()
        self.enroll_page.set_status("Active profile deleted locally and remotely", ok=True)
        self._refresh_dashboard()

    def closeEvent(self, event):
        if self._task_thread is not None and self._task_thread.isRunning():
            self._task_thread.wait(5000)
            self._clear_task()
        self._invalidate_remote_board_client()
        super().closeEvent(event)
