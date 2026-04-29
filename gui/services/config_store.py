import json
from copy import deepcopy

from gui.services.paths import config_path
from gui.services.passphrase_match import MATCH_MODE_CHOICES, MATCH_MODE_EXACT


DEFAULT_THRESHOLD_POLICY = "eer"
THRESHOLD_POLICY_CHOICES = ("eer", "far_1", "far_0p1")
THRESHOLD_SOURCE_UNSET = "unset"
THRESHOLD_SOURCE_MANUAL = "manual"
THRESHOLD_SOURCE_CALIBRATION = "calibration"

BACKEND_THRESHOLD_KEYS = {
    "local_onnx": "local_onnx_threshold",
    "local_pytorch": "local_pytorch_threshold",
    "remote_rknn": "remote_rknn_threshold",
    "wespeaker_onnx": "wespeaker_onnx_threshold",
}
BACKEND_THRESHOLD_POLICY_KEYS = {
    "local_onnx": "local_onnx_threshold_policy",
    "local_pytorch": "local_pytorch_threshold_policy",
    "remote_rknn": "remote_rknn_threshold_policy",
    "wespeaker_onnx": "wespeaker_onnx_threshold_policy",
}
BACKEND_THRESHOLD_SOURCE_KEYS = {
    "local_onnx": "local_onnx_threshold_source",
    "local_pytorch": "local_pytorch_threshold_source",
    "remote_rknn": "remote_rknn_threshold_source",
    "wespeaker_onnx": "wespeaker_onnx_threshold_source",
}
BACKEND_THRESHOLD_CALIBRATED_KEYS = {
    "local_onnx": "local_onnx_threshold_calibrated",
    "local_pytorch": "local_pytorch_threshold_calibrated",
    "remote_rknn": "remote_rknn_threshold_calibrated",
    "wespeaker_onnx": "wespeaker_threshold_calibrated",
}
LEGACY_DEFAULT_THRESHOLDS = {
    "local_onnx": 0.213456,
    "local_pytorch": 0.213456,
    "remote_rknn": 0.187723,
    "wespeaker_onnx": 0.250000,
}
_KEEP = object()


DEFAULT_CONFIG = {
    "backend_mode": "local_onnx",
    "local_onnx_model_path": "checkpoints/resnet_v7_vox2ft_s1_latest_fixed.onnx",
    "local_pytorch_checkpoint_path": "checkpoints/resnet_v7_vox2ft_s1_latest.pth",
    "wespeaker_onnx_model_path": "checkpoints/wespeaker/cnceleb_resnet34_LM.onnx",
    "board_host": "192.168.50.2",
    "board_port": 22,
    "board_username": "root",
    "board_password": "",
    "remote_transport_preference": "http_first",
    "remote_http_base_url": "",
    "remote_model_path": "/root/models/resnet_v7_vox2ft_s1_latest_fixed_rt160_nonorm_v232.rknn",
    "remote_runner_script": "/root/models/rknn_remote_runner.py",
    "remote_embed_script": "/root/models/rknn_remote_runner.py",
    "remote_workdir": "/root/models/gui_jobs",
    "remote_python": "python3",
    "threshold": None,
    "threshold_policy": DEFAULT_THRESHOLD_POLICY,
    "threshold_source": THRESHOLD_SOURCE_UNSET,
    "threshold_calibrated": False,
    "local_onnx_threshold": None,
    "local_pytorch_threshold": None,
    "remote_rknn_threshold": 0.342024,
    "wespeaker_onnx_threshold": None,
    "local_onnx_threshold_policy": DEFAULT_THRESHOLD_POLICY,
    "local_pytorch_threshold_policy": DEFAULT_THRESHOLD_POLICY,
    "remote_rknn_threshold_policy": "far_1",
    "wespeaker_onnx_threshold_policy": DEFAULT_THRESHOLD_POLICY,
    "local_onnx_threshold_source": THRESHOLD_SOURCE_UNSET,
    "local_pytorch_threshold_source": THRESHOLD_SOURCE_UNSET,
    "remote_rknn_threshold_source": THRESHOLD_SOURCE_MANUAL,
    "wespeaker_onnx_threshold_source": THRESHOLD_SOURCE_UNSET,
    "local_onnx_threshold_calibrated": False,
    "local_pytorch_threshold_calibrated": False,
    "remote_rknn_threshold_calibrated": False,
    "wespeaker_threshold_calibrated": False,
    "enroll_samples": 3,
    "device_name": "",
    "device_index": None,
    "passphrase_verification_enabled": False,
    "expected_passphrase": "",
    "asr_backend": "sensevoice_onnx",
    "asr_model_name": "iic/SenseVoiceSmall",
    "asr_language": "zh",
    "asr_cache_dir": "F:/speakerreg_artifacts/funasr_cache",
    "passphrase_match_mode": "normalized_exact",
}

REMOTE_TRANSPORT_CHOICES = ("http_first", "ssh_only")


def backend_threshold_key(backend_mode):
    return BACKEND_THRESHOLD_KEYS[str(backend_mode)]


def backend_threshold_policy_key(backend_mode):
    return BACKEND_THRESHOLD_POLICY_KEYS[str(backend_mode)]


def backend_threshold_source_key(backend_mode):
    return BACKEND_THRESHOLD_SOURCE_KEYS[str(backend_mode)]


def backend_threshold_calibrated_key(backend_mode):
    return BACKEND_THRESHOLD_CALIBRATED_KEYS[str(backend_mode)]


def set_backend_threshold_metadata(
    config,
    backend_mode,
    *,
    threshold=_KEEP,
    threshold_policy=_KEEP,
    source=_KEEP,
    calibrated=_KEEP,
):
    threshold_key = backend_threshold_key(backend_mode)
    policy_key = backend_threshold_policy_key(backend_mode)
    source_key = backend_threshold_source_key(backend_mode)
    calibrated_key = backend_threshold_calibrated_key(backend_mode)

    if threshold is not _KEEP:
        config[threshold_key] = threshold
    if threshold_policy is not _KEEP:
        config[policy_key] = str(threshold_policy)
    if source is not _KEEP:
        config[source_key] = str(source)
    if calibrated is not _KEEP:
        config[calibrated_key] = bool(calibrated)
    return config


def clear_backend_threshold(config, backend_mode):
    return set_backend_threshold_metadata(
        config,
        backend_mode,
        threshold=None,
        threshold_policy=DEFAULT_THRESHOLD_POLICY,
        source=THRESHOLD_SOURCE_UNSET,
        calibrated=False,
    )


def _coerce_optional_float(value):
    if value in (None, ""):
        return None
    return float(value)


def _normalize_threshold_fields(config):
    for backend_mode in BACKEND_THRESHOLD_KEYS:
        threshold_key = backend_threshold_key(backend_mode)
        policy_key = backend_threshold_policy_key(backend_mode)
        source_key = backend_threshold_source_key(backend_mode)
        calibrated_key = backend_threshold_calibrated_key(backend_mode)

        config[threshold_key] = _coerce_optional_float(
            config.get(threshold_key, DEFAULT_CONFIG[threshold_key])
        )
        policy = str(config.get(policy_key, DEFAULT_CONFIG[policy_key]) or DEFAULT_THRESHOLD_POLICY)
        if policy not in THRESHOLD_POLICY_CHOICES:
            policy = DEFAULT_THRESHOLD_POLICY
        config[policy_key] = policy
        config[source_key] = str(
            config.get(source_key, DEFAULT_CONFIG[source_key]) or THRESHOLD_SOURCE_UNSET
        )
        config[calibrated_key] = bool(
            config.get(calibrated_key, DEFAULT_CONFIG[calibrated_key])
        )


def _drop_legacy_default_thresholds(config):
    for backend_mode, legacy_value in LEGACY_DEFAULT_THRESHOLDS.items():
        threshold_key = backend_threshold_key(backend_mode)
        source_key = backend_threshold_source_key(backend_mode)
        calibrated_key = backend_threshold_calibrated_key(backend_mode)
        threshold_value = config.get(threshold_key)
        if threshold_value is None:
            continue
        if bool(config.get(calibrated_key, False)):
            continue
        source = str(config.get(source_key, THRESHOLD_SOURCE_UNSET) or THRESHOLD_SOURCE_UNSET)
        if source not in ("", THRESHOLD_SOURCE_UNSET, "legacy_default"):
            continue
        if abs(float(threshold_value) - float(legacy_value)) <= 1e-9:
            config[threshold_key] = None
            config[source_key] = THRESHOLD_SOURCE_UNSET


def _resolve_backend_threshold(config, backend_mode):
    return _coerce_optional_float(config.get(backend_threshold_key(backend_mode)))


class ConfigStore:
    def __init__(self):
        self.path = config_path()

    def load(self):
        config = deepcopy(DEFAULT_CONFIG)
        if self.path.is_file():
            with self.path.open("r", encoding="utf-8") as f:
                stored = json.load(f)
            if isinstance(stored, dict):
                config.update(stored)

        _normalize_threshold_fields(config)
        _drop_legacy_default_thresholds(config)

        config["board_port"] = int(config.get("board_port", DEFAULT_CONFIG["board_port"]))
        config["remote_transport_preference"] = str(
            config.get(
                "remote_transport_preference",
                DEFAULT_CONFIG["remote_transport_preference"],
            )
            or DEFAULT_CONFIG["remote_transport_preference"]
        )
        if config["remote_transport_preference"] not in REMOTE_TRANSPORT_CHOICES:
            config["remote_transport_preference"] = DEFAULT_CONFIG["remote_transport_preference"]
        config["remote_http_base_url"] = str(
            config.get("remote_http_base_url", DEFAULT_CONFIG["remote_http_base_url"]) or ""
        ).strip()
        config["passphrase_verification_enabled"] = bool(
            config.get(
                "passphrase_verification_enabled",
                DEFAULT_CONFIG["passphrase_verification_enabled"],
            )
        )
        config["expected_passphrase"] = str(
            config.get("expected_passphrase", DEFAULT_CONFIG["expected_passphrase"]) or ""
        ).strip()
        config["asr_backend"] = str(
            config.get("asr_backend", DEFAULT_CONFIG["asr_backend"]) or DEFAULT_CONFIG["asr_backend"]
        ).strip()
        config["asr_model_name"] = str(
            config.get("asr_model_name", DEFAULT_CONFIG["asr_model_name"])
            or DEFAULT_CONFIG["asr_model_name"]
        ).strip()
        config["asr_language"] = str(
            config.get("asr_language", DEFAULT_CONFIG["asr_language"])
            or DEFAULT_CONFIG["asr_language"]
        ).strip()
        config["asr_cache_dir"] = str(
            config.get("asr_cache_dir", DEFAULT_CONFIG["asr_cache_dir"])
            or DEFAULT_CONFIG["asr_cache_dir"]
        ).strip()
        config["passphrase_match_mode"] = str(
            config.get(
                "passphrase_match_mode",
                DEFAULT_CONFIG["passphrase_match_mode"],
            )
            or DEFAULT_CONFIG["passphrase_match_mode"]
        ).strip()
        if config["passphrase_match_mode"] not in MATCH_MODE_CHOICES:
            config["passphrase_match_mode"] = MATCH_MODE_EXACT
        if not config.get("remote_runner_script"):
            config["remote_runner_script"] = str(
                config.get("remote_embed_script", DEFAULT_CONFIG["remote_runner_script"])
            )
        config["threshold"] = self._resolve_threshold(config)
        config["threshold_policy"] = self._resolve_threshold_policy(config)
        config["threshold_source"] = self._resolve_threshold_source(config)
        config["threshold_calibrated"] = self._resolve_threshold_calibrated(config)
        config["enroll_samples"] = max(
            1, int(config.get("enroll_samples", DEFAULT_CONFIG["enroll_samples"]))
        )
        return config

    def save(self, config):
        payload = deepcopy(DEFAULT_CONFIG)
        payload.update(config)
        _normalize_threshold_fields(payload)
        _drop_legacy_default_thresholds(payload)
        payload["board_port"] = int(
            payload.get("board_port", DEFAULT_CONFIG["board_port"])
        )
        payload["remote_transport_preference"] = str(
            payload.get(
                "remote_transport_preference",
                DEFAULT_CONFIG["remote_transport_preference"],
            )
            or DEFAULT_CONFIG["remote_transport_preference"]
        )
        if payload["remote_transport_preference"] not in REMOTE_TRANSPORT_CHOICES:
            payload["remote_transport_preference"] = DEFAULT_CONFIG["remote_transport_preference"]
        payload["remote_http_base_url"] = str(
            payload.get("remote_http_base_url", DEFAULT_CONFIG["remote_http_base_url"]) or ""
        ).strip()
        payload["passphrase_verification_enabled"] = bool(
            payload.get(
                "passphrase_verification_enabled",
                DEFAULT_CONFIG["passphrase_verification_enabled"],
            )
        )
        payload["expected_passphrase"] = str(
            payload.get("expected_passphrase", DEFAULT_CONFIG["expected_passphrase"]) or ""
        ).strip()
        payload["asr_backend"] = str(
            payload.get("asr_backend", DEFAULT_CONFIG["asr_backend"])
            or DEFAULT_CONFIG["asr_backend"]
        ).strip()
        payload["asr_model_name"] = str(
            payload.get("asr_model_name", DEFAULT_CONFIG["asr_model_name"])
            or DEFAULT_CONFIG["asr_model_name"]
        ).strip()
        payload["asr_language"] = str(
            payload.get("asr_language", DEFAULT_CONFIG["asr_language"])
            or DEFAULT_CONFIG["asr_language"]
        ).strip()
        payload["asr_cache_dir"] = str(
            payload.get("asr_cache_dir", DEFAULT_CONFIG["asr_cache_dir"])
            or DEFAULT_CONFIG["asr_cache_dir"]
        ).strip()
        payload["passphrase_match_mode"] = str(
            payload.get("passphrase_match_mode", DEFAULT_CONFIG["passphrase_match_mode"])
            or DEFAULT_CONFIG["passphrase_match_mode"]
        ).strip()
        if payload["passphrase_match_mode"] not in MATCH_MODE_CHOICES:
            payload["passphrase_match_mode"] = MATCH_MODE_EXACT
        if not payload.get("remote_runner_script"):
            payload["remote_runner_script"] = str(
                payload.get("remote_embed_script", DEFAULT_CONFIG["remote_runner_script"])
            )
        payload["remote_embed_script"] = str(
            payload.get("remote_runner_script", payload.get("remote_embed_script", ""))
        )
        payload["threshold"] = self._resolve_threshold(payload)
        payload["threshold_policy"] = self._resolve_threshold_policy(payload)
        payload["threshold_source"] = self._resolve_threshold_source(payload)
        payload["threshold_calibrated"] = self._resolve_threshold_calibrated(payload)
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return payload

    @staticmethod
    def _resolve_threshold(config):
        backend_mode = str(config.get("backend_mode", DEFAULT_CONFIG["backend_mode"]))
        return _resolve_backend_threshold(config, backend_mode)

    @staticmethod
    def _resolve_threshold_policy(config):
        backend_mode = str(config.get("backend_mode", DEFAULT_CONFIG["backend_mode"]))
        return str(
            config.get(
                backend_threshold_policy_key(backend_mode),
                DEFAULT_CONFIG[backend_threshold_policy_key(backend_mode)],
            )
        )

    @staticmethod
    def _resolve_threshold_source(config):
        backend_mode = str(config.get("backend_mode", DEFAULT_CONFIG["backend_mode"]))
        return str(
            config.get(
                backend_threshold_source_key(backend_mode),
                DEFAULT_CONFIG[backend_threshold_source_key(backend_mode)],
            )
        )

    @staticmethod
    def _resolve_threshold_calibrated(config):
        backend_mode = str(config.get("backend_mode", DEFAULT_CONFIG["backend_mode"]))
        return bool(
            config.get(
                backend_threshold_calibrated_key(backend_mode),
                DEFAULT_CONFIG[backend_threshold_calibrated_key(backend_mode)],
            )
        )
