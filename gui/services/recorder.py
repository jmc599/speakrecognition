from pathlib import Path
from uuid import uuid4

import numpy as np
import soundfile as sf
from PyQt5.QtCore import QObject, pyqtSignal

from gui.services.paths import local_tmp_dir


def _get_sounddevice():
    try:
        import sounddevice as sd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "sounddevice 未安装。请先执行: pip install -r requirements_gui.txt"
        ) from exc
    return sd


class Recorder(QObject):
    level_changed = pyqtSignal(float)

    def __init__(self, device_name="", device_index=None):
        super().__init__()
        self.device_name = device_name or None
        self.device_index = device_index if device_index not in ("", None) else None
        self._stream = None
        self._chunks = []
        self._samplerate = 16000
        self._channels = 1
        self._recording_path = None

    @staticmethod
    def list_input_devices():
        sd = _get_sounddevice()
        hostapis = sd.query_hostapis()
        devices = []
        for index, device in enumerate(sd.query_devices()):
            if device.get("max_input_channels", 0) > 0:
                hostapi_index = device.get("hostapi", -1)
                hostapi_name = (
                    hostapis[hostapi_index].get("name", "Unknown API")
                    if 0 <= hostapi_index < len(hostapis)
                    else "Unknown API"
                )
                name = device.get("name", "Unknown input")
                label = f"[{index}] {name}, {hostapi_name}"
                devices.append(
                    {
                        "index": index,
                        "name": name,
                        "hostapi": hostapi_name,
                        "label": label,
                    }
                )
        return devices

    def set_device(self, device_name="", device_index=None):
        self.device_name = device_name or None
        self.device_index = device_index if device_index not in ("", None) else None

    def set_device_name(self, device_name):
        self.set_device(device_name=device_name, device_index=self.device_index)

    def start_recording(self, prefix="recording"):
        if self._stream is not None:
            raise RuntimeError("Recorder is already running.")
        sd = _get_sounddevice()

        self._chunks = []
        filename = f"{prefix}_{uuid4().hex}.wav"
        self._recording_path = Path(local_tmp_dir()) / filename

        def callback(indata, frames, time_info, status):  # pragma: no cover
            if status:
                raise RuntimeError(str(status))
            chunk = np.copy(indata[:, : self._channels])
            self._chunks.append(chunk)
            rms = float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0
            self.level_changed.emit(min(rms * 10.0, 1.0))

        device = self.device_index if self.device_index is not None else self.device_name
        self._stream = sd.InputStream(
            samplerate=self._samplerate,
            channels=self._channels,
            dtype="float32",
            device=device,
            callback=callback,
        )
        self._stream.start()
        return str(self._recording_path)

    def stop_recording(self):
        if self._stream is None:
            raise RuntimeError("Recorder is not running.")

        stream = self._stream
        self._stream = None
        try:
            stream.stop()
            stream.close()
        finally:
            self.level_changed.emit(0.0)

        if not self._chunks:
            raise RuntimeError("Recorded empty audio. Please try again.")

        audio = np.concatenate(self._chunks, axis=0).astype(np.float32, copy=False)
        sf.write(str(self._recording_path), audio, self._samplerate)
        return str(self._recording_path)
