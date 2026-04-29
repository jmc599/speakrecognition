from pathlib import Path


class AsrService:
    def __init__(self, config):
        self.config = config
        self._model = None
        self._model_spec = None
        self._postprocess = None

    def _ensure_model(self):
        model_name = str(
            self.config.get("asr_model_name", "iic/SenseVoiceSmall") or "iic/SenseVoiceSmall"
        ).strip()
        if self._model is not None and self._model_spec == model_name:
            return

        try:
            from funasr_onnx import SenseVoiceSmall
            from funasr_onnx.utils.postprocess_utils import rich_transcription_postprocess
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "ASR dependencies are missing. Install funasr and funasr-onnx in the current environment."
            ) from exc

        cache_dir = Path(
            str(
                self.config.get("asr_cache_dir", "F:/speakerreg_artifacts/funasr_cache")
                or "F:/speakerreg_artifacts/funasr_cache"
            ).strip()
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_dir = self._resolve_model_dir(model_name, cache_dir)

        # Prefer the simplest official ONNX runtime path for desktop success:
        # quantized CPU inference with SenseVoiceSmall.
        self._model = SenseVoiceSmall(
            str(model_dir),
            batch_size=1,
            quantize=True,
            cache_dir=str(cache_dir),
            disable_update=True,
        )
        self._postprocess = rich_transcription_postprocess
        self._model_spec = model_name

    @staticmethod
    def _resolve_model_dir(model_name, cache_dir):
        model_path = Path(model_name)
        if model_path.exists():
            return model_path

        normalized = model_name.replace("\\", "/").strip("/")
        cached_path = cache_dir / normalized
        if cached_path.exists():
            return cached_path
        return model_name

    def transcribe(self, audio_path):
        self._ensure_model()

        language = str(self.config.get("asr_language", "zh") or "zh").strip()
        result = self._model([str(audio_path)], language=language, use_itn=True)
        raw_item = result[0] if result else ""

        if isinstance(raw_item, dict):
            raw_text = str(raw_item.get("text", ""))
        else:
            raw_text = str(raw_item)

        processed_text = self._postprocess(raw_text) if self._postprocess else raw_text
        return {
            "backend": str(self.config.get("asr_backend", "sensevoice_onnx") or "sensevoice_onnx"),
            "model_name": self._model_spec,
            "language": language,
            "audio_path": str(Path(audio_path)),
            "raw_text": raw_text,
            "text": str(processed_text).strip(),
        }
