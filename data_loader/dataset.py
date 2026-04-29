import os
import random

import torch
import torchaudio
from torch.utils.data import Dataset

from utils.audio_io import load_audio_mono_16k
from utils.inference_audio import detect_active_spans
from utils.legacy_feature import DEFAULT_MAX_FRAMES, LEGACY_NUM_MELS


AUDIO_EXTENSIONS = {".wav", ".flac", ".m4a", ".mp3", ".ogg"}


def _parse_snr_range(raw_value, default_min, default_max):
    if isinstance(raw_value, (tuple, list)) and len(raw_value) == 2:
        lo, hi = float(raw_value[0]), float(raw_value[1])
    else:
        tokens = [part.strip() for part in str(raw_value).split(",") if part.strip()]
        if len(tokens) != 2:
            lo, hi = float(default_min), float(default_max)
        else:
            lo, hi = float(tokens[0]), float(tokens[1])
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


class SpeakerDataset(Dataset):
    def __init__(
        self,
        data_list_path,
        base_path,
        max_frames=DEFAULT_MAX_FRAMES,
        train=True,
        augment=False,
        noise_std=0.003,
        gain_db=6.0,
        freq_mask_param=6,
        time_mask_param=10,
        speed_perturb_factors=None,
        preprocess_for_inference=False,
        train_crop_mode="random",
        musan_root="",
        musan_prob=0.0,
        musan_snr_noise="0,15",
        musan_snr_speech="13,20",
        musan_snr_music="5,15",
        rir_root="",
        reverb_prob=0.0,
    ):
        """
        data_list_path: each line is '<relative_path> <speaker_id>'
        base_path: dataset root directory
        preprocess_for_inference: apply trim/normalize to align training with inference
        """
        self.base_path = base_path
        self.max_frames = max_frames
        self.train = train
        self.augment = augment
        self.noise_std = noise_std
        self.gain_db = gain_db
        self.speed_perturb_factors = []
        if speed_perturb_factors:
            self.speed_perturb_factors = [
                float(factor) for factor in speed_perturb_factors if float(factor) > 0
            ]
        self.preprocess_for_inference = preprocess_for_inference
        self.train_crop_mode = str(train_crop_mode).strip().lower() or "random"
        if self.train_crop_mode not in {"random", "active"}:
            raise ValueError(
                f"Unsupported train_crop_mode: {train_crop_mode}. "
                "Expected one of: random, active."
            )
        self.max_load_retries = 5
        self.musan_root = str(musan_root).strip()
        self.musan_prob = max(0.0, min(float(musan_prob), 1.0))
        self.rir_root = str(rir_root).strip()
        self.reverb_prob = max(0.0, min(float(reverb_prob), 1.0))
        self.musan_snr_ranges = {
            "noise": _parse_snr_range(musan_snr_noise, 0.0, 15.0),
            "speech": _parse_snr_range(musan_snr_speech, 13.0, 20.0),
            "music": _parse_snr_range(musan_snr_music, 5.0, 15.0),
        }

        raw_data = []
        with open(data_list_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 2:
                    raise ValueError(f"Bad list line: {line}")
                path, spk_id = parts
                raw_data.append((path, int(spk_id)))

        # Subset lists may have sparse speaker IDs. Remap them to [0, n_class).
        unique_spk_ids = sorted({spk_id for _, spk_id in raw_data})
        self.spk_id_map = {spk_id: idx for idx, spk_id in enumerate(unique_spk_ids)}
        self.data = [(path, self.spk_id_map[spk_id]) for path, spk_id in raw_data]

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=512,
            win_length=400,
            hop_length=160,
            n_mels=LEGACY_NUM_MELS,
        )
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param)
        self.musan_files = self._collect_musan_files(self.musan_root)
        self.rir_files = self._collect_audio_files(self.rir_root, label="RIR")

    @staticmethod
    def _collect_audio_files(root_path, label="audio"):
        root = str(root_path).strip()
        if not root:
            return []
        if not os.path.isdir(root):
            raise FileNotFoundError(f"{label} root not found: {root}")
        files = []
        for current_root, _, names in os.walk(root):
            for name in names:
                if os.path.splitext(name)[1].lower() in AUDIO_EXTENSIONS:
                    files.append(os.path.join(current_root, name))
        if not files:
            raise RuntimeError(f"No {label} files found under: {root}")
        return sorted(files)

    def _collect_musan_files(self, root_path):
        root = str(root_path).strip()
        if not root:
            return {}
        files = self._collect_audio_files(root, label="MUSAN")
        categorized = {"speech": [], "music": [], "noise": []}
        for path in files:
            lower_parts = {
                part.lower() for part in os.path.normpath(path).split(os.sep) if part
            }
            if "speech" in lower_parts:
                categorized["speech"].append(path)
            elif "music" in lower_parts:
                categorized["music"].append(path)
            elif "noise" in lower_parts:
                categorized["noise"].append(path)
        usable = {key: value for key, value in categorized.items() if value}
        if not usable:
            raise RuntimeError(
                "MUSAN root found, but no speech/music/noise category files were detected."
            )
        return usable

    @staticmethod
    def _rms(waveform):
        return float(torch.sqrt(torch.mean(waveform * waveform) + 1e-12).item())

    @staticmethod
    def _match_length(waveform, target_length):
        current_length = int(waveform.shape[-1])
        if current_length == target_length:
            return waveform
        if current_length > target_length:
            start = random.randint(0, current_length - target_length)
            return waveform[:, start : start + target_length]
        repeat_count = max(1, (target_length + current_length - 1) // max(current_length, 1))
        repeated = waveform.repeat(1, repeat_count)
        return repeated[:, :target_length]

    def _load_aux_waveform(self, path):
        waveform, _ = load_audio_mono_16k(path, for_inference=False)
        return waveform

    def _apply_reverb(self, waveform):
        if not self.rir_files or random.random() >= self.reverb_prob:
            return waveform
        rir_path = random.choice(self.rir_files)
        rir_waveform = self._load_aux_waveform(rir_path)
        rir_kernel = rir_waveform.reshape(1, 1, -1).float()
        rir_kernel = rir_kernel / (torch.norm(rir_kernel, p=2) + 1e-6)
        reverbed = torch.nn.functional.conv1d(
            waveform.reshape(1, 1, -1),
            torch.flip(rir_kernel, dims=[-1]),
            padding=rir_kernel.shape[-1] - 1,
        )
        reverbed = reverbed.reshape(1, -1)[:, : waveform.shape[-1]]
        return reverbed

    def _mix_musan_category(self, waveform, category):
        category_files = self.musan_files.get(category, [])
        if not category_files:
            return waveform
        mix_count = 1
        if category == "speech":
            mix_count = random.randint(3, 7)
        noise_mix = torch.zeros_like(waveform)
        for _ in range(mix_count):
            noise_path = random.choice(category_files)
            noise_waveform = self._load_aux_waveform(noise_path)
            noise_waveform = self._match_length(noise_waveform, waveform.shape[-1])
            noise_mix = noise_mix + noise_waveform
        clean_rms = self._rms(waveform)
        noise_rms = self._rms(noise_mix)
        if clean_rms <= 1e-6 or noise_rms <= 1e-6:
            return waveform
        snr_min, snr_max = self.musan_snr_ranges[category]
        snr_db = random.uniform(snr_min, snr_max)
        scale = clean_rms / ((10.0 ** (snr_db / 20.0)) * noise_rms + 1e-12)
        return waveform + (noise_mix * scale)

    def _apply_musan(self, waveform):
        if not self.musan_files or random.random() >= self.musan_prob:
            return waveform
        available_categories = list(self.musan_files.keys())
        category = random.choice(available_categories)
        return self._mix_musan_category(waveform, category)

    def _crop_training_mel(self, mel_spec):
        _, _, time = mel_spec.shape
        if time < self.max_frames:
            pad_len = self.max_frames - time
            return torch.nn.functional.pad(mel_spec, (0, pad_len))
        start = random.randint(0, time - self.max_frames)
        return mel_spec[:, :, start : start + self.max_frames]

    def _crop_training_active(self, waveform, mel_spec):
        _, _, time = mel_spec.shape
        if time < self.max_frames:
            pad_len = self.max_frames - time
            return torch.nn.functional.pad(mel_spec, (0, pad_len))

        spans = detect_active_spans(
            waveform.squeeze(0).cpu().numpy(),
            sample_rate=16000,
            frame_ms=25,
            hop_ms=10,
            top_db=35.0,
            min_active_ms=200,
        )
        if not spans:
            return self._crop_training_mel(mel_spec)

        hop_length = 160
        max_len_samples = self.max_frames * hop_length
        eligible = [span for span in spans if (span[1] - span[0]) >= max_len_samples]
        if eligible:
            span_start, span_end = random.choice(eligible)
            span_frames = max(1, (span_end - span_start) // hop_length)
            frame_start_min = max(0, span_start // hop_length)
            max_offset = max(0, span_frames - self.max_frames)
            frame_start = frame_start_min + (
                random.randint(0, max_offset) if max_offset > 0 else 0
            )
            frame_start = min(frame_start, time - self.max_frames)
            return mel_spec[:, :, frame_start : frame_start + self.max_frames]

        span_start, span_end = max(spans, key=lambda span: span[1] - span[0])
        frame_start = max(0, span_start // hop_length)
        frame_end = min(time, max(frame_start + 1, span_end // hop_length))
        cropped = mel_spec[:, :, frame_start:frame_end]
        cropped_time = cropped.shape[-1]
        if cropped_time >= self.max_frames:
            return cropped[:, :, : self.max_frames]
        pad_len = self.max_frames - cropped_time
        return torch.nn.functional.pad(cropped, (0, pad_len))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        waveform = None
        label = None
        last_exc = None
        for _ in range(self.max_load_retries):
            rel_path, label = self.data[idx]
            full_path = os.path.join(self.base_path, rel_path)
            try:
                waveform, _ = load_audio_mono_16k(
                    full_path,
                    for_inference=self.preprocess_for_inference,
                )
                break
            except Exception as exc:
                last_exc = exc
                if self.train:
                    idx = random.randint(0, len(self.data) - 1)
                else:
                    idx = (idx + 1) % len(self.data)
        if waveform is None or label is None:
            raise RuntimeError(f"Failed to load sample after retries: {last_exc}")

        crop_reference_waveform = waveform
        if self.train and self.augment:
            if self.speed_perturb_factors:
                speed = random.choice(self.speed_perturb_factors)
                if abs(speed - 1.0) > 1e-6:
                    perturbed_rate = max(1000, int(round(16000.0 / speed)))
                    waveform = torchaudio.functional.resample(
                        waveform,
                        orig_freq=16000,
                        new_freq=perturbed_rate,
                    )
            crop_reference_waveform = waveform
            waveform = self._apply_reverb(waveform)
            waveform = self._apply_musan(waveform)
            gain = random.uniform(-self.gain_db, self.gain_db)
            waveform = waveform * (10.0 ** (gain / 20.0))
            waveform = waveform + self.noise_std * torch.randn_like(waveform)

        mel_spec = self.mel_transform(waveform)
        mel_spec = torch.log(mel_spec + 1e-6)

        _, _, time = mel_spec.shape
        if self.train:
            if self.train_crop_mode == "active":
                mel_spec = self._crop_training_active(crop_reference_waveform, mel_spec)
            else:
                mel_spec = self._crop_training_mel(mel_spec)
        else:
            if time < self.max_frames:
                pad_len = self.max_frames - time
                mel_spec = torch.nn.functional.pad(mel_spec, (0, pad_len))
            else:
                start = max((time - self.max_frames) // 2, 0)
                mel_spec = mel_spec[:, :, start : start + self.max_frames]

        if self.train and self.augment:
            mel_spec = self.freq_mask(mel_spec)
            mel_spec = self.time_mask(mel_spec)

        return mel_spec, label
