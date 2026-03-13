import os
import random

import torch
import torchaudio
from torch.utils.data import Dataset

from utils.audio_io import load_audio_mono_16k


class SpeakerDataset(Dataset):
    def __init__(
        self,
        data_list_path,
        base_path,
        max_frames=200,
        train=True,
        augment=False,
        noise_std=0.003,
        gain_db=6.0,
        freq_mask_param=6,
        time_mask_param=10,
    ):
        """
        data_list_path: each line is '<relative_path> <speaker_id>'
        base_path: dataset root directory
        """
        self.base_path = base_path
        self.max_frames = max_frames
        self.train = train
        self.augment = augment
        self.noise_std = noise_std
        self.gain_db = gain_db

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
            n_mels=64,
        )
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rel_path, label = self.data[idx]
        full_path = os.path.join(self.base_path, rel_path)

        waveform, _ = load_audio_mono_16k(full_path)

        if self.train and self.augment:
            gain = random.uniform(-self.gain_db, self.gain_db)
            waveform = waveform * (10.0 ** (gain / 20.0))
            waveform = waveform + self.noise_std * torch.randn_like(waveform)

        mel_spec = self.mel_transform(waveform)
        mel_spec = torch.log(mel_spec + 1e-6)

        _, _, time = mel_spec.shape
        if time < self.max_frames:
            pad_len = self.max_frames - time
            mel_spec = torch.nn.functional.pad(mel_spec, (0, pad_len))
        else:
            if self.train:
                start = random.randint(0, time - self.max_frames)
            else:
                start = max((time - self.max_frames) // 2, 0)
            mel_spec = mel_spec[:, :, start : start + self.max_frames]

        if self.train and self.augment:
            mel_spec = self.freq_mask(mel_spec)
            mel_spec = self.time_mask(mel_spec)

        return mel_spec, label
