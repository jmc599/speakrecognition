import argparse
import csv
import math
import os
import random
import time
from collections import Counter

import torch
from torch.utils.data import DataLoader, Sampler, Subset
from tqdm import tqdm

from core.loss import AAMSoftmax
from core.model import ResNet34_SE
from data_loader.dataset import SpeakerDataset
from make_list import build_speaker_list
from utils.checkpoint_io import load_torch_checkpoint
from utils.speaker_verification import build_mel_transform
from utils.verification_eval import evaluate_trial_list, load_enroll_map, sample_trials

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    matplotlib = None
    plt = None


def parse_args():
    parser = argparse.ArgumentParser(description="Train speaker model on real data.")
    parser.add_argument(
        "--base-path",
        type=str,
        default=r"F:\cn_celeb\cn-celeb_v2\CN-Celeb_flac",
        help="Dataset root folder.",
    )
    parser.add_argument(
        "--list-path",
        type=str,
        default="lists/train_list.txt",
        help="Training list path.",
    )
    parser.add_argument(
        "--include-subdir",
        type=str,
        default="data",
        help="Subfolder under base-path used for list generation.",
    )
    parser.add_argument(
        "--rebuild-list",
        action="store_true",
        help="Force regenerate list before training.",
    )
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--accum-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps. 1 disables accumulation.",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "sgd"],
        help="Optimizer type.",
    )
    parser.add_argument(
        "--sgd-momentum",
        type=float,
        default=0.9,
        help="Momentum used when --optimizer sgd.",
    )
    parser.add_argument(
        "--sgd-nesterov",
        action="store_true",
        help="Enable Nesterov momentum when --optimizer sgd.",
    )
    parser.add_argument(
        "--fixed-lr",
        action="store_true",
        help="Disable scheduler and keep learning rate fixed.",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=0,
        help="Linear warmup epochs before cosine decay. 0 disables warmup.",
    )
    parser.add_argument(
        "--warmup-start-factor",
        type=float,
        default=0.1,
        help="Warmup starting LR factor relative to --lr.",
    )
    parser.add_argument(
        "--aam-scale",
        type=float,
        default=30.0,
        help="AAMSoftmax scale parameter.",
    )
    parser.add_argument(
        "--aam-margin",
        type=float,
        default=0.20,
        help="AAMSoftmax angular margin parameter.",
    )
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument(
        "--train-crop-mode",
        type=str,
        default="random",
        choices=["random", "active"],
        help=(
            "Training crop strategy. random keeps the legacy random 3-second crop; "
            "active prioritizes spans with detected speech activity."
        ),
    )
    parser.add_argument(
        "--speed-perturb-factors",
        type=str,
        default="",
        help=(
            "Comma-separated training speed perturb factors, e.g. 0.9,1.0,1.1. "
            "Empty disables speed perturb."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument(
        "--freq-mask-param",
        type=int,
        default=6,
        help="FrequencyMasking width for training augmentation.",
    )
    parser.add_argument(
        "--time-mask-param",
        type=int,
        default=10,
        help="TimeMasking width for training augmentation.",
    )
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--ckpt-prefix", type=str, default="cnceleb")
    parser.add_argument(
        "--train-sampler",
        type=str,
        default="random",
        choices=["random", "balanced_speaker"],
        help=(
            "Training sample strategy. balanced_speaker reweights samples by inverse "
            "speaker frequency on the train split."
        ),
    )
    parser.add_argument(
        "--train-sample-repeat-cap",
        type=int,
        default=3,
        help=(
            "Maximum repeat count per audio sample within one epoch when "
            "--train-sampler balanced_speaker is used. 0 disables the cap."
        ),
    )
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument(
        "--init-model",
        type=str,
        default="",
        help=(
            "Optional pretrained PyTorch checkpoint/state-dict path used to initialize "
            "model weights only (no optimizer/scheduler state). ONNX is not supported."
        ),
    )
    parser.add_argument(
        "--init-model-key",
        type=str,
        default="",
        help=(
            "Optional key name inside --init-model checkpoint dict that stores model "
            "state dict (e.g. model_state_dict/state_dict/model)."
        ),
    )
    parser.add_argument(
        "--init-model-strict",
        action="store_true",
        help="Load --init-model with strict=True. Default is strict=False.",
    )
    parser.add_argument(
        "--reset-optimizer",
        action="store_true",
        help="When resuming, reload model weights but reinitialize optimizer/scheduler.",
    )
    parser.add_argument(
        "--resume-lr",
        type=float,
        default=0.0,
        help="Override learning rate after resume. 0 keeps the configured lr.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument(
        "--monitor-base-path",
        type=str,
        default="",
        help="Base path for trial-based verification monitoring.",
    )
    parser.add_argument(
        "--monitor-trials",
        type=str,
        default="",
        help="Trial list used for verification monitoring.",
    )
    parser.add_argument(
        "--monitor-enroll-list",
        type=str,
        default="",
        help="Enrollment map used for verification monitoring.",
    )
    parser.add_argument(
        "--monitor-limit",
        type=int,
        default=2000,
        help="Randomly sample N official trials for per-epoch monitoring. 0 means all.",
    )
    parser.add_argument(
        "--monitor-seed",
        type=int,
        default=42,
        help="Random seed for trial sampling.",
    )
    parser.add_argument(
        "--monitor-sample-mode",
        type=str,
        default="balanced",
        choices=["balanced", "random"],
        help="Sampling mode used when monitor-limit > 0.",
    )
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument(
        "--show-grad-norm",
        action="store_true",
        help="Show model/classifier gradient norms on the train progress bar.",
    )
    parser.add_argument(
        "--tensorboard-logdir",
        type=str,
        default="",
        help="Optional TensorBoard log directory. Empty disables TensorBoard logging.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--metrics-csv",
        type=str,
        default="",
        help="Optional CSV path to append per-epoch training/eval metrics.",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable waveform/spec augmentation for training.",
    )
    parser.add_argument(
        "--preprocess-for-inference",
        action="store_true",
        help="Apply inference preprocessing (trim silence, normalize loudness) during training to align domains.",
    )
    return parser.parse_args()


def parse_speed_perturb_factors(raw_value):
    if not raw_value:
        return []
    factors = []
    for token in raw_value.split(","):
        token = token.strip()
        if not token:
            continue
        factor = float(token)
        if factor <= 0:
            raise ValueError("Speed perturb factors must be positive.")
        factors.append(factor)
    return factors


class EpochWarmupCosineScheduler:
    def __init__(
        self,
        optimizer,
        max_epochs,
        base_lr,
        min_lr,
        warmup_epochs=0,
        warmup_start_factor=0.1,
    ):
        self.optimizer = optimizer
        self.max_epochs = max(int(max_epochs), 1)
        self.base_lr = float(base_lr)
        self.min_lr = float(min_lr)
        self.warmup_epochs = max(int(warmup_epochs), 0)
        self.warmup_start_factor = float(warmup_start_factor)
        self.current_epoch = 0
        self._set_lr(self._lr_for_epoch(self.current_epoch))

    def _lr_for_epoch(self, epoch_index):
        if self.warmup_epochs > 0 and epoch_index < self.warmup_epochs:
            if self.warmup_epochs == 1:
                factor = 1.0
            else:
                factor = self.warmup_start_factor + (
                    (1.0 - self.warmup_start_factor)
                    * (epoch_index / (self.warmup_epochs - 1))
                )
            return self.base_lr * factor

        cosine_epochs = max(self.max_epochs - self.warmup_epochs, 1)
        cosine_index = max(epoch_index - self.warmup_epochs, 0)
        if cosine_epochs == 1:
            cosine_scale = 1.0
        else:
            cosine_scale = 0.5 * (
                1.0 + math.cos(math.pi * cosine_index / (cosine_epochs - 1))
            )
        return self.min_lr + (self.base_lr - self.min_lr) * cosine_scale

    def _set_lr(self, lr_value):
        for group in self.optimizer.param_groups:
            group["lr"] = lr_value

    def step(self):
        self.current_epoch += 1
        self._set_lr(self._lr_for_epoch(self.current_epoch))

    def state_dict(self):
        return {
            "current_epoch": self.current_epoch,
            "max_epochs": self.max_epochs,
            "base_lr": self.base_lr,
            "min_lr": self.min_lr,
            "warmup_epochs": self.warmup_epochs,
            "warmup_start_factor": self.warmup_start_factor,
        }

    def load_state_dict(self, state_dict):
        self.current_epoch = int(state_dict.get("current_epoch", 0))
        self.max_epochs = int(state_dict.get("max_epochs", self.max_epochs))
        self.base_lr = float(state_dict.get("base_lr", self.base_lr))
        self.min_lr = float(state_dict.get("min_lr", self.min_lr))
        self.warmup_epochs = int(state_dict.get("warmup_epochs", self.warmup_epochs))
        self.warmup_start_factor = float(
            state_dict.get("warmup_start_factor", self.warmup_start_factor)
        )
        self._set_lr(self._lr_for_epoch(self.current_epoch))


def build_optimizer(args, model, criterion):
    params = [{"params": model.parameters()}, {"params": criterion.parameters()}]
    if args.optimizer == "sgd":
        return torch.optim.SGD(
            params,
            lr=args.lr,
            momentum=args.sgd_momentum,
            nesterov=args.sgd_nesterov,
            weight_decay=2e-5,
        )
    return torch.optim.AdamW(
        params,
        lr=args.lr,
        weight_decay=2e-5,
    )


def build_scheduler(args, optimizer):
    if args.fixed_lr:
        return None
    return EpochWarmupCosineScheduler(
        optimizer=optimizer,
        max_epochs=max(args.epochs, 1),
        base_lr=args.lr,
        min_lr=args.min_lr,
        warmup_epochs=args.warmup_epochs,
        warmup_start_factor=args.warmup_start_factor,
    )


def maybe_build_list(args):
    need_build = args.rebuild_list or (not os.path.exists(args.list_path))
    if not need_build:
        return

    include_subdir = args.include_subdir.strip()
    total, n_spk = build_speaker_list(
        dataset_root=args.base_path,
        output_list=args.list_path,
        include_subdir=include_subdir,
    )
    print(f"Generated list: {args.list_path}")
    print(f"Found {total} audio files from {n_spk} speakers")


def format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes > 0:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


METRICS_FIELDS = [
    "epoch",
    "lr",
    "train_loss",
    "train_grad_model_norm",
    "train_grad_cls_norm",
    "val_loss",
    "selected_metric",
    "best_metric",
    "epoch_time_sec",
    "eta_sec",
    "num_trials",
    "num_target_trials",
    "num_non_target_trials",
    "eer",
    "eer_threshold",
    "min_dcf",
    "threshold_far_1",
    "frr_at_far_1",
    "threshold_far_0p1",
    "frr_at_far_0p1",
    "accuracy",
    "far",
    "frr",
    "tp",
    "tn",
    "fp",
    "fn",
    "precision",
    "recall",
    "f1",
    "target_mean",
    "non_target_mean",
]

class BalancedSpeakerSampler(Sampler):
    def __init__(
        self,
        speaker_labels,
        num_samples,
        max_repeat_per_sample=3,
        seed=42,
    ):
        self.num_samples = int(num_samples)
        self.max_repeat_per_sample = int(max_repeat_per_sample)
        self.seed = int(seed)
        self.epoch = 0
        self.speaker_to_indices = {}
        for idx, label in enumerate(speaker_labels):
            self.speaker_to_indices.setdefault(int(label), []).append(idx)
        self.speaker_ids = sorted(self.speaker_to_indices.keys())
        if not self.speaker_ids:
            raise RuntimeError("BalancedSpeakerSampler received no speaker labels.")

    def __len__(self):
        return self.num_samples

    def _speaker_capacity(self, speaker_id):
        sample_count = len(self.speaker_to_indices[speaker_id])
        if self.max_repeat_per_sample <= 0:
            return self.num_samples
        return sample_count * self.max_repeat_per_sample

    def _build_quotas(self, rng):
        shuffled = list(self.speaker_ids)
        rng.shuffle(shuffled)
        num_speakers = len(shuffled)
        base = self.num_samples // num_speakers
        remainder = self.num_samples % num_speakers

        quotas = {}
        remaining = self.num_samples
        for idx, speaker_id in enumerate(shuffled):
            target = base + (1 if idx < remainder else 0)
            quota = min(target, self._speaker_capacity(speaker_id))
            quotas[speaker_id] = quota
            remaining -= quota

        while remaining > 0:
            expandable = [
                speaker_id
                for speaker_id in shuffled
                if quotas[speaker_id] < self._speaker_capacity(speaker_id)
            ]
            if not expandable:
                break
            rng.shuffle(expandable)
            for speaker_id in expandable:
                if remaining <= 0:
                    break
                quotas[speaker_id] += 1
                remaining -= 1

        if remaining > 0:
            raise RuntimeError(
                "Balanced speaker sampler cannot fill an epoch with the current repeat cap. "
                "Increase --train-sample-repeat-cap or switch to --train-sampler random."
            )
        return quotas

    @staticmethod
    def _allocate_indices(indices, quota, rng):
        if quota <= 0:
            return []
        result = []
        count = len(indices)
        full_cycles, remainder = divmod(quota, count)
        for _ in range(full_cycles):
            cycle = list(indices)
            rng.shuffle(cycle)
            result.extend(cycle)
        if remainder:
            cycle = list(indices)
            rng.shuffle(cycle)
            result.extend(cycle[:remainder])
        return result

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        quotas = self._build_quotas(rng)
        sampled = []
        for speaker_id in self.speaker_ids:
            sampled.extend(
                self._allocate_indices(
                    self.speaker_to_indices[speaker_id],
                    quotas[speaker_id],
                    rng,
                )
            )
        rng.shuffle(sampled)
        return iter(sampled)



def _binary_prf(tp, fp, fn, eps=1e-12):
    precision = float(tp / max(tp + fp, eps))
    recall = float(tp / max(tp + fn, eps))
    f1 = float((2.0 * precision * recall) / max(precision + recall, eps))
    return precision, recall, f1


def _compute_grad_norm(parameters, eps=1e-12):
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        grad = param.grad.detach()
        norm = float(grad.norm(2).item())
        total += norm * norm
    return (total + eps) ** 0.5


def _ensure_metrics_csv(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_FIELDS)
        writer.writeheader()


def _append_metrics_row(path, row):
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_FIELDS)
        writer.writerow(row)


def _plot_metric_curves(history_rows, output_path):
    if plt is None:
        return
    if not history_rows:
        return

    epochs = [int(row["epoch"]) for row in history_rows]

    def values(key, scale=1.0):
        out = []
        for row in history_rows:
            value = row.get(key)
            out.append(None if value is None else float(value) * scale)
        return out

    train_loss = values("train_loss")
    val_loss = values("val_loss")
    eer = values("eer", scale=100.0)
    min_dcf = values("min_dcf")
    accuracy = values("accuracy", scale=100.0)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plots = [
        ("Loss", train_loss, val_loss, "Train Loss", "Val Loss"),
        ("EER (%)", eer, None, "EER", None),
        ("minDCF", min_dcf, None, "minDCF", None),
        ("Accuracy (%)", accuracy, None, "Accuracy", None),
    ]

    for ax, (title, primary, secondary, primary_label, secondary_label) in zip(
        axes.flatten(), plots
    ):
        primary_points = [
            (epoch, value)
            for epoch, value in zip(epochs, primary)
            if value is not None
        ]
        if primary_points:
            ax.plot(
                [x for x, _ in primary_points],
                [y for _, y in primary_points],
                marker="o",
                label=primary_label,
            )
        if secondary is not None:
            secondary_points = [
                (epoch, value)
                for epoch, value in zip(epochs, secondary)
                if value is not None
            ]
            if secondary_points:
                ax.plot(
                    [x for x, _ in secondary_points],
                    [y for _, y in secondary_points],
                    marker="o",
                    label=secondary_label,
                )
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        if ax.lines:
            ax.legend()

    fig.tight_layout()
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _extract_state_dict(checkpoint, preferred_key=""):
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "Init checkpoint is not a dict-like state container. "
            "Please provide a PyTorch checkpoint/state-dict."
        )

    if preferred_key:
        value = checkpoint.get(preferred_key)
        if not isinstance(value, dict):
            raise KeyError(
                f"Key '{preferred_key}' not found as dict in init checkpoint."
            )
        return value, preferred_key

    candidate_keys = [
        "model_state_dict",
        "state_dict",
        "model",
        "backbone_state_dict",
        "encoder_state_dict",
        "net",
        "weights",
    ]
    for key in candidate_keys:
        value = checkpoint.get(key)
        if isinstance(value, dict) and any(torch.is_tensor(v) for v in value.values()):
            return value, key

    if any(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint, "<root>"

    for key, value in checkpoint.items():
        if isinstance(value, dict) and any(torch.is_tensor(v) for v in value.values()):
            return value, key

    raise RuntimeError(
        "No tensor state-dict found in init checkpoint. "
        "Try --init-model-key with the correct key."
    )


def _normalize_state_dict_keys(state_dict):
    prefixes = (
        "module.",
        "model.",
        "backbone.",
        "speaker_model.",
        "encoder.",
    )
    normalized = {}
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        out_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if out_key.startswith(prefix):
                    out_key = out_key[len(prefix) :]
                    changed = True
        normalized[out_key] = value
    return normalized


def _categorize_model_key(key):
    if key.startswith(("input_norm.", "conv1.", "bn1.")):
        return "stem/front"
    for idx in range(1, 5):
        if key.startswith(f"layer{idx}."):
            return f"layer{idx}"
    if key.startswith(("pool.", "pool_bn.", "fc.", "bn_final.")):
        return "pooling/projection"
    return "other"


def _summarize_module_coverage(model_state_dict, loaded_keys):
    categories = [
        "stem/front",
        "layer1",
        "layer2",
        "layer3",
        "layer4",
        "pooling/projection",
        "classifier head",
        "other",
    ]
    totals = {key: 0 for key in categories}
    loaded = {key: 0 for key in categories}

    loaded_set = set(loaded_keys)
    for key in model_state_dict:
        if key.endswith("num_batches_tracked"):
            continue
        category = _categorize_model_key(key)
        totals[category] += 1
        if key in loaded_set:
            loaded[category] += 1

    totals["classifier head"] = 0
    loaded["classifier head"] = 0

    return {
        category: {
            "loaded": int(loaded[category]),
            "total": int(totals[category]),
            "ratio": (
                0.0
                if totals[category] <= 0
                else float(loaded[category] / totals[category])
            ),
        }
        for category in categories
    }


def maybe_init_model_weights(model, args, device):
    if not args.init_model:
        return
    if args.resume:
        print("--init-model is ignored because --resume is provided.")
        return
    if args.init_model.lower().endswith(".onnx"):
        raise RuntimeError(
            "--init-model does not support ONNX. "
            "Please provide a PyTorch checkpoint (.pth/.pt)."
        )
    checkpoint = load_torch_checkpoint(args.init_model, map_location="cpu")
    state_dict, source_key = _extract_state_dict(
        checkpoint, preferred_key=args.init_model_key
    )
    state_dict = _normalize_state_dict_keys(state_dict)
    if not state_dict:
        raise RuntimeError("Extracted init state-dict is empty.")
    model_state = model.state_dict()
    filtered_state = {}
    shape_mismatch = []
    unexpected = []
    for key, value in state_dict.items():
        target_value = model_state.get(key)
        if target_value is None:
            unexpected.append(key)
            continue
        if tuple(target_value.shape) != tuple(value.shape):
            shape_mismatch.append(key)
            continue
        filtered_state[key] = value

    incompatible = model.load_state_dict(
        filtered_state, strict=bool(args.init_model_strict)
    )
    missing = list(getattr(incompatible, "missing_keys", []))
    coverage = _summarize_module_coverage(model_state, filtered_state.keys())
    backbone_total = sum(
        entry["total"]
        for name, entry in coverage.items()
        if name not in {"classifier head", "other"}
    )
    backbone_loaded = sum(
        entry["loaded"]
        for name, entry in coverage.items()
        if name not in {"classifier head", "other"}
    )
    backbone_ratio = (
        0.0 if backbone_total <= 0 else float(backbone_loaded / backbone_total)
    )
    print(
        f"Initialized model from: {args.init_model} "
        f"(source key: {source_key}, strict={bool(args.init_model_strict)})"
    )
    print(
        f"Init load summary: loaded={len(filtered_state)} "
        f"shape_mismatch={len(shape_mismatch)} missing={len(missing)} "
        f"unexpected={len(unexpected)}"
    )
    for name in [
        "stem/front",
        "layer1",
        "layer2",
        "layer3",
        "layer4",
        "pooling/projection",
        "classifier head",
    ]:
        entry = coverage[name]
        if entry["total"] > 0:
            print(
                f"Init module coverage [{name}]: "
                f"{entry['loaded']}/{entry['total']} ({entry['ratio'] * 100:.1f}%)"
            )
        else:
            print(f"Init module coverage [{name}]: n/a")
    if shape_mismatch:
        print("Shape mismatch keys (first 8):", shape_mismatch[:8])
    if missing:
        print("Missing keys (first 8):", missing[:8])
    if unexpected:
        print("Unexpected keys (first 8):", unexpected[:8])
    if backbone_ratio < 0.5:
        print(
            "WARNING: weak-compatible init detected "
            f"(backbone coverage {backbone_ratio * 100:.1f}% < 50.0%)"
        )


def build_loaders(args, device):
    speed_perturb_factors = parse_speed_perturb_factors(args.speed_perturb_factors)
    train_dataset = SpeakerDataset(
        data_list_path=args.list_path,
        base_path=args.base_path,
        max_frames=args.max_frames,
        train=True,
        augment=not args.no_augment,
        freq_mask_param=args.freq_mask_param,
        time_mask_param=args.time_mask_param,
        speed_perturb_factors=speed_perturb_factors,
        preprocess_for_inference=args.preprocess_for_inference,
        train_crop_mode=args.train_crop_mode,
    )
    val_dataset = SpeakerDataset(
        data_list_path=args.list_path,
        base_path=args.base_path,
        max_frames=args.max_frames,
        train=False,
        augment=False,
        preprocess_for_inference=args.preprocess_for_inference,
    )

    dataset_size = len(train_dataset)
    if dataset_size == 0:
        raise RuntimeError("Empty dataset. Check base-path and list-path.")

    generator = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(dataset_size, generator=generator).tolist()

    val_size = int(dataset_size * args.val_ratio)
    if args.val_ratio > 0 and dataset_size > 1:
        val_size = max(1, min(val_size, dataset_size - 1))
    else:
        val_size = 0

    val_indices = perm[:val_size]
    train_indices = perm[val_size:]

    if val_indices:
        all_speaker_counts = Counter(label for _, label in train_dataset.data)
        train_speaker_counts = Counter(train_dataset.data[idx][1] for idx in train_indices)
        val_speaker_to_indices = {}
        for idx in val_indices:
            label = train_dataset.data[idx][1]
            val_speaker_to_indices.setdefault(label, []).append(idx)
        moved_from_val = []
        moved_to_val = []
        for speaker_id, total_count in all_speaker_counts.items():
            if total_count <= 1 or train_speaker_counts[speaker_id] > 0:
                continue
            restore_idx = val_speaker_to_indices[speaker_id].pop()
            swap_idx = None
            for candidate in reversed(train_indices):
                candidate_label = train_dataset.data[candidate][1]
                if train_speaker_counts[candidate_label] > 1:
                    swap_idx = candidate
                    break
            if swap_idx is None:
                train_indices.append(restore_idx)
                val_indices.remove(restore_idx)
                moved_from_val.append(restore_idx)
            else:
                train_indices.remove(swap_idx)
                val_indices.remove(restore_idx)
                train_indices.append(restore_idx)
                val_indices.append(swap_idx)
                moved_from_val.append(restore_idx)
                moved_to_val.append(swap_idx)
                swap_label = train_dataset.data[swap_idx][1]
                train_speaker_counts[swap_label] -= 1
            train_speaker_counts[speaker_id] += 1
        if moved_from_val:
            print(
                f"Adjusted split to keep {len(moved_from_val)} speakers in train "
                f"(swapped_out={len(moved_to_val)})."
            )

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices) if val_indices else None

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device == "cuda",
    }
    train_sampler = None
    train_shuffle = True
    train_speaker_labels = [train_dataset.data[idx][1] for idx in train_indices]
    train_speaker_counts = Counter(train_speaker_labels)
    if args.train_sampler == "balanced_speaker":
        train_sampler = BalancedSpeakerSampler(
            speaker_labels=train_speaker_labels,
            num_samples=len(train_speaker_labels),
            max_repeat_per_sample=args.train_sample_repeat_cap,
            seed=args.seed,
        )
        train_shuffle = False

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        **loader_kwargs,
    )
    val_loader = None
    if val_subset is not None:
        val_loader = DataLoader(
            val_subset,
            batch_size=args.batch_size,
            shuffle=False,
            **loader_kwargs,
        )

    num_classes = len({spk_id for _, spk_id in train_dataset.data})
    print(
        f"Train samples: {len(train_subset)} | "
        f"Val samples: {len(val_subset) if val_subset is not None else 0} | "
        f"Num speakers: {num_classes}"
    )
    print(f"Optimizer: {args.optimizer}")
    print(f"Gradient accumulation steps: {args.accum_steps}")
    print(f"Train sampler: {args.train_sampler}")
    sorted_counts = sorted(train_speaker_counts.values())
    print(
        "Train speaker distribution: "
        f"min={sorted_counts[0]} "
        f"median={sorted_counts[len(sorted_counts)//2]} "
        f"max={sorted_counts[-1]}"
    )
    if args.train_sampler == "balanced_speaker":
        print(f"Train sample repeat cap: {args.train_sample_repeat_cap}")
    if speed_perturb_factors:
        print(f"Train speed perturb factors: {speed_perturb_factors}")
    if not args.no_augment:
        print(
            "SpecAugment params: "
            f"freq_mask={args.freq_mask_param} time_mask={args.time_mask_param}"
        )
    return train_loader, val_loader, num_classes


def run_train_epoch(
    model,
    criterion,
    optimizer,
    loader,
    device,
    epoch,
    total_epochs,
    grad_clip,
    accum_steps=1,
    show_grad_norm=False,
    tb_writer=None,
    global_step_start=0,
):
    model.train()
    criterion.train()
    total_loss = 0.0
    total_model_grad_norm = 0.0
    total_cls_grad_norm = 0.0
    grad_update_steps = 0
    global_step = int(global_step_start)

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [train]")
    optimizer.zero_grad()
    num_batches = len(loader)
    for batch_idx, (x, y) in enumerate(pbar):
        x = x.to(device)
        y = y.to(device=device, dtype=torch.long)

        emb = model(x)
        loss = criterion(emb, y)
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite training loss at epoch {epoch}. "
                "Stop and resume from the last good checkpoint."
            )
        scaled_loss = loss / max(int(accum_steps), 1)
        scaled_loss.backward()
        total_loss += loss.item()
        should_step = (
            ((batch_idx + 1) % max(int(accum_steps), 1) == 0)
            or (batch_idx + 1 == num_batches)
        )
        model_grad_norm = 0.0
        cls_grad_norm = 0.0
        if should_step:
            if grad_clip > 0:
                model_grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                )
                cls_grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        criterion.parameters(), max_norm=grad_clip
                    )
                )
            else:
                model_grad_norm = _compute_grad_norm(model.parameters())
                cls_grad_norm = _compute_grad_norm(criterion.parameters())
            optimizer.step()
            optimizer.zero_grad()
            total_model_grad_norm += model_grad_norm
            total_cls_grad_norm += cls_grad_norm
            grad_update_steps += 1

        postfix = {"loss": f"{loss.item():.4f}"}
        if show_grad_norm and should_step:
            postfix["g_model"] = f"{model_grad_norm:.2f}"
            postfix["g_cls"] = f"{cls_grad_norm:.2f}"
        pbar.set_postfix(**postfix)

        if tb_writer is not None:
            tb_writer.add_scalar("train/loss_step", float(loss.item()), global_step)
            if should_step:
                tb_writer.add_scalar(
                    "train/grad_model_norm_step", float(model_grad_norm), global_step
                )
                tb_writer.add_scalar(
                    "train/grad_cls_norm_step", float(cls_grad_norm), global_step
                )
        global_step += 1

    num_steps = max(len(loader), 1)
    grad_steps = max(grad_update_steps, 1)
    return {
        "loss": total_loss / num_steps,
        "grad_model_norm": total_model_grad_norm / grad_steps,
        "grad_cls_norm": total_cls_grad_norm / grad_steps,
        "global_step": global_step,
    }


@torch.no_grad()
def run_eval_epoch(model, criterion, loader, device, epoch, total_epochs):
    model.eval()
    criterion.eval()
    total_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [val]")
    for x, y in pbar:
        x = x.to(device)
        y = y.to(device=device, dtype=torch.long)
        emb = model(x)
        loss = criterion(emb, y)
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    maybe_build_list(args)
    train_loader, val_loader, num_classes = build_loaders(args, device)
    monitor_trials = []
    mel_transform = build_mel_transform()
    monitor_fields = [
        bool(args.monitor_base_path),
        bool(args.monitor_trials),
        bool(args.monitor_enroll_list),
    ]
    if any(monitor_fields) and not all(monitor_fields):
        raise ValueError(
            "--monitor-base-path, --monitor-trials, and --monitor-enroll-list "
            "must either all be provided or all be empty."
        )
    if all(monitor_fields):
        enroll_map = load_enroll_map(args.monitor_enroll_list)
        monitor_trials = sample_trials(
            args.monitor_trials,
            enroll_map=enroll_map,
            limit=args.monitor_limit,
            seed=args.monitor_seed,
            sample_mode=args.monitor_sample_mode,
        )
        target_trials = sum(label for _, _, label in monitor_trials)
        print(
            f"Monitor trials: {len(monitor_trials)} "
            f"(target={target_trials}, non_target={len(monitor_trials) - target_trials})"
        )

    model = ResNet34_SE(
        embedding_dim=args.embedding_dim,
        input_channels=1,
    ).to(device)
    maybe_init_model_weights(model, args, device)
    criterion = AAMSoftmax(
        in_features=args.embedding_dim,
        n_class=num_classes,
        s=args.aam_scale,
        m=args.aam_margin,
    ).to(device)
    optimizer = build_optimizer(args, model, criterion)
    scheduler = build_scheduler(args, optimizer)

    os.makedirs(args.save_dir, exist_ok=True)
    metrics_csv_path = args.metrics_csv or os.path.join(
        args.save_dir, f"{args.ckpt_prefix}_metrics.csv"
    )
    _ensure_metrics_csv(metrics_csv_path)
    print(f"Metrics CSV: {metrics_csv_path}")
    tb_writer = None
    if args.tensorboard_logdir:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "TensorBoard logging requested, but tensorboard is not installed. "
                "Install it with: pip install tensorboard"
            ) from exc
        tb_writer = SummaryWriter(log_dir=args.tensorboard_logdir)
        print(f"TensorBoard logdir: {args.tensorboard_logdir}")
    global_step = 0
    metrics_history = []

    start_epoch = 1
    best_metric = float("inf")

    if args.resume:
        checkpoint = load_torch_checkpoint(args.resume, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            if "criterion_state_dict" in checkpoint:
                criterion.load_state_dict(checkpoint["criterion_state_dict"])
            if (not args.reset_optimizer) and "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint.get("epoch", 0) + 1
            best_metric = checkpoint.get("best_metric", checkpoint.get("best_loss", best_metric))
            saved_args = checkpoint.get("args", {})
            saved_epochs = saved_args.get("epochs")
            if (
                scheduler is not None
                and (not args.reset_optimizer)
                and "scheduler_state_dict" in checkpoint
                and checkpoint["scheduler_state_dict"] is not None
                and saved_epochs == args.epochs
            ):
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            elif scheduler is not None:
                # Rebuild the scheduler when the target epoch count changes.
                for group in optimizer.param_groups:
                    group["lr"] = args.lr
                scheduler = build_scheduler(args, optimizer)
                for _ in range(start_epoch - 1):
                    scheduler.step()
                if saved_epochs is not None and saved_epochs != args.epochs:
                    print(
                        f"Scheduler reset for new total epochs: "
                        f"{saved_epochs} -> {args.epochs}"
                    )
            if args.reset_optimizer:
                print("Optimizer and scheduler were reset on resume.")
            if args.resume_lr > 0:
                for group in optimizer.param_groups:
                    group["lr"] = args.resume_lr
                if not args.fixed_lr:
                    original_lr = args.lr
                    args.lr = args.resume_lr
                    scheduler = build_scheduler(args, optimizer)
                    args.lr = original_lr
                    for _ in range(start_epoch - 1):
                        scheduler.step()
                print(f"Resume learning rate overridden to {args.resume_lr:.6f}")
            print(
                f"Resumed full checkpoint: {args.resume} "
                f"(next epoch: {start_epoch})"
            )
            if isinstance(checkpoint.get("global_step"), int):
                global_step = int(checkpoint["global_step"])
        else:
            model.load_state_dict(checkpoint)
            print(f"Loaded model weights only: {args.resume}")

    train_start = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_stats = run_train_epoch(
            model,
            criterion,
            optimizer,
            train_loader,
            device,
            epoch,
            args.epochs,
            args.grad_clip,
            accum_steps=args.accum_steps,
            show_grad_norm=args.show_grad_norm,
            tb_writer=tb_writer,
            global_step_start=global_step,
        )
        train_loss = float(train_stats["loss"])
        train_grad_model_norm = float(train_stats["grad_model_norm"])
        train_grad_cls_norm = float(train_stats["grad_cls_norm"])
        global_step = int(train_stats["global_step"])
        val_loss = None
        if val_loader is not None and len(val_loader) > 0:
            val_loss = run_eval_epoch(
                model, criterion, val_loader, device, epoch, args.epochs
            )
        veri_metrics = None
        if monitor_trials:
            veri_metrics = evaluate_trial_list(
                model,
                device=device,
                mel_transform=mel_transform,
                base_path=args.monitor_base_path,
                trials=monitor_trials,
                max_frames=args.max_frames,
                num_eval=5,
                preprocess_for_inference=args.preprocess_for_inference,
                show_progress=True,
            )

        if scheduler is not None:
            scheduler.step()

        epoch_time = time.time() - epoch_start
        elapsed = time.time() - train_start
        completed_epochs = epoch - start_epoch + 1
        avg_epoch_time = elapsed / completed_epochs
        remaining_epochs = args.epochs - epoch
        eta = avg_epoch_time * remaining_epochs

        metric = veri_metrics["eer"] if veri_metrics is not None else (
            val_loss if val_loss is not None else train_loss
        )
        if metric < best_metric:
            best_metric = metric

        summary = (
            f"Epoch {epoch} train loss: {train_loss:.4f} | "
            f"lr: {current_lr:.6f}"
        )
        if args.show_grad_norm:
            summary += (
                f" | g_model: {train_grad_model_norm:.2f}"
                f" | g_cls: {train_grad_cls_norm:.2f}"
            )
        if val_loss is not None:
            summary += f" | val loss: {val_loss:.4f}"
        if veri_metrics is not None:
            summary += (
                f" | val EER: {veri_metrics['eer'] * 100:.2f}%"
                f" | FRR@FAR<=1%: {veri_metrics['frr_at_far_1'] * 100:.2f}%"
                f" | t_mean: {veri_metrics['target_mean']:.4f}"
                f" | nt_mean: {veri_metrics['non_target_mean']:.4f}"
            )
        summary += (
            f" | epoch time: {format_duration(epoch_time)}"
            f" | remaining: {format_duration(eta)}"
        )
        print(summary)
        if tb_writer is not None:
            tb_writer.add_scalar("train/loss_epoch", train_loss, epoch)
            tb_writer.add_scalar("train/grad_model_norm_epoch", train_grad_model_norm, epoch)
            tb_writer.add_scalar("train/grad_cls_norm_epoch", train_grad_cls_norm, epoch)
            tb_writer.add_scalar("optim/lr", float(current_lr), epoch)
            if val_loss is not None:
                tb_writer.add_scalar("val/loss_epoch", float(val_loss), epoch)
            if veri_metrics is not None:
                tb_writer.add_scalar("val/eer_epoch", float(veri_metrics["eer"]), epoch)
                tb_writer.add_scalar("val/min_dcf_epoch", float(veri_metrics["min_dcf"]), epoch)
                tb_writer.add_scalar("val/frr_at_far_1_epoch", float(veri_metrics["frr_at_far_1"]), epoch)
                tb_writer.add_scalar("val/frr_at_far_0p1_epoch", float(veri_metrics["frr_at_far_0p1"]), epoch)
                tb_writer.add_scalar("val/far_epoch", float(veri_metrics["far"]), epoch)
                tb_writer.add_scalar("val/frr_epoch", float(veri_metrics["frr"]), epoch)
                tb_writer.add_scalar("val/accuracy_epoch", float(veri_metrics["accuracy"]), epoch)

        if veri_metrics is not None:
            precision, recall, f1 = _binary_prf(
                tp=veri_metrics["tp"],
                fp=veri_metrics["fp"],
                fn=veri_metrics["fn"],
            )
        else:
            precision = recall = f1 = None

        metrics_row = {
            "epoch": int(epoch),
            "lr": float(current_lr),
            "train_loss": float(train_loss),
            "train_grad_model_norm": float(train_grad_model_norm),
            "train_grad_cls_norm": float(train_grad_cls_norm),
            "val_loss": float(val_loss) if val_loss is not None else None,
            "selected_metric": float(metric),
            "best_metric": float(best_metric),
            "epoch_time_sec": float(epoch_time),
            "eta_sec": float(eta),
            "num_trials": int(veri_metrics["num_trials"]) if veri_metrics is not None else None,
            "num_target_trials": int(veri_metrics["num_target_trials"]) if veri_metrics is not None else None,
            "num_non_target_trials": int(veri_metrics["num_non_target_trials"]) if veri_metrics is not None else None,
            "eer": float(veri_metrics["eer"]) if veri_metrics is not None else None,
            "eer_threshold": float(veri_metrics["eer_threshold"]) if veri_metrics is not None else None,
            "min_dcf": float(veri_metrics["min_dcf"]) if veri_metrics is not None else None,
            "threshold_far_1": float(veri_metrics["threshold_far_1"]) if veri_metrics is not None else None,
            "frr_at_far_1": float(veri_metrics["frr_at_far_1"]) if veri_metrics is not None else None,
            "threshold_far_0p1": float(veri_metrics["threshold_far_0p1"]) if veri_metrics is not None else None,
            "frr_at_far_0p1": float(veri_metrics["frr_at_far_0p1"]) if veri_metrics is not None else None,
            "accuracy": float(veri_metrics["accuracy"]) if veri_metrics is not None else None,
            "far": float(veri_metrics["far"]) if veri_metrics is not None else None,
            "frr": float(veri_metrics["frr"]) if veri_metrics is not None else None,
            "tp": int(veri_metrics["tp"]) if veri_metrics is not None else None,
            "tn": int(veri_metrics["tn"]) if veri_metrics is not None else None,
            "fp": int(veri_metrics["fp"]) if veri_metrics is not None else None,
            "fn": int(veri_metrics["fn"]) if veri_metrics is not None else None,
            "precision": float(precision) if precision is not None else None,
            "recall": float(recall) if recall is not None else None,
            "f1": float(f1) if f1 is not None else None,
            "target_mean": float(veri_metrics["target_mean"]) if veri_metrics is not None else None,
            "non_target_mean": float(veri_metrics["non_target_mean"]) if veri_metrics is not None else None,
        }
        _append_metrics_row(metrics_csv_path, metrics_row)
        metrics_history.append(metrics_row)
        curves_path = os.path.join(
            os.path.dirname(metrics_csv_path) or ".",
            f"{args.ckpt_prefix}_curves.png",
        )
        if plt is not None:
            _plot_metric_curves(metrics_history, curves_path)
            print(f"Updated curves: {curves_path}")

        checkpoint = {
            "epoch": epoch,
            "best_metric": best_metric,
            "global_step": int(global_step),
            "model_state_dict": model.state_dict(),
            "criterion_state_dict": criterion.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "verification_metrics": veri_metrics,
            "args": vars(args),
        }

        save_path = os.path.join(args.save_dir, f"{args.ckpt_prefix}_epoch_{epoch}.pth")
        latest_path = os.path.join(args.save_dir, f"{args.ckpt_prefix}_latest.pth")
        torch.save(checkpoint, save_path)
        torch.save(checkpoint, latest_path)
        print("Saved:", save_path)
        print("Updated latest:", latest_path)

        if metric == best_metric:
            best_path = os.path.join(args.save_dir, f"{args.ckpt_prefix}_best.pth")
            torch.save(checkpoint, best_path)
            if veri_metrics is not None:
                print(f"Updated best: {best_path} (val EER {best_metric * 100:.2f}%)")
            else:
                label = "val loss" if val_loss is not None else "train loss"
                print(f"Updated best: {best_path} ({label} {best_metric:.4f})")

    if tb_writer is not None:
        tb_writer.close()
    print("Done!")


if __name__ == "__main__":
    main()
