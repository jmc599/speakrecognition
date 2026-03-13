import argparse
import os
import time

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from core.loss import AAMSoftmax
from core.model import ResNet34_SE
from data_loader.dataset import SpeakerDataset
from make_list import build_speaker_list
from utils.speaker_verification import build_mel_transform
from utils.verification_eval import evaluate_trial_list, load_enroll_map, sample_trials


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
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument(
        "--fixed-lr",
        action="store_true",
        help="Disable scheduler and keep learning rate fixed.",
    )
    parser.add_argument(
        "--aam-scale",
        type=float,
        default=20.0,
        help="AAMSoftmax scale parameter.",
    )
    parser.add_argument(
        "--aam-margin",
        type=float,
        default=0.20,
        help="AAMSoftmax angular margin parameter.",
    )
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--ckpt-prefix", type=str, default="cnceleb")
    parser.add_argument("--resume", type=str, default="")
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
        default=r"F:\cn_celeb\cn-celeb_v2\CN-Celeb_flac\eval",
        help="Base path for trial-based verification monitoring.",
    )
    parser.add_argument(
        "--monitor-trials",
        type=str,
        default=r"F:\cn_celeb\cn-celeb_v2\CN-Celeb_flac\eval\lists\trials.lst",
        help="Trial list used for verification monitoring.",
    )
    parser.add_argument(
        "--monitor-enroll-list",
        type=str,
        default=r"F:\cn_celeb\cn-celeb_v2\CN-Celeb_flac\eval\lists\enroll.lst",
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable waveform/spec augmentation for training.",
    )
    return parser.parse_args()


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


def build_loaders(args, device):
    train_dataset = SpeakerDataset(
        data_list_path=args.list_path,
        base_path=args.base_path,
        max_frames=args.max_frames,
        train=True,
        augment=not args.no_augment,
    )
    val_dataset = SpeakerDataset(
        data_list_path=args.list_path,
        base_path=args.base_path,
        max_frames=args.max_frames,
        train=False,
        augment=False,
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

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices) if val_indices else None

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device == "cuda",
    }
    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
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
):
    model.train()
    criterion.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [train]")
    for x, y in pbar:
        x = x.to(device)
        y = y.to(device=device, dtype=torch.long)

        optimizer.zero_grad()
        emb = model(x)
        loss = criterion(emb, y)
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite training loss at epoch {epoch}. "
                "Stop and resume from the last good checkpoint."
            )
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            torch.nn.utils.clip_grad_norm_(criterion.parameters(), max_norm=grad_clip)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)


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
    if args.monitor_trials:
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
    criterion = AAMSoftmax(
        in_features=args.embedding_dim,
        n_class=num_classes,
        s=args.aam_scale,
        m=args.aam_margin,
    ).to(device)
    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}, {"params": criterion.parameters()}],
        lr=args.lr,
    )
    scheduler = None
    if not args.fixed_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(args.epochs, 1),
            eta_min=args.min_lr,
        )

    os.makedirs(args.save_dir, exist_ok=True)

    start_epoch = 1
    best_metric = float("inf")

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
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
                    group["initial_lr"] = args.lr
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=max(args.epochs, 1),
                    eta_min=args.min_lr,
                )
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
                    group["initial_lr"] = args.resume_lr
                if not args.fixed_lr:
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer,
                        T_max=max(args.epochs, 1),
                        eta_min=min(args.min_lr, args.resume_lr * 0.1),
                    )
                    for _ in range(start_epoch - 1):
                        scheduler.step()
                print(f"Resume learning rate overridden to {args.resume_lr:.6f}")
            print(
                f"Resumed full checkpoint: {args.resume} "
                f"(next epoch: {start_epoch})"
            )
        else:
            model.load_state_dict(checkpoint)
            print(f"Loaded model weights only: {args.resume}")

    train_start = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss = run_train_epoch(
            model,
            criterion,
            optimizer,
            train_loader,
            device,
            epoch,
            args.epochs,
            args.grad_clip,
        )
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
        if val_loss is not None:
            summary += f" | val loss: {val_loss:.4f}"
        if veri_metrics is not None:
            summary += (
                f" | val EER: {veri_metrics['eer'] * 100:.2f}%"
                f" | t_mean: {veri_metrics['target_mean']:.4f}"
                f" | nt_mean: {veri_metrics['non_target_mean']:.4f}"
            )
        summary += (
            f" | epoch time: {format_duration(epoch_time)}"
            f" | remaining: {format_duration(eta)}"
        )
        print(summary)

        checkpoint = {
            "epoch": epoch,
            "best_metric": best_metric,
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

    print("Done!")


if __name__ == "__main__":
    main()
