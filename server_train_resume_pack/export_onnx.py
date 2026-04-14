import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model import ResNet34_SE
from utils.checkpoint_io import load_torch_checkpoint
from utils.legacy_feature import DEFAULT_MAX_FRAMES, LEGACY_NUM_MELS


class EmbeddingExportWrapper(nn.Module):
    def __init__(self, model, normalize_output=False):
        super().__init__()
        self.model = model
        self.normalize_output = normalize_output

    def forward(self, x):
        emb = self.model(x)
        if self.normalize_output:
            emb = F.normalize(emb, dim=1)
        return emb


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def infer_embedding_dim(state_dict):
    fc_weight = state_dict.get("fc.weight")
    if fc_weight is None:
        raise KeyError("Cannot infer embedding_dim from checkpoint: missing fc.weight")
    return int(fc_weight.shape[0])


def load_model_for_export(checkpoint_path, embedding_dim=None):
    checkpoint = load_torch_checkpoint(checkpoint_path, map_location="cpu")
    state_dict = extract_state_dict(checkpoint)

    if embedding_dim is None:
        embedding_dim = infer_embedding_dim(state_dict)

    model = ResNet34_SE(embedding_dim=embedding_dim, input_channels=1)
    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export the speaker embedding model to ONNX."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/resnet_v7_best.pth",
        help="Path to the PyTorch checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output ONNX path. Defaults to a fixed or dynamic filename based on --fixed-frames.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=0,
        help="Embedding dimension. Default: infer from checkpoint.",
    )
    parser.add_argument(
        "--n-mels",
        type=int,
        default=LEGACY_NUM_MELS,
        help="Number of mel bins expected by the model.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=DEFAULT_MAX_FRAMES,
        help="Dummy input frame length used during export.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Dummy batch size used during export.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version.",
    )
    parser.add_argument(
        "--normalize-output",
        action="store_true",
        help="Export L2-normalized embeddings.",
    )
    parser.add_argument(
        "--fixed-frames",
        action="store_true",
        help="Export with a fixed frame dimension instead of a dynamic time axis.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if args.output:
        output_path = Path(args.output)
    elif args.fixed_frames:
        output_path = Path("checkpoints/resnet_v7_best_fixed.onnx")
    else:
        output_path = Path("checkpoints/resnet_v7_best.onnx")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    embedding_dim = args.embedding_dim or None
    model, checkpoint = load_model_for_export(
        str(checkpoint_path),
        embedding_dim=embedding_dim,
    )
    exported_model = EmbeddingExportWrapper(
        model=model,
        normalize_output=args.normalize_output,
    )
    exported_model.eval()

    if embedding_dim is None:
        state_dict = extract_state_dict(checkpoint)
        embedding_dim = infer_embedding_dim(state_dict)

    n_mels = int(getattr(model, "n_mels", args.n_mels))

    dummy_input = torch.randn(
        args.batch_size,
        1,
        n_mels,
        args.max_frames,
        dtype=torch.float32,
    )

    dynamic_axes = {
        "input": {0: "batch"},
        "embedding": {0: "batch"},
    }
    if not args.fixed_frames:
        dynamic_axes["input"][3] = "frames"

    with torch.no_grad():
        torch.onnx.export(
            exported_model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["embedding"],
            dynamic_axes=dynamic_axes,
        )

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_path}")
    print(f"Input shape: [batch, 1, {n_mels}, {args.max_frames}]")
    print(f"Dynamic frames: {'no' if args.fixed_frames else 'yes'}")
    print(f"Embedding dim: {embedding_dim}")
    print(f"Normalize output: {'yes' if args.normalize_output else 'no'}")
    print("Export finished.")
    print("Note: this ONNX model expects log-mel features, not raw waveform.")


if __name__ == "__main__":
    main()
