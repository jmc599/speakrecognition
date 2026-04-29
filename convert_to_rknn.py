import argparse
from pathlib import Path

from rknn.api import RKNN

from utils.legacy_feature import STATIC_LEGACY_INPUT_SHAPE_TEXT


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a fixed-shape ONNX speaker model to RKNN."
    )
    parser.add_argument(
        "--onnx-model",
        type=str,
        default="checkpoints/resnet_v7_vox2ft_s1_latest_fixed.onnx",
        help="Path to the input ONNX model.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional output RKNN path. Defaults to a name derived from --onnx-model.",
    )
    parser.add_argument(
        "--input-name",
        type=str,
        default="input",
        help="ONNX input tensor name.",
    )
    parser.add_argument(
        "--input-shape",
        type=str,
        default=STATIC_LEGACY_INPUT_SHAPE_TEXT,
        help="Static input shape as comma-separated integers.",
    )
    parser.add_argument(
        "--target-platform",
        type=str,
        default="rk3588",
        help="RKNN target platform.",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Enable quantization during RKNN build.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Quantization dataset file path. Required when --quantize is enabled.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce RKNN toolkit logging noise.",
    )
    return parser.parse_args()


def parse_shape(text):
    values = [item.strip() for item in text.split(",") if item.strip()]
    shape = [int(item) for item in values]
    if len(shape) != 4:
        raise ValueError(
            f"Expected 4 integers in --input-shape, got {text!r}"
        )
    return shape


def default_output_path(onnx_model):
    stem = onnx_model.stem
    if stem.endswith("_nonorm_op12"):
        target_stem = stem[: -len("_nonorm_op12")] + "_rt160_nonorm"
    elif stem.endswith("_op12"):
        target_stem = stem[: -len("_op12")] + "_rt160"
    else:
        target_stem = stem + "_rt160"
    return onnx_model.with_name(f"{target_stem}.rknn")


def main():
    args = parse_args()
    onnx_model = Path(args.onnx_model)
    output_path = Path(args.output) if args.output else default_output_path(onnx_model)

    if not onnx_model.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_model}")
    if args.quantize and not args.dataset:
        raise ValueError("--dataset is required when --quantize is enabled.")

    input_shape = parse_shape(args.input_shape)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rknn = RKNN(verbose=not args.quiet)
    try:
        print("--> Config model")
        ret = rknn.config(target_platform=args.target_platform)
        if ret != 0:
            raise RuntimeError(f"config failed: {ret}")
        print("done")

        print("--> Loading ONNX model")
        ret = rknn.load_onnx(
            model=str(onnx_model),
            inputs=[args.input_name],
            input_size_list=[input_shape],
        )
        if ret != 0:
            raise RuntimeError(f"load_onnx failed: {ret}")
        print("done")

        print("--> Building model")
        build_kwargs = {"do_quantization": args.quantize}
        if args.quantize:
            build_kwargs["dataset"] = args.dataset
        ret = rknn.build(**build_kwargs)
        if ret != 0:
            raise RuntimeError(f"build failed: {ret}")
        print("done")

        print("--> Export RKNN model")
        ret = rknn.export_rknn(str(output_path))
        if ret != 0:
            raise RuntimeError(f"export_rknn failed: {ret}")
        print("done")
    finally:
        rknn.release()

    print(f"ONNX model: {onnx_model}")
    print(f"Output: {output_path}")
    print(f"Input name: {args.input_name}")
    print(f"Input shape: {input_shape}")
    print(f"Target platform: {args.target_platform}")
    print(f"Quantized: {'yes' if args.quantize else 'no'}")


if __name__ == "__main__":
    main()
