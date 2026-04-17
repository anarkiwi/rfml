"""
Export a trained YOLO spectrogram model to a TorchServe MAR file.

Usage
-----
python rfml/export_spec_model.py \
    --model_name my_spec_model \
    --pt_file runs/detect/train/weights/best.pt \
    --index_to_name index_to_name.json

The resulting MAR can be served with:
  torchserve --start --model-store models/ --models my_spec_model.mar

Send predictions with:
  curl --header "Content-Type:image/png" --data-binary @spectrogram.png \
      http://localhost:8080/predictions/my_spec_model
"""

import argparse
import os
from pathlib import Path

from rfml.export_model import export_model


def argument_parser():
    parser = argparse.ArgumentParser(
        description="Package a YOLO spectrogram model into a TorchServe MAR file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model (used for the output .mar filename).",
    )
    parser.add_argument(
        "--pt_file",
        type=str,
        required=True,
        help="Path to the trained YOLO .pt weights file.",
    )
    parser.add_argument(
        "--index_to_name",
        type=str,
        required=True,
        help="Path to JSON file mapping label index to class name.",
    )
    parser.add_argument(
        "--custom_handler",
        type=str,
        default="custom_handlers/spectrogram_custom_handler.py",
        help="Path to the TorchServe custom handler.",
    )
    parser.add_argument(
        "--export_path",
        type=str,
        default="models/",
        help="Directory to write the MAR file.",
    )
    return parser


def main():
    args = argument_parser().parse_args()

    if not os.path.isfile(args.pt_file):
        raise FileNotFoundError(f"Model weights not found: {args.pt_file}")
    if not os.path.isfile(args.index_to_name):
        raise FileNotFoundError(f"index_to_name file not found: {args.index_to_name}")

    os.makedirs(args.export_path, exist_ok=True)

    print(
        f"\nExporting MAR:\n"
        f"  model_name  : {args.model_name}\n"
        f"  pt_file     : {args.pt_file}\n"
        f"  handler     : {args.custom_handler}\n"
        f"  extra_files : {args.index_to_name}\n"
        f"  export_path : {args.export_path}\n"
    )

    export_model(
        model_name=args.model_name,
        torchscript_file=args.pt_file,
        custom_handler=args.custom_handler,
        index_to_name=args.index_to_name,
        export_path=args.export_path,
    )

    print(f"\nMAR file written to: {Path(args.export_path, args.model_name + '.mar')}")


if __name__ == "__main__":
    main()
