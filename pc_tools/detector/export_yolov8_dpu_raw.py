"""Export the YOLOv8 convolutional heads without box decoding or NMS.

The exported network keeps the trained YOLOv8 backbone, neck, and prediction
convolutions.  The ARM CPU performs DFL decoding, anchor conversion, and NMS
after the DPU returns the raw prediction maps.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from ultralytics import YOLO


class RawYOLOv8(nn.Module):
    """Run a YOLOv8 model through its raw convolutional prediction heads."""

    def __init__(self, detection_model: nn.Module) -> None:
        super().__init__()
        self.detection_model = detection_model
        self.detect = detection_model.model[-1]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features: list[torch.Tensor | None] = []
        x = image
        head_features: list[torch.Tensor] | None = None
        for layer in self.detection_model.model:
            if layer.f != -1:
                x = (
                    features[layer.f]
                    if isinstance(layer.f, int)
                    else [x if index == -1 else features[index] for index in layer.f]
                )
            if layer is self.detect:
                if not isinstance(x, list):
                    raise RuntimeError("YOLOv8 detection head input was not a feature list")
                head_features = x
                break
            x = layer(x)
            features.append(x if layer.i in self.detection_model.save else None)

        if head_features is None:
            raise RuntimeError("YOLOv8 detection head was not reached")

        boxes = torch.cat(
            [
                self.detect.cv2[index](head_features[index]).view(
                    image.shape[0], 4 * self.detect.reg_max, -1
                )
                for index in range(self.detect.nl)
            ],
            dim=2,
        )
        scores = torch.cat(
            [
                self.detect.cv3[index](head_features[index]).view(
                    image.shape[0], self.detect.nc, -1
                )
                for index in range(self.detect.nl)
            ],
            dim=2,
        )
        return torch.cat((boxes, scores), dim=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("yolov8_raw.pt"))
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--torchscript", type=Path)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    loaded = YOLO(str(args.checkpoint))
    detection_model = loaded.model
    detection_model.eval()
    raw_model = RawYOLOv8(detection_model).eval()

    example = torch.zeros(1, 3, args.imgsz, args.imgsz)
    with torch.inference_mode():
        raw_output = raw_model(example)
    expected_channels = detection_model.model[-1].reg_max * 4 + detection_model.model[-1].nc
    if raw_output.shape != (1, expected_channels, 8400):
        raise RuntimeError(
            f"unexpected raw output shape {tuple(raw_output.shape)}; "
            f"expected (1, {expected_channels}, 8400)"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": raw_model,
            "imgsz": args.imgsz,
            "raw_shape": tuple(raw_output.shape),
        },
        args.output,
    )
    if args.onnx is not None:
        args.onnx.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(
            raw_model,
            example,
            args.onnx,
            input_names=["images"],
            output_names=["raw_predictions"],
            opset_version=13,
            do_constant_folding=True,
        )
        print(f"saved {args.onnx}")
    if args.torchscript is not None:
        args.torchscript.parent.mkdir(parents=True, exist_ok=True)
        traced = torch.jit.trace(raw_model, example, strict=False)
        traced.save(str(args.torchscript))
        print(f"saved {args.torchscript}")
    print(f"saved {args.output}")
    print(f"input: {(1, 3, args.imgsz, args.imgsz)}")
    print(f"raw output: {tuple(raw_output.shape)}")


if __name__ == "__main__":
    main()
