"""Vitis AI 2.5 quantization entry point for the raw YOLOv8 detector head."""
from __future__ import print_function

import argparse
from pathlib import Path

import cv2
import torch
from torch import nn
from ultralytics import YOLO
from ultralytics.yolo.data.dataloaders.v5augmentations import letterbox


class RawYOLOv8(nn.Module):
    """Run YOLOv8 through convolutional prediction maps only."""

    def __init__(self, detection_model):
        super().__init__()
        self.detection_model = detection_model
        self.detect = detection_model.model[-1]

    def forward(self, image):
        saved = []
        x = image
        head_features = None
        for layer in self.detection_model.model:
            if layer.f != -1:
                x = (
                    saved[layer.f]
                    if isinstance(layer.f, int)
                    else [x if index == -1 else saved[index] for index in layer.f]
                )
            if layer is self.detect:
                head_features = x
                break
            x = layer(x)
            saved.append(x if layer.i in self.detection_model.save else None)

        if head_features is None:
            raise RuntimeError("YOLOv8 detection head was not reached")
        return tuple(
            torch.cat(
                (
                    self.detect.cv2[index](head_features[index]),
                    self.detect.cv3[index](head_features[index]),
                ),
                dim=1,
            )
            for index in range(self.detect.nl)
        )


class DecomposedSiLU(nn.Module):
    def forward(self, values):
        return values * torch.sigmoid(values)


def replace_silu(module):
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.SiLU):
            setattr(module, name, DecomposedSiLU())
            replaced += 1
        else:
            replaced += replace_silu(child)
    return replaced


def image_batch(paths, image_size):
    images = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("Unable to read calibration image {}".format(path))
        image = letterbox(
            image,
            (image_size, image_size),
            auto=False,
            scaleup=False,
        )[0]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(torch.from_numpy(image.transpose(2, 0, 1)).float().div_(255.0))
    return torch.stack(images)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("calib", "test", "deploy"), required=True)
    parser.add_argument("--calib-images", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    model = YOLO(str(args.checkpoint)).model.eval()
    replaced = replace_silu(model)
    print("decomposed {} SiLU activations".format(replaced))
    raw_model = RawYOLOv8(model).eval()
    image_paths = sorted(
        path for path in args.images.iterdir() if path.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if not image_paths:
        raise ValueError("No calibration images found in {}".format(args.images))
    if args.mode == "calib":
        image_paths = image_paths[: args.calib_images]
    elif args.mode == "deploy":
        image_paths = image_paths[:1]

    from pytorch_nndct.apis import torch_quantizer

    example = image_batch(image_paths[:1], args.imgsz)
    quant_mode = "test" if args.mode == "deploy" else args.mode
    quantizer = torch_quantizer(
        quant_mode,
        raw_model,
        (example,),
        output_dir=str(args.output),
        device=torch.device("cpu"),
    )
    quantized_model = quantizer.quant_model
    with torch.no_grad():
        for start in range(0, len(image_paths), args.batch_size):
            batch = image_batch(image_paths[start : start + args.batch_size], args.imgsz)
            quantized_model(batch)
            if args.mode == "calib" and (start + args.batch_size) % 25 == 0:
                print("calibrated {}/{}".format(min(start + args.batch_size, len(image_paths)), len(image_paths)))

    if args.mode == "calib":
        quantizer.export_quant_config()
        print("calibration complete: {} images".format(len(image_paths)))
    elif args.mode == "deploy":
        quantizer.export_xmodel(output_dir=str(args.output), deploy_check=False)
        print("xmodel exported to {}".format(args.output))
    else:
        print("quantized test complete: {} images".format(len(image_paths)))


if __name__ == "__main__":
    main()
