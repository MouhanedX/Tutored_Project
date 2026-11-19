"""Vitis AI 3.0 quantization/export entry point for the GTSRB classifier."""
import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model import TrafficSignCNN

MEAN = (0.3403, 0.3121, 0.3214)
STD = (0.2724, 0.2608, 0.2669)
IMG_SIZE = 48


class SignRows(Dataset):
    def __init__(self, root, rows):
        self.root = Path(root)
        self.rows = rows
        self.transform = T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = Image.open(self.root / row["Path"]).convert("RGB")
        image = image.crop((
            int(row["Roi.X1"]), int(row["Roi.Y1"]),
            int(row["Roi.X2"]) + 1, int(row["Roi.Y2"]) + 1,
        ))
        return self.transform(image), int(row["ClassId"])


def read_rows(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def build_deploy_model(path):
    model = TrafficSignCNN(num_classes=43)
    model.pool = nn.AvgPool2d(2)
    state = {key: torch.from_numpy(value) for key, value in np.load(path).items()}
    model.load_state_dict(state)
    return model.eval()


@torch.no_grad()
def accuracy(model, loader):
    correct = total = 0
    for images, labels in loader:
        correct += int((model(images).argmax(1) == labels).sum())
        total += labels.numel()
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("float", "calib", "test"), required=True)
    parser.add_argument("--deploy", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=Path("archive"))
    parser.add_argument("--weights", type=Path,
                        default=Path("outputs/deploy/gtsrb_float.npz"))
    parser.add_argument("--out", type=Path,
                        default=Path("outputs/deploy/quantize_result"))
    parser.add_argument("--calib-images", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    model = build_deploy_model(args.weights)
    test_rows = read_rows(args.data_dir / "Test.csv")
    train_rows = read_rows(args.data_dir / "Train.csv")
    if args.calib_images > len(train_rows):
        raise ValueError("--calib-images exceeds training rows")
    calib_rows = random.Random(0).sample(train_rows, args.calib_images)

    def loader(rows, batch_size):
        return DataLoader(
            SignRows(args.data_dir, rows),
            batch_size=batch_size,
            num_workers=args.num_workers,
        )

    if args.mode == "float":
        print("float deploy accuracy: {:.2f}%".format(
            accuracy(model, loader(test_rows, args.batch_size)) * 100
        ))
        return

    from pytorch_nndct.apis import torch_quantizer

    quantizer = torch_quantizer(
        args.mode,
        model,
        (torch.randn(1, 3, IMG_SIZE, IMG_SIZE),),
        output_dir=str(args.out),
        device=torch.device("cpu"),
    )
    quantized_model = quantizer.quant_model
    if args.mode == "calib":
        score = accuracy(quantized_model, loader(calib_rows, args.batch_size))
        quantizer.export_quant_config()
        print("calibrated {} images, accuracy {:.2f}%".format(
            len(calib_rows), score * 100
        ))
    elif args.deploy:
        accuracy(quantized_model, loader(test_rows[:1], 1))
        quantizer.export_xmodel(output_dir=str(args.out), deploy_check=False)
        print("exported {}".format(args.out / "TrafficSignCNN_int.xmodel"))
    else:
        print("INT8 simulated accuracy: {:.2f}%".format(
            accuracy(quantized_model, loader(test_rows, args.batch_size)) * 100
        ))


if __name__ == "__main__":
    main()
