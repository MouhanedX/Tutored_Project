"""Prepare a single-class YOLO dataset from the existing GTSDB tree.

The source labels contain GTSDB sign categories. The first pipeline only needs
to detect a sign; the separate GTSRB model performs classification. Source
images and labels are copied, while every valid source box is rewritten as
class 0 ("sign").
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


def read_label(path: Path) -> list[str]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) != 5:
            raise ValueError("{} has invalid row: {!r}".format(path, line))
        values = [float(value) for value in fields[1:]]
        if not all(0 <= value <= 1 for value in values):
            raise ValueError("{} has non-normalized row: {!r}".format(path, line))
        rows.append("0 " + " ".join(fields[1:]))
    return rows


def copy_split(source: Path, destination: Path, names: list[str]) -> None:
    image_dir = destination / "images"
    label_dir = destination / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        source_image = source / "images" / name
        if not source_image.exists():
            raise FileNotFoundError(source_image)
        shutil.copy2(source_image, image_dir / name)
        rows = read_label(source / "labels" / (Path(name).stem + ".txt"))
        if rows:
            (label_dir / (Path(name).stem + ".txt")).write_text("\n".join(rows) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("GTSDB"))
    parser.add_argument("--output", type=Path, default=Path("data/gtsdb_detector"))
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.val_fraction < 1:
        parser.error("--val-fraction must be between 0 and 1")

    train_images = sorted(
        p.name for p in (args.source / "Train" / "images").iterdir()
        if p.is_file()
    )
    rng = random.Random(args.seed)
    rng.shuffle(train_images)
    n_val = max(1, round(len(train_images) * args.val_fraction))
    val_images = sorted(train_images[:n_val])
    train_images = sorted(train_images[n_val:])
    test_images = sorted(
        p.name for p in (args.source / "Test" / "images").iterdir()
        if p.is_file()
    )

    if args.output.exists():
        shutil.rmtree(args.output)
    copy_split(args.source / "Train", args.output / "train", train_images)
    copy_split(args.source / "Train", args.output / "val", val_images)
    copy_split(args.source / "Test", args.output / "test", test_images)
    (args.output / "gtsdb_sign.yaml").write_text(
        "path: {}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        "names:\n"
        "  0: sign\n".format(args.output.resolve().as_posix())
    )
    print("prepared {} train, {} val, {} test images".format(
        len(train_images), len(val_images), len(test_images)
    ))


if __name__ == "__main__":
    main()
