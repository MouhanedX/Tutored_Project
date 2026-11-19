"""Fine-tune a ReLU/C3 YOLOv8n detector for Vitis AI DPU compatibility."""
from __future__ import print_function

import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("yolo/yolov8n_dpu.yaml"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("../runs/detect/gtsdb_yolov8n_dpu_c3/weights/best.pt"),
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/gtsdb_detector/gtsdb_sign.yaml"),
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--project", type=Path, default=Path("../runs/detect"))
    parser.add_argument("--name", default="gtsdb_yolov8n_dpu_relu")
    args = parser.parse_args()

    original_load = torch.load

    def trusted_checkpoint_load(*load_args, **load_kwargs):
        load_kwargs.setdefault("weights_only", False)
        return original_load(*load_args, **load_kwargs)

    torch.load = trusted_checkpoint_load

    from ultralytics import YOLO
    from ultralytics.nn.tasks import intersect_dicts

    checkpoint = torch.load(str(args.checkpoint), map_location="cpu")
    source_model = checkpoint.get("ema") or checkpoint["model"]
    model = YOLO(str(args.config))
    transferable = intersect_dicts(
        source_model.float().state_dict(),
        model.model.state_dict(),
    )
    if len(transferable) != len(model.model.state_dict()):
        raise RuntimeError(
            "only {} of {} model tensors can be transferred".format(
                len(transferable), len(model.model.state_dict())
            )
        )
    model.model.load_state_dict(transferable, strict=False)
    print("transferred {} checkpoint tensors".format(len(transferable)))

    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=0,
        device="cpu",
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        patience=12,
        lr0=0.001,
        seed=0,
        deterministic=True,
    )


if __name__ == "__main__":
    main()
