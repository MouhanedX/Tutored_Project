"""Export the trained GTSRB checkpoint as NumPy arrays for Vitis AI."""
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import load_model


def main():
    model, checkpoint = load_model(
        Path("outputs/best_model.pt"), torch.device("cpu")
    )
    output = Path("outputs/deploy/gtsrb_float.npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **{
        key: value.cpu().numpy()
        for key, value in model.state_dict().items()
    })
    print(
        "epoch {}, val acc {:.2f}% -> {}".format(
            checkpoint["epoch"], checkpoint["val_acc"] * 100, output
        )
    )


if __name__ == "__main__":
    main()
