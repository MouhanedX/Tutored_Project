"""Benchmark the compiled GTSRB classifier on the Ultra96-V2 DPU."""
import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import vart
import xir
from pynq_dpu import DpuOverlay

MEAN = np.array([0.3403, 0.3121, 0.3214], dtype=np.float32)
STD = np.array([0.2724, 0.2608, 0.2669], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("xmodel", type=Path)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    overlay = DpuOverlay(
        "/usr/local/share/pynq-venv/lib/python3.10/site-packages/pynq_dpu/dpu.bit"
    )
    graph = xir.Graph.deserialize(str(args.xmodel))
    dpu_subgraphs = [
        subgraph
        for subgraph in graph.get_root_subgraph().toposort_child_subgraph()
        if subgraph.has_attr("device")
        and subgraph.get_attr("device").upper() == "DPU"
    ]
    if len(dpu_subgraphs) != 1:
        raise RuntimeError("expected one DPU subgraph")

    runner = vart.Runner.create_runner(dpu_subgraphs[0], "run")
    input_tensor = runner.get_input_tensors()[0]
    output_tensor = runner.get_output_tensors()[0]
    _, height, width, _ = input_tensor.dims
    input_scale = 2 ** input_tensor.get_attr("fix_point")
    input_buffer = np.empty(tuple(input_tensor.dims), dtype=np.int8)
    output_buffer = np.empty(tuple(output_tensor.dims), dtype=np.int8)

    with args.csv.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if args.limit:
        rows = rows[:args.limit]

    correct = 0
    preprocess_seconds = 0.0
    dpu_seconds = 0.0
    for index, row in enumerate(rows, 1):
        start = time.perf_counter()
        image = cv2.imread(str(args.csv.parent / row["Path"]))
        if image is None:
            raise FileNotFoundError(args.csv.parent / row["Path"])
        x1, y1, x2, y2 = (
            int(row[key]) for key in ("Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2")
        )
        crop = image[y1:y2 + 1, x1:x2 + 1]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        interpolation = cv2.INTER_AREA if rgb.shape[0] > height else cv2.INTER_LINEAR
        rgb = cv2.resize(rgb, (width, height), interpolation=interpolation)
        normalized = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
        input_buffer[0] = np.clip(
            np.round(normalized * input_scale), -128, 127
        ).astype(np.int8)
        middle = time.perf_counter()

        job = runner.execute_async([input_buffer], [output_buffer])
        runner.wait(job)
        predicted = int(output_buffer.reshape(-1).argmax())
        end = time.perf_counter()

        correct += predicted == int(row["ClassId"])
        preprocess_seconds += middle - start
        dpu_seconds += end - middle
        if index % 1000 == 0:
            print("processed {}/{}".format(index, len(rows)), flush=True)

    count = len(rows)
    if not count:
        raise ValueError("CSV contains no rows")
    print("accuracy {}/{} = {:.2f}%".format(
        correct, count, correct / count * 100
    ))
    print("preprocess {:.2f} ms, DPU {:.2f} ms, total {:.2f} ms".format(
        preprocess_seconds / count * 1000,
        dpu_seconds / count * 1000,
        (preprocess_seconds + dpu_seconds) / count * 1000,
    ))


if __name__ == "__main__":
    main()
