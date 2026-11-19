"""Run GTSDB detection followed by 43-class GTSRB classification on the Ultra96-V2."""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import vart
import xir
from pynq_dpu import DpuOverlay

sys.path.insert(0, str(Path(__file__).resolve().parent))
import yolo_dpu_benchmark as detector


CLASS_NAMES = [
    "Speed limit 20", "Speed limit 30", "Speed limit 50", "Speed limit 60",
    "Speed limit 70", "Speed limit 80", "End speed limit 80", "Speed limit 100",
    "Speed limit 120", "No passing", "No passing >3.5t", "Right-of-way at intersection",
    "Priority road", "Yield", "Stop", "No vehicles",
    "Vehicles >3.5t prohibited", "No entry", "General caution", "Dangerous curve left",
    "Dangerous curve right", "Double curve", "Bumpy road", "Slippery road",
    "Road narrows right", "Road work", "Traffic signals", "Pedestrians",
    "Children crossing", "Bicycles crossing", "Beware ice/snow", "Wild animals crossing",
    "End all limits", "Turn right ahead", "Turn left ahead", "Ahead only",
    "Straight or right", "Straight or left", "Keep right", "Keep left",
    "Roundabout mandatory", "End no passing", "End no passing >3.5t",
]

CLASSIFIER_MEAN = np.array([0.3403, 0.3121, 0.3214], dtype=np.float32)
CLASSIFIER_STD = np.array([0.2724, 0.2608, 0.2669], dtype=np.float32)


def create_detector(model_path):
    graph = xir.Graph.deserialize(str(model_path))
    subgraphs = graph.get_root_subgraph().toposort_child_subgraph()
    cpu_subgraphs = [
        subgraph for subgraph in subgraphs
        if subgraph.has_attr("device")
        and subgraph.get_attr("device").upper() == "CPU"
    ]
    dpu_subgraphs = [
        subgraph for subgraph in subgraphs
        if subgraph.has_attr("device")
        and subgraph.get_attr("device").upper() == "DPU"
    ]
    user_subgraphs = [
        subgraph for subgraph in subgraphs
        if subgraph.has_attr("device")
        and subgraph.get_attr("device").upper() == "USER"
    ]
    if not dpu_subgraphs:
        raise RuntimeError("detector contains no DPU subgraphs")
    if len(user_subgraphs) != 1:
        raise RuntimeError("detector must contain exactly one USER input subgraph")

    input_tensor = list(user_subgraphs[0].get_output_tensors())[0]
    output_tensors = list(graph.get_root_subgraph().get_output_tensors())
    if len(output_tensors) != 3:
        raise RuntimeError("detector must contain three output tensors")
    if not input_tensor.has_attr("fix_point"):
        raise RuntimeError("detector input tensor is missing fix_point")

    runners = {}
    for index, subgraph in enumerate(dpu_subgraphs, 1):
        runners[subgraph.get_name()] = vart.Runner.create_runner(subgraph, "run")
        if index % 10 == 0:
            print(
                "created {}/{} detector DPU runners".format(
                    index, len(dpu_subgraphs)
                ),
                flush=True,
            )
    return (
        graph, subgraphs, cpu_subgraphs, dpu_subgraphs, input_tensor,
        output_tensors, runners,
    )


def detector_run(image, detector_state, confidence, iou, max_detections):
    (
        graph, subgraphs, cpu_subgraphs, dpu_subgraphs,
        input_tensor, output_tensors, runners,
    ) = detector_state
    input_scale = 2 ** input_tensor.get_attr("fix_point")
    _, input_height, input_width, input_channels = input_tensor.dims
    if input_height != input_width or input_channels != 3:
        raise RuntimeError("unexpected detector input dimensions")

    padded, ratio, pad_left, pad_top = detector.letterbox(image, input_width)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    quantized = np.clip(
        np.rint(rgb.astype(np.float32) / 255.0 * input_scale),
        -128,
        127,
    ).astype(np.int8)
    tensors = {input_tensor.name: np.ascontiguousarray(quantized[None, ...])}

    for index, subgraph in enumerate(subgraphs, 1):
        device = subgraph.get_attr("device").upper()
        if device == "USER":
            continue
        if device == "CPU":
            detector.execute_cpu_partition(subgraph, tensors)
        elif device == "DPU":
            detector.execute_dpu_partition(
                subgraph,
                runners[subgraph.get_name()],
                tensors,
            )
        else:
            raise RuntimeError("unsupported detector partition device {}".format(device))
        if index % 10 == 0:
            print("executed {}/{} detector partitions".format(index, len(subgraphs)), flush=True)

    output_fix_points = [
        tensor.get_attr("fix_point") if tensor.has_attr("fix_point") else None
        for tensor in output_tensors
    ]
    return detector.decode_outputs(
        [tensors[tensor.name] for tensor in output_tensors],
        output_fix_points,
        input_width,
        ratio,
        pad_left,
        pad_top,
        image.shape,
        confidence,
        iou,
        max_detections,
    )


def create_classifier(model_path):
    graph = xir.Graph.deserialize(str(model_path))
    dpu_subgraphs = [
        subgraph for subgraph in graph.get_root_subgraph().toposort_child_subgraph()
        if subgraph.has_attr("device")
        and subgraph.get_attr("device").upper() == "DPU"
    ]
    if len(dpu_subgraphs) != 1:
        raise RuntimeError("classifier must contain exactly one DPU subgraph")

    runner = vart.Runner.create_runner(dpu_subgraphs[0], "run")
    input_tensor = runner.get_input_tensors()[0]
    output_tensor = runner.get_output_tensors()[0]
    if not input_tensor.has_attr("fix_point"):
        raise RuntimeError("classifier input tensor is missing fix_point")
    _, height, width, channels = input_tensor.dims
    if height != 48 or width != 48 or channels != 3:
        raise RuntimeError("unexpected classifier input dimensions")
    if tuple(output_tensor.dims) != (1, 43):
        raise RuntimeError("unexpected classifier output dimensions")

    input_buffer = np.empty(tuple(input_tensor.dims), dtype=np.int8)
    output_buffer = np.empty(tuple(output_tensor.dims), dtype=np.int8)
    input_scale = 2 ** input_tensor.get_attr("fix_point")
    return (
        graph, dpu_subgraphs, runner, input_tensor, output_tensor,
        input_buffer, output_buffer, input_scale,
    )


def classify_crop(image, classifier_state):
    (
        graph, subgraphs, runner, input_tensor, output_tensor,
        input_buffer, output_buffer, input_scale,
    ) = classifier_state
    _, height, width, _ = input_tensor.dims
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    normalized = (rgb.astype(np.float32) / 255.0 - CLASSIFIER_MEAN) / CLASSIFIER_STD
    input_buffer[0] = np.clip(
        np.rint(normalized * input_scale),
        -128,
        127,
    ).astype(np.int8)
    print("classifier DPU inference starting", flush=True)
    job = runner.execute_async([input_buffer], [output_buffer])
    runner.wait(job)
    print("classifier DPU inference finished", flush=True)
    logits = output_buffer.reshape(-1).astype(np.float32)
    class_id = int(logits.argmax())
    return class_id, CLASS_NAMES[class_id]


def draw_result(image, detections, classifier_state):
    annotated = image.copy()
    for index, (box, _detector_score) in enumerate(detections, 1):
        x1, y1, x2, y2 = [int(round(value)) for value in box]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = image[y1:y2, x1:x2]
        print(
            "classifying crop {}/{}".format(index, len(detections)),
            flush=True,
        )
        class_id, class_name = classify_crop(crop, classifier_state)
        print(
            "crop {} classified as {} ({})".format(index, class_name, class_id),
            flush=True,
        )
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = "{} ({})".format(class_name, class_id)
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        )
        image_height, image_width = image.shape[:2]
        label_width = text_width + 6
        label_height = text_height + baseline + 4
        label_x = min(max(0, x1), max(0, image_width - label_width))
        if y1 >= label_height:
            label_y = y1 - label_height
        else:
            label_y = y2
        label_y = min(max(0, label_y), max(0, image_height - label_height))
        cv2.rectangle(
            annotated,
            (label_x, label_y),
            (
                min(image_width - 1, label_x + label_width),
                min(image_height - 1, label_y + label_height),
            ),
            (0, 255, 0),
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (label_x + 3, min(image_height - 1, label_y + text_height + 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    return annotated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("annotated.jpg"))
    parser.add_argument("--confidence", type=float, default=0.01)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    args = parser.parse_args()

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(args.image)

    overlay = DpuOverlay(
        "/usr/local/share/pynq-venv/lib/python3.10/site-packages/pynq_dpu/dpu.bit"
    )
    print("DPU overlay loaded", flush=True)

    start = time.perf_counter()
    detector_state = create_detector(args.detector)
    print("detector runners ready", flush=True)
    classifier_state = create_classifier(args.classifier)
    print("classifier runner ready", flush=True)
    setup_seconds = time.perf_counter() - start

    start = time.perf_counter()
    print("running detector", flush=True)
    detections = detector_run(
        image,
        detector_state,
        args.confidence,
        args.iou,
        args.max_det,
    )
    detect_seconds = time.perf_counter() - start
    print("detector found {} signs".format(len(detections)), flush=True)

    start = time.perf_counter()
    print("classifying detected crops", flush=True)
    annotated = draw_result(image, detections, classifier_state)
    classify_seconds = time.perf_counter() - start

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), annotated):
        raise RuntimeError("could not write {}".format(args.output))

    print("detections: {}".format(len(detections)))
    print("detector setup: {:.2f} ms".format(setup_seconds * 1000))
    print("detection: {:.2f} ms".format(detect_seconds * 1000))
    print("classification: {:.2f} ms".format(classify_seconds * 1000))
    print("total inference: {:.2f} ms".format(
        (detect_seconds + classify_seconds) * 1000
    ))
    print("output: {}".format(args.output))


if __name__ == "__main__":
    main()
