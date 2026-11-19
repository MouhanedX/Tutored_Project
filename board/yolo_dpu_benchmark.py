"""Benchmark the compiled YOLO graph with VART DPU and NumPy CPU partitions."""
import argparse
from collections import Counter
import time
from pathlib import Path

import cv2
import numpy as np
import vart
import xir
from pynq_dpu import DpuOverlay


def letterbox(bgr, size):
    height, width = bgr.shape[:2]
    ratio = min(size / height, size / width, 1.0)
    resized_width = int(round(width * ratio))
    resized_height = int(round(height * ratio))
    resized = cv2.resize(
        bgr,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    pad_width = size - resized_width
    pad_height = size - resized_height
    left = int(round(pad_width / 2 - 0.1))
    right = pad_width - left
    top = int(round(pad_height / 2 - 0.1))
    bottom = pad_height - top
    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return padded, ratio, left, top


def sigmoid(values):
    values = np.clip(values, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-values))


def tensor_numpy_dtype(tensor):
    element_count = tensor.get_element_num()
    if element_count <= 0 or tensor.get_data_size() % element_count:
        raise RuntimeError("invalid tensor storage size for {}".format(tensor.name))
    bytes_per_element = tensor.get_data_size() // element_count
    if bytes_per_element == 1:
        return np.int8
    if bytes_per_element == 4:
        return np.float32
    raise RuntimeError(
        "unsupported tensor element size {} bytes for {}".format(
            bytes_per_element, tensor.name
        )
    )


def execute_cpu_partition(subgraph, tensors):
    inputs = list(subgraph.get_input_tensors())
    outputs = list(subgraph.get_output_tensors())
    op_types = [operation.get_type() for operation in subgraph.get_ops()]
    if len(inputs) != 1 or len(outputs) != 1:
        raise RuntimeError(
            "expected one input and output for CPU partition {}, got {} and {}".format(
                subgraph.get_name(), len(inputs), len(outputs)
            )
        )
    input_tensor = inputs[0]
    output_tensor = outputs[0]
    if input_tensor.name not in tensors:
        raise RuntimeError(
            "missing input tensor {} for CPU partition {}".format(
                input_tensor.name, subgraph.get_name()
            )
        )
    values = tensors[input_tensor.name]

    if Counter(op_types) == Counter(("fix2float",)):
        if not input_tensor.has_attr("fix_point"):
            raise RuntimeError(
                "CPU partition input {} is missing fix_point".format(input_tensor.name)
            )
        result = values.astype(np.float32) / (2 ** input_tensor.get_attr("fix_point"))
    elif Counter(op_types) == Counter(("fix2float", "sigmoid", "float2fix")):
        if not input_tensor.has_attr("fix_point") or not output_tensor.has_attr("fix_point"):
            raise RuntimeError(
                "CPU sigmoid partition is missing input/output fix_point"
            )
        real_values = values.astype(np.float32) / (
            2 ** input_tensor.get_attr("fix_point")
        )
        quantized = np.rint(
            sigmoid(real_values) * (2 ** output_tensor.get_attr("fix_point"))
        )
        result = np.clip(quantized, -128, 127).astype(np.int8)
    else:
        raise RuntimeError(
            "unsupported CPU ops in {}: {}".format(
                subgraph.get_name(), ", ".join(op_types)
            )
        )

    if tuple(result.shape) != tuple(output_tensor.dims):
        raise RuntimeError(
            "CPU partition {} produced shape {}, expected {}".format(
                subgraph.get_name(), result.shape, output_tensor.dims
            )
        )
    tensors[output_tensor.name] = np.ascontiguousarray(result)


def execute_dpu_partition(subgraph, runner, tensors):
    input_tensors = runner.get_input_tensors()
    output_tensors = runner.get_output_tensors()
    input_buffers = []
    for tensor in input_tensors:
        if tensor.name not in tensors:
            raise RuntimeError(
                "missing input tensor {} for DPU partition {}".format(
                    tensor.name, subgraph.get_name()
                )
            )
        value = tensors[tensor.name]
        if tuple(value.shape) != tuple(tensor.dims):
            raise RuntimeError(
                "input tensor {} has shape {}, expected {}".format(
                    tensor.name, value.shape, tensor.dims
                )
            )
        input_buffers.append(np.ascontiguousarray(value))

    output_buffers = [
        np.empty(tuple(tensor.dims), dtype=tensor_numpy_dtype(tensor))
        for tensor in output_tensors
    ]
    job = runner.execute_async(input_buffers, output_buffers)
    runner.wait(job)
    for tensor, value in zip(output_tensors, output_buffers):
        tensors[tensor.name] = value


def read_yolo_labels(path, image_shape):
    if not path.is_file():
        return np.zeros((0, 5), dtype=np.float32), False
    height, width = image_shape[:2]
    rows = []
    with path.open() as label_file:
        for line_number, line in enumerate(label_file, 1):
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(
                    "invalid YOLO label at {}:{}: expected 5 columns".format(
                        path, line_number
                    )
                )
            class_id, center_x, center_y, box_width, box_height = map(float, fields)
            if class_id != 0 or not np.all(
                np.isfinite((center_x, center_y, box_width, box_height))
            ):
                raise ValueError(
                    "invalid one-class label at {}:{}".format(path, line_number)
                )
            x1 = (center_x - box_width / 2) * width
            y1 = (center_y - box_height / 2) * height
            x2 = (center_x + box_width / 2) * width
            y2 = (center_y + box_height / 2) * height
            rows.append((class_id, x1, y1, x2, y2))
    return np.asarray(rows, dtype=np.float32).reshape(-1, 5), True


def box_iou(boxes_a, boxes_b):
    if not len(boxes_a) or not len(boxes_b):
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    top_left = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    intersection_size = np.maximum(bottom_right - top_left, 0)
    intersection = intersection_size[..., 0] * intersection_size[..., 1]
    area_a = np.prod(np.maximum(boxes_a[:, 2:] - boxes_a[:, :2], 0), axis=1)
    area_b = np.prod(np.maximum(boxes_b[:, 2:] - boxes_b[:, :2], 0), axis=1)
    return intersection / (area_a[:, None] + area_b[None, :] - intersection + 1e-7)


def match_predictions(detections, labels):
    thresholds = np.linspace(0.5, 0.95, 10)
    correct = np.zeros((len(detections), len(thresholds)), dtype=bool)
    if not len(detections) or not len(labels):
        return correct
    predictions = np.asarray([detection[0] for detection in detections], dtype=np.float32)
    ious = box_iou(labels[:, 1:], predictions)
    for index, threshold in enumerate(thresholds):
        label_indices, prediction_indices = np.where(ious >= threshold)
        if not len(label_indices):
            continue
        matches = np.column_stack(
            (label_indices, prediction_indices, ious[label_indices, prediction_indices])
        )
        if len(matches) > 1:
            matches = matches[np.argsort(-matches[:, 2])]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        correct[matches[:, 1].astype(np.int64), index] = True
    return correct


def smooth(values, fraction=0.1):
    filter_size = round(len(values) * fraction * 2) // 2 + 1
    padding = np.ones(filter_size // 2)
    padded = np.concatenate((padding * values[0], values, padding * values[-1]))
    return np.convolve(padded, np.ones(filter_size) / filter_size, mode="valid")


def compute_ap(recall, precision):
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([1.0], precision, [0.0]))
    precision = np.flip(np.maximum.accumulate(np.flip(precision)))
    recall_grid = np.linspace(0.0, 1.0, 101)
    return np.trapz(np.interp(recall_grid, recall, precision), recall_grid)


def detection_metrics(correct, confidences, predicted_classes, target_classes):
    order = np.argsort(-confidences)
    correct = correct[order]
    confidences = confidences[order]
    predicted_classes = predicted_classes[order]
    classes, target_counts = np.unique(target_classes, return_counts=True)
    if not len(classes):
        raise ValueError("the label set contains no ground-truth signs")
    confidence_grid = np.linspace(0.0, 1.0, 1000)
    ap = np.zeros((len(classes), correct.shape[1]))
    precision_curve = np.zeros((len(classes), len(confidence_grid)))
    recall_curve = np.zeros_like(precision_curve)
    for class_index, class_id in enumerate(classes):
        selected = predicted_classes == class_id
        if not np.any(selected):
            continue
        class_confidences = confidences[selected]
        true_positives = correct[selected].cumsum(axis=0)
        false_positives = (1 - correct[selected]).cumsum(axis=0)
        recall = true_positives / (target_counts[class_index] + 1e-16)
        precision = true_positives / (true_positives + false_positives + 1e-16)
        recall_curve[class_index] = np.interp(
            -confidence_grid, -class_confidences, recall[:, 0], left=0
        )
        precision_curve[class_index] = np.interp(
            -confidence_grid, -class_confidences, precision[:, 0], left=1
        )
        for threshold_index in range(correct.shape[1]):
            ap[class_index, threshold_index] = compute_ap(
                recall[:, threshold_index], precision[:, threshold_index]
            )
    f1 = (
        2 * precision_curve * recall_curve
        / (precision_curve + recall_curve + 1e-16)
    )
    best_index = smooth(f1.mean(axis=0)).argmax()
    return (
        float(precision_curve[:, best_index].mean()),
        float(recall_curve[:, best_index].mean()),
        float(f1[:, best_index].mean()),
        float(confidence_grid[best_index]),
        float(ap[:, 0].mean()),
        float(ap.mean()),
    )


def decode_outputs(outputs, fix_points, image_size, ratio, pad_left, pad_top, original_shape,
                   confidence, iou_threshold, max_detections):
    if len(outputs) != 3 or len(fix_points) != 3:
        raise RuntimeError(
            "expected three YOLO scale outputs, got {} outputs and {} fix_points".format(
                len(outputs), len(fix_points)
            )
        )
    boxes = []
    scores = []
    for output, fix_point in zip(outputs, fix_points):
        if np.issubdtype(output.dtype, np.integer):
            if fix_point is None:
                raise RuntimeError("quantized YOLO output is missing its fix_point")
            prediction = output.astype(np.float32) / (2 ** fix_point)
        else:
            prediction = output.astype(np.float32)
        prediction = np.squeeze(prediction, axis=0)
        if prediction.ndim != 3:
            raise RuntimeError("expected 3D YOLO output, got {}".format(prediction.shape))
        if prediction.shape[-1] == 65:
            pass
        elif prediction.shape[0] == 65:
            prediction = prediction.transpose(1, 2, 0)
        else:
            raise RuntimeError("unexpected YOLO output shape {}".format(prediction.shape))

        height, width, channels = prediction.shape
        if channels != 65:
            raise RuntimeError("expected 65 output channels, got {}".format(channels))
        stride = image_size / height
        grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
        centers_x = (grid_x.astype(np.float32) + 0.5) * stride
        centers_y = (grid_y.astype(np.float32) + 0.5) * stride

        distributions = prediction[..., :64].reshape(height, width, 4, 16)
        distributions -= distributions.max(axis=-1, keepdims=True)
        probabilities = np.exp(distributions)
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        distances = (probabilities * np.arange(16, dtype=np.float32)).sum(axis=-1) * stride
        score_map = sigmoid(prediction[..., 64])
        selected_y, selected_x = np.where(score_map >= confidence)
        if selected_x.size == 0:
            continue
        left = centers_x[selected_y, selected_x] - distances[selected_y, selected_x, 0]
        top = centers_y[selected_y, selected_x] - distances[selected_y, selected_x, 1]
        right = centers_x[selected_y, selected_x] + distances[selected_y, selected_x, 2]
        bottom = centers_y[selected_y, selected_x] + distances[selected_y, selected_x, 3]
        boxes.extend(np.stack((left, top, right, bottom), axis=1).tolist())
        scores.extend(score_map[selected_y, selected_x].tolist())

    if not boxes:
        return []
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    xywh = np.column_stack(
        (
            boxes[:, 0],
            boxes[:, 1],
            boxes[:, 2] - boxes[:, 0],
            boxes[:, 3] - boxes[:, 1],
        )
    )
    kept = cv2.dnn.NMSBoxes(
        xywh.tolist(),
        scores.tolist(),
        confidence,
        iou_threshold,
        top_k=max_detections,
    )
    if len(kept) == 0:
        return []
    kept = np.asarray(kept).reshape(-1)
    boxes = boxes[kept]
    scores = scores[kept]

    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_left) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_top) / ratio
    original_height, original_width = original_shape[:2]
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, original_width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, original_height)
    return [(box, float(score)) for box, score in zip(boxes, scores)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("xmodel", type=Path)
    parser.add_argument("images", type=Path)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--confidence", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    overlay = DpuOverlay(
        "/usr/local/share/pynq-venv/lib/python3.10/site-packages/pynq_dpu/dpu.bit"
    )
    graph = xir.Graph.deserialize(str(args.xmodel))
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
        raise RuntimeError("model contains no DPU subgraphs")
    if len(user_subgraphs) != 1:
        raise RuntimeError(
            "expected one USER input partition, got {}".format(len(user_subgraphs))
        )

    input_tensors = list(user_subgraphs[0].get_output_tensors())
    if len(input_tensors) != 1:
        raise RuntimeError(
            "expected one graph input tensor, got {}".format(len(input_tensors))
        )
    input_tensor = input_tensors[0]
    output_tensors = list(graph.get_root_subgraph().get_output_tensors())
    if len(output_tensors) != 3:
        raise RuntimeError(
            "expected three YOLO graph outputs, got {}".format(len(output_tensors))
        )
    _, input_height, input_width, input_channels = input_tensor.dims
    if input_height != input_width or input_channels != 3:
        raise RuntimeError(
            "expected square NHWC RGB input, got {}".format(input_tensor.dims)
        )
    if not input_tensor.has_attr("fix_point"):
        raise RuntimeError("quantized graph input is missing its fix_point")
    input_scale = 2 ** input_tensor.get_attr("fix_point")
    output_fix_points = [
        tensor.get_attr("fix_point") if tensor.has_attr("fix_point") else None
        for tensor in output_tensors
    ]

    runner_by_subgraph = {}
    for index, subgraph in enumerate(dpu_subgraphs, 1):
        runner_by_subgraph[subgraph.get_name()] = vart.Runner.create_runner(
            subgraph, "run"
        )
        if index % 10 == 0:
            print("created {}/{} DPU runners".format(index, len(dpu_subgraphs)), flush=True)
    print(
        "graph partitions: {} DPU, {} CPU; CPU fallback ops run in NumPy".format(
            len(dpu_subgraphs), len(cpu_subgraphs)
        ),
        flush=True,
    )

    image_paths = sorted(
        path for path in args.images.iterdir()
        if path.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if args.labels is not None and not args.labels.is_dir():
        raise NotADirectoryError(args.labels)
    if args.limit > 0:
        image_paths = image_paths[:args.limit]
    if not image_paths:
        raise ValueError("no image files found in {}".format(args.images))

    preprocessing_seconds = 0.0
    partition_execution_seconds = 0.0
    postprocessing_seconds = 0.0
    detection_count = 0
    correct_stats = []
    confidence_stats = []
    prediction_class_stats = []
    target_class_stats = []
    missing_label_count = 0
    ground_truth_count = 0
    for index, path in enumerate(image_paths):
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)

        start = time.perf_counter()
        padded, ratio, pad_left, pad_top = letterbox(bgr, input_width)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        quantized = np.clip(
            np.rint(rgb.astype(np.float32) / 255.0 * input_scale),
            -128,
            127,
        ).astype(np.int8)
        tensors = {input_tensor.name: np.ascontiguousarray(quantized[None, ...])}
        middle = time.perf_counter()

        for subgraph in subgraphs:
            device = (
                subgraph.get_attr("device").upper()
                if subgraph.has_attr("device")
                else ""
            )
            if device == "USER":
                continue
            if device == "CPU":
                execute_cpu_partition(subgraph, tensors)
            elif device == "DPU":
                execute_dpu_partition(
                    subgraph,
                    runner_by_subgraph[subgraph.get_name()],
                    tensors,
                )
            else:
                raise RuntimeError(
                    "unsupported graph partition device {!r} for {}".format(
                        device, subgraph.get_name()
                    )
                )
        end = time.perf_counter()

        detections = decode_outputs(
            [tensors[tensor.name] for tensor in output_tensors],
            output_fix_points,
            input_width,
            ratio,
            pad_left,
            pad_top,
            bgr.shape,
            args.confidence,
            args.iou,
            args.max_det,
        )
        finish = time.perf_counter()
        if args.labels is not None:
            labels, label_file_exists = read_yolo_labels(
                args.labels / (path.stem + ".txt"), bgr.shape
            )
            if not label_file_exists:
                missing_label_count += 1
            ground_truth_count += len(labels)
            correct_stats.append(match_predictions(detections, labels))
            confidence_stats.append(
                np.asarray([detection[1] for detection in detections], dtype=np.float32)
            )
            prediction_class_stats.append(
                np.zeros(len(detections), dtype=np.float32)
            )
            target_class_stats.append(labels[:, 0])
        if index >= args.warmup:
            detection_count += len(detections)
        preprocessing_seconds += middle - start
        partition_execution_seconds += end - middle
        postprocessing_seconds += finish - end
        if index < args.warmup:
            preprocessing_seconds -= middle - start
            partition_execution_seconds -= end - middle
            postprocessing_seconds -= finish - end
        if (index + 1) % 25 == 0:
            print("processed {}/{}".format(index + 1, len(image_paths)), flush=True)

    measured = max(len(image_paths) - min(args.warmup, len(image_paths)), 1)
    print("images: {}".format(len(image_paths)))
    print("input tensor: {}".format(input_tensor.dims))
    print("output tensors: {}".format([tensor.dims for tensor in output_tensors]))
    print("mean detections/image: {:.2f}".format(detection_count / measured))
    if args.labels is not None:
        correct = np.concatenate(correct_stats, axis=0)
        confidences = np.concatenate(confidence_stats)
        prediction_classes = np.concatenate(prediction_class_stats)
        target_classes = np.concatenate(target_class_stats)
        precision, recall, f1, best_confidence, map50, map5095 = detection_metrics(
            correct, confidences, prediction_classes, target_classes
        )
        print("ground-truth signs: {}".format(ground_truth_count))
        print("images with no label file: {}".format(missing_label_count))
        print(
            "precision: {:.4f}, recall: {:.4f}, F1: {:.4f} "
            "(at confidence {:.3f})".format(
                precision, recall, f1, best_confidence
            )
        )
        print("mAP50: {:.4f}".format(map50))
        print("mAP50-95: {:.4f}".format(map5095))
    print("preprocess: {:.2f} ms/image".format(preprocessing_seconds / measured * 1000))
    print(
        "graph schedule (NumPy CPU + VART DPU): {:.2f} ms/image".format(
            partition_execution_seconds / measured * 1000
        )
    )
    print("decode+NMS: {:.2f} ms/image".format(postprocessing_seconds / measured * 1000))
    print(
        "total: {:.2f} ms/image ({:.2f} FPS)".format(
            (preprocessing_seconds + partition_execution_seconds + postprocessing_seconds)
            / measured
            * 1000,
            measured
            / (preprocessing_seconds + partition_execution_seconds + postprocessing_seconds),
        )
    )


if __name__ == "__main__":
    main()
