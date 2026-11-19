"""Serve a browser-based image inference demo on the Ultra96-V2."""
import argparse
import base64
import binascii
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import cv2
import numpy as np
from pynq_dpu import DpuOverlay

import combined_demo as demo


BOARD_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BOARD_DIR.parent
WEB_PAGE = BOARD_DIR / "web" / "index.html"
OVERLAY_PATH = Path(
    "/usr/local/share/pynq-venv/lib/python3.10/site-packages/pynq_dpu/dpu.bit"
)
MAX_REQUEST_BYTES = 12 * 1024 * 1024
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000


class BoardInference:
    def __init__(self, detector_path, classifier_path):
        setup_start = time.perf_counter()
        self.overlay = DpuOverlay(str(OVERLAY_PATH))
        print("DPU overlay loaded", flush=True)
        self.detector_state = demo.create_detector(detector_path)
        print("detector runners ready", flush=True)
        self.classifier_state = demo.create_classifier(classifier_path)
        print("classifier runner ready", flush=True)
        self.setup_ms = (time.perf_counter() - setup_start) * 1000
        print("model setup: {:.2f} ms".format(self.setup_ms), flush=True)

    def run(self, image, confidence, iou, max_detections):
        start = time.perf_counter()
        detections = demo.detector_run(
            image,
            self.detector_state,
            confidence,
            iou,
            max_detections,
        )
        detector_ms = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        annotated = demo.draw_result(
            image,
            detections,
            self.classifier_state,
        )
        classification_and_annotation_ms = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        success, encoded = cv2.imencode(
            ".jpg",
            annotated,
            [cv2.IMWRITE_JPEG_QUALITY, 90],
        )
        if not success:
            raise RuntimeError("could not encode annotated image")
        encoding_ms = (time.perf_counter() - start) * 1000

        return {
            "annotated_image": "data:image/jpeg;base64,"
            + base64.b64encode(encoded.tobytes()).decode("ascii"),
            "detections": len(detections),
            "confidence": confidence,
            "timings_ms": {
                "detector": round(detector_ms, 2),
                "classification_and_annotation": round(
                    classification_and_annotation_ms, 2
                ),
                "inference_total": round(
                    detector_ms + classification_and_annotation_ms, 2
                ),
                "image_encoding": round(encoding_ms, 2),
            },
        }


class DemoRequestHandler(BaseHTTPRequestHandler):
    engine = None
    default_confidence = 0.01
    iou = 0.7
    max_detections = 300

    def _send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path not in ("/", "/index.html"):
            self._send_json(404, {"error": "page not found"})
            return
        page = WEB_PAGE.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(page)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(page)

    def do_POST(self):
        if self.path != "/infer":
            self._send_json(404, {"error": "endpoint not found"})
            return
        if self.headers.get_content_type() != "application/json":
            self._send_json(415, {"error": "send an application/json request"})
            return

        length_header = self.headers.get("Content-Length")
        if length_header is None:
            self._send_json(411, {"error": "Content-Length is required"})
            return
        try:
            content_length = int(length_header)
        except ValueError:
            self._send_json(400, {"error": "invalid Content-Length"})
            return
        if content_length <= 0:
            self._send_json(400, {"error": "request body is empty"})
            return
        if content_length > MAX_REQUEST_BYTES:
            self._send_json(413, {"error": "request is too large"})
            return

        try:
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            self._send_json(400, {"error": "request body must be valid JSON"})
            return
        if not isinstance(payload, dict):
            self._send_json(400, {"error": "request body must be a JSON object"})
            return

        image_data = payload.get("image")
        if not isinstance(image_data, str):
            self._send_json(400, {"error": "image must be a base64-encoded image"})
            return
        if image_data.startswith("data:"):
            header, separator, image_data = image_data.partition(",")
            if (
                not separator
                or not header.startswith("data:image/")
                or not header.endswith(";base64")
            ):
                self._send_json(400, {"error": "unsupported image data URL"})
                return
        try:
            image_bytes = base64.b64decode(image_data, validate=True)
        except (binascii.Error, ValueError):
            self._send_json(400, {"error": "image is not valid base64"})
            return
        if not image_bytes or len(image_bytes) > MAX_IMAGE_BYTES:
            self._send_json(413, {"error": "image must be between 1 byte and 8 MB"})
            return

        try:
            image = cv2.imdecode(
                np.frombuffer(image_bytes, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
        except cv2.error:
            self._send_json(400, {"error": "uploaded file is not a supported image"})
            return
        if image is None:
            self._send_json(400, {"error": "uploaded file is not a supported image"})
            return
        if image.shape[0] * image.shape[1] > MAX_IMAGE_PIXELS:
            self._send_json(413, {"error": "image dimensions are too large"})
            return

        confidence = payload.get("confidence", self.default_confidence)
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0 <= confidence <= 1
        ):
            self._send_json(400, {"error": "confidence must be a number from 0 to 1"})
            return

        result = self.engine.run(
            image,
            float(confidence),
            self.iou,
            self.max_detections,
        )
        self._send_json(200, result)


def main():
    parser = argparse.ArgumentParser(
        description="Run the Ultra96-V2 traffic sign image upload web demo."
    )
    parser.add_argument(
        "--detector",
        type=Path,
        default=PROJECT_DIR / "models" / "detector" / "YOLOv8GTSDB.xmodel",
    )
    parser.add_argument(
        "--classifier",
        type=Path,
        default=PROJECT_DIR / "models" / "classifier" / "TrafficSignCNN.xmodel",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--confidence", type=float, default=0.01)
    args = parser.parse_args()

    if not 0 <= args.confidence <= 1:
        parser.error("--confidence must be from 0 to 1")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be from 1 to 65535")
    if not args.detector.is_file():
        parser.error("detector model not found: {}".format(args.detector))
    if not args.classifier.is_file():
        parser.error("classifier model not found: {}".format(args.classifier))
    if not WEB_PAGE.is_file():
        parser.error("web page not found: {}".format(WEB_PAGE))

    engine = BoardInference(args.detector, args.classifier)
    handler = type(
        "ConfiguredDemoRequestHandler",
        (DemoRequestHandler,),
        {"engine": engine, "default_confidence": args.confidence},
    )
    server = HTTPServer((args.host, args.port), handler)
    print(
        "web demo listening on {}:{}; open http://<BOARD_IP>:{}".format(
            args.host, args.port, args.port
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopping web demo", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
