# Ultra96-V2 Traffic Sign Detection and Classification

This repository contains an image-upload demo for the Avnet Ultra96-V2. It
detects traffic signs with a YOLOv8-derived model trained for GTSDB, then
classifies each detected crop using a separate 43-class GTSRB CNN.

The board-hosted web app lets a user upload one image, set the detector
confidence threshold, and view an annotated image and timing breakdown. The
detector and classifier remain separate because localization and traffic-sign
classification are distinct tasks.

> **Result status:** The classifier DPU test is a full 12,630-image test-set
> benchmark. The detector board metrics are provisional: its compiled graph
> includes CPU partitions executed by a custom NumPy fallback that has not been
> checked against a vendor CPU-task implementation. The web server is an
> unauthenticated local-network prototype, not an Internet-facing service.

## Contents

| Path | Purpose |
|---|---|
| `board/web_app.py` | Board HTTP server and image-upload inference endpoint |
| `board/web/index.html` | Browser user interface (HTML, CSS, vanilla JavaScript) |
| `board/combined_demo.py` | Runs detector then classifier, and annotates the image |
| `board/yolo_dpu_benchmark.py` | Detector graph execution, CPU fallback, decoding, NMS, and evaluation |
| `board/gtsrb_dpu_benchmark.py` | Full GTSRB classifier DPU benchmark |
| `models/detector/YOLOv8GTSDB.xmodel` | Compiled one-class traffic-sign detector |
| `models/classifier/TrafficSignCNN.xmodel` | Compiled 43-class traffic-sign classifier |
| `models/*/meta.json` | Compiler/runtime metadata for the model artifacts |
| `pc_tools/classifier/` | Classifier training, evaluation, weight export, and quantization sources |
| `pc_tools/detector/` | Detector training, export, quantization, and preparation sources |
| `src/src/` | Classifier architecture, data pipeline, training/evaluation helpers, and class labels |
| `pc_tools/classifier/dpu-pynq-2.5.0-ultra96v2/` | DPU overlay files included with this project; check upstream terms before redistribution |
| `metrics/model_metrics.json` | Recorded board and model metrics |
| `docs/MODEL_BUILD.md` | Build and deployment workflow notes |
| `docker/` | PC-side classifier container definition and dependencies |

Datasets and training checkpoints are **not included**. Training or reproducing
the reported evaluations requires obtaining the appropriate GTSRB/GTSDB data
and arranging it as expected by the scripts. The precompiled `.xmodel` files
are included for running the board demo.

## System overview

```text
Browser (PC)
    │ upload image + detector threshold
    ▼
Python HTTP server on Ultra96-V2 (ARM / PS)
    ├── preprocess image
    ├── execute YOLO graph: DPU partitions (FPGA / PL)
    │                       + NumPy CPU fallback partitions (ARM / PS)
    ├── decode boxes and apply NMS (ARM / PS)
    ├── classify each detected crop using the GTSRB DPU model
    └── return annotated JPEG + timing JSON
```

The detector is a single-class localizer (`sign`); it does not predict GTSRB
class IDs. The classifier receives each detector crop and always returns its
top-scoring class from 43 outputs; this demo has no classifier rejection
threshold.

### Hardware/software target

- Board: Avnet Ultra96-V2 (Zynq UltraScale+ MPSoC)
- Board OS: PYNQ Linux 3.0.1
- Board runtime: PYNQ-DPU 2.5.1, VART, and XIR
- DPU target: `DPUCZDX8G_ISA1_B1600`
- Input image processing and orchestration: Python, OpenCV, and NumPy
- Web server: Python standard-library `http.server`
- Browser interface: HTML, CSS, and vanilla JavaScript
- PC model workflow: PyTorch and Vitis AI quantization/compilation tools

TFLite artifacts from the broader project were used for separate CPU
experiments; they are not used by this board DPU inference path.

## Board setup and run

### Requirements

- Ultra96-V2 running the matching PYNQ Linux 3.0.1 image
- PYNQ-DPU 2.5.1 setup with the B1600 overlay and VART/XIR
- Python dependencies already available in the board's PYNQ environment:
  OpenCV (`cv2`) and NumPy
- Both `.xmodel` files and the board scripts from this repository

The application programs the DPU overlay and therefore must be run as root.
The project files should be stored in a persistent SD-card location such as
`/root/ultra96_demo`, not `/tmp` (which can be cleared on reboot).

### Copy files to the board

From Windows PowerShell, copy the project files to the board's temporary
directory (replace `pynq.local` with the board IP if name resolution is
unavailable):

```powershell
$project = "C:\path\to\Ultra96V2_TrafficSign_Demo"
scp "$project\board\web_app.py" xilinx@pynq.local:/tmp/web_app.py
scp "$project\board\combined_demo.py" xilinx@pynq.local:/tmp/combined_demo.py
scp "$project\board\yolo_dpu_benchmark.py" xilinx@pynq.local:/tmp/yolo_dpu_benchmark.py
scp "$project\board\web\index.html" xilinx@pynq.local:/tmp/web_index.html
scp "$project\models\detector\YOLOv8GTSDB.xmodel" xilinx@pynq.local:/tmp/YOLOv8GTSDB.xmodel
scp "$project\models\classifier\TrafficSignCNN.xmodel" xilinx@pynq.local:/tmp/TrafficSignCNN.xmodel
```

In the board SSH terminal, install them in a persistent directory:

```bash
sudo -i
mkdir -p /root/ultra96_demo/board/web
mkdir -p /root/ultra96_demo/models/detector
mkdir -p /root/ultra96_demo/models/classifier
cp /tmp/web_app.py /root/ultra96_demo/board/web_app.py
cp /tmp/combined_demo.py /root/ultra96_demo/board/combined_demo.py
cp /tmp/yolo_dpu_benchmark.py /root/ultra96_demo/board/yolo_dpu_benchmark.py
cp /tmp/web_index.html /root/ultra96_demo/board/web/index.html
cp /tmp/YOLOv8GTSDB.xmodel /root/ultra96_demo/models/detector/
cp /tmp/TrafficSignCNN.xmodel /root/ultra96_demo/models/classifier/
```

### Start the browser demo

After each board reboot, SSH to the board and run:

```bash
sudo -i
cd /root/ultra96_demo
. /etc/profile.d/xrt_setup.sh
. /etc/profile.d/pynq_venv.sh
python3 board/web_app.py --host 0.0.0.0 --port 8000
```

Keep the terminal open while the server is in use. On a PC on the same trusted
local network, open `http://pynq.local:8000` or
`http://<BOARD_IP>:8000`. Stop the server with Ctrl+C.

The server loads the overlay and models once at startup, then handles upload
requests sequentially because it reuses the same DPU runners. The confidence
field defaults to `0.01`; it can be changed per request. A lower threshold may
retain weaker detections but can increase false positives. Confidence applies
to the detector only; the classifier always selects its top class.

The result page reports:

- Detector stage time (image preparation, hybrid graph execution, and box
  decoding/NMS)
- Classification and annotation time
- Total inference time (detector plus classification/annotation)
- JPEG encoding time separately

Model startup and network upload/download time are excluded. The web server has
no authentication, request serving is sequential, and it should only be used
on a trusted local network.

### Command-line image demo

With an input image available on the board:

```bash
cd /root/ultra96_demo
. /etc/profile.d/xrt_setup.sh
. /etc/profile.d/pynq_venv.sh
python3 board/combined_demo.py /path/to/input.jpg \
  --detector models/detector/YOLOv8GTSDB.xmodel \
  --classifier models/classifier/TrafficSignCNN.xmodel \
  --output /path/to/annotated.jpg \
  --confidence 0.01 \
  --iou 0.7
```

## Model workflows

### GTSRB classifier

The classifier is a compact VGG-style CNN with approximately 0.6 million
parameters. It takes a 48×48 RGB crop and predicts one of 43 GTSRB classes.
The training pipeline uses ROI crops, GTSRB normalization, augmentations, and
a track-aware train/validation split.

The DPU artifact follows a PyTorch/Vitis AI path, not the TFLite path:

```text
PyTorch checkpoint
  → exported NumPy weights
  → Vitis AI INT8 calibration/quantization
  → compiled TrafficSignCNN.xmodel for B1600
  → VART runner on the board DPU
```

TFLite float32/INT8 models are separate CPU experiment artifacts and do not
feed the DPU compiler path.

### GTSDB detector

The detector is a YOLOv8-derived, one-class model adapted for compilation.
The selected graph uses three output scales at 640×640 input. Its compiled
graph runs across DPU and CPU partitions; supported DPU operations execute
through VART, while known CPU fallback operations are implemented in NumPy.
Box decoding, confidence filtering, coordinate mapping, and NMS run on the ARM
CPU.

See [docs/MODEL_BUILD.md](docs/MODEL_BUILD.md) for the recorded build sequence
and source files. Rebuilding requires the datasets, appropriate checkpoints,
and compatible AMD/Xilinx Vitis AI tooling; those large data/checkpoint assets
are not included here.

## Results

Full machine-readable values and test scope are in
[`metrics/model_metrics.json`](metrics/model_metrics.json).

### GTSRB classifier on board DPU

- Test set: 12,630 GTSRB images, cropped using the supplied ROI annotations
- Correct: 12,559; accuracy: **99.44%**
- Average DPU inference: **1.14 ms/crop**
- Average preprocessing: **2.31 ms/crop**
- Average combined preprocessing plus DPU: **3.46 ms/crop**
- The approximately **99.43% PC CPU** baseline is from the separate PyTorch
  float evaluation. It is not a measured board ARM CPU result.

### GTSDB detector board benchmark

- Test set: 300 images, 361 labeled signs; 65 images had no label files and
  were treated as background
- mAP@50: **0.7440**; mAP@50–95: **0.4915**
- Precision: **0.9025**; recall: **0.5623**; F1: **0.6929**
- Best-F1 threshold measured on that evaluation: **0.068**
- Mean processing time: **1871.46 ms/image**, approximately **0.53 FPS**
- Breakdown: preprocessing **45.76 ms**, hybrid graph schedule
  **1742.52 ms**, decode/NMS **83.18 ms**
- Compiled graph partitions: **55 DPU** and **68 CPU**

**Detector metric caveat:** the custom NumPy implementation for graph CPU
partitions has not been compared numerically against the vendor CPU-task
implementation. The board metrics are therefore provisional for this
hybrid implementation, not a fully validated vendor-runtime reference. The
best-F1 value 0.068 is an evaluation operating point; the web UI default 0.01
is a user-selected lower threshold and is not the measured best-F1 setting.

### Combined demo smoke tests

These are individual-image functional checks, not dataset accuracy or sustained
throughput benchmarks:

- GTSDB test image `00601.jpg`: one detection; combined inference **1981.52
  ms**, excluding **1855.21 ms** model setup.
- `imx.png` at confidence 0.068: one detection classified as **General caution
  (class 18)**; combined inference **1965.12 ms**, excluding **2136.96 ms**
  setup.

## Repository scope and limitations

- Datasets and original training checkpoints are not included.
- Model training and evaluation sources are included, but exact retraining and
  recompilation need external data and compatible tooling.
- The detector is a hybrid DPU/ARM graph and currently runs around 0.53 FPS in
  the measured setup; this is not real-time video inference.
- Detector metrics are provisional pending validation of the custom NumPy
  CPU-partition implementation.
- The classifier always predicts one of 43 classes for every detector crop;
  there is no classifier rejection threshold.
- The browser server is a local-network demo with no authentication.

## Acknowledgements and licensing

This project uses AMD/Xilinx PYNQ, Vitis AI, VART/XIR, and DPU-PYNQ components,
as well as GTSRB/GTSDB data and YOLO-related software. Review the upstream
licenses and redistribution terms for datasets, dependencies, and included
overlay/model binaries before redistributing this repository. The upstream
DPU-PYNQ v2.5.1 `LICENSE`, including its binary redistribution conditions, is
included beside the Ultra96-V2 overlay files. No project license is granted by
this README.
