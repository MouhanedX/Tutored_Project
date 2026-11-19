# Model Build and Deployment Record

## GTSRB classifier

The classifier is a compact VGG-style CNN with 43 output classes.

1. Train on GTSRB using track-aware train/validation splitting.
2. Evaluate the float model on the official test set.
3. Export PyTorch weights to `gtsrb_float.npz`.
4. Calibrate INT8 using GTSRB training images.
5. Run simulated INT8 test evaluation.
6. Export `TrafficSignCNN_int.xmodel`.
7. Compile for `DPUCZDX8G_ISA1_B1600`.
8. Load the compiled `TrafficSignCNN.xmodel` with PYNQ-DPU.

Relevant PC sources:

- `pc_tools/classifier/train.py`
- `pc_tools/classifier/evaluate.py`
- `pc_tools/classifier/export_weights.py`
- `pc_tools/classifier/quantize.py`
- `src/model.py`
- `src/data.py`
- `src/classes.py`

Relevant board sources:

- `board/gtsrb_dpu_benchmark.py`
- `board/combined_demo.py`

## GTSDB detector

The detector is a YOLOv8-derived C3-SiLU, one-class model. C2f was replaced
by C3 for Vitis compatibility. The SiLU layers were decomposed into multiply
plus sigmoid for the quantization/export path.

1. Train the compatible YOLO architecture on GTSDB.
2. Export the convolutional prediction heads without NMS/DFL.
3. Calibrate in INT8 using 100 GTSDB training images.
4. Export and compile for Ultra96-V2 B1600.
5. Execute DPU partitions with VART.
6. Execute the graph's known CPU partitions with NumPy.
7. Decode DFL distributions and run NMS on the ARM CPU.

Relevant sources:

- `pc_tools/detector/train_yolov8_dpu.py`
- `pc_tools/detector/export_yolov8_dpu_raw.py`
- `pc_tools/detector/vitis_quantize_yolov8_raw.py`
- `pc_tools/detector/prepare_gtsdb_detector.py`
- `board/yolo_dpu_benchmark.py`

## Board setup

The board runtime uses the Ultra96-V2 DPU overlay in
`pc_tools/classifier/dpu-pynq-2.5.0-ultra96v2/`.

The architecture metadata is in
`pc_tools/classifier/arch_ultra96.json`.
