# Complete Reproduction & Execution Guide: AgriBot YOLOv4

This document outlines the end-to-end execution flow for reproducing and adapting the plant leaf disease detection methodology of **Aldakheel, Zakariah, and Alabdalall (2024)** (*Frontiers in Plant Science 15:1355941*) for potato leaf disease detection locally on Windows 11 with WSL2 and NVIDIA RTX GPU.

---

## 📋 System Prerequisites
- **OS**: Windows 11 with WSL2 Ubuntu
- **GPU**: NVIDIA RTX 3050 Laptop GPU (6 GB VRAM, sm_86) or higher
- **Python**: 3.10+ (Local virtual environment `.venv`)
- **Compilers**: GCC/G++ with CUDA 12.4+ and OpenCV 4.x inside WSL2

---

## 🚀 Step-by-Step Stage Execution

### Stage 1: Environment Setup & Darknet Toolchain
```powershell
# Create Python virtual environment
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt

# Download pretrained YOLOv4 backbone weights
python scripts/download_weights.py

# Inside WSL2 Ubuntu, compile AlexeyAB Darknet with GPU & OpenCV:
# make GPU=1 OPENCV=1 LIBSO=1 ARCH=sm_86
```

### Stage 2: Dataset Acquisition & Deduplication Audit
```powershell
# Clean and verify raw images from PlantVillage
# Audit near-duplicate images using perceptual hash (Hamming distance <= 5)
.\.venv\Scripts\python.exe scripts/audit_dataset.py
```
*Outputs*: `data/manifests/audit.csv`, `results/class_distribution.png`, `results/sample_grid.png`

### Stage 3: Leak-Free Split & Targeted Augmentation
```powershell
# Execute group-aware stratified 10-fold split preventing near-duplicate leakage
# Apply train-only Y-channel Histogram Equalization & Horizontal Flips for healthy leaves
.\.venv\Scripts\python.exe scripts/split_and_augment.py
```
*Outputs*: `data/manifests/split.csv`, `data/processed/`, `results/augmentation_demo.png`

### Stage 4: Automated HSV Leaf Segmentation & Bounding Boxes
```powershell
# Perform HSV green/diseased leaf segmentation to extract bounding boxes
# Form normalized YOLO labels (class xc yc w h) without re-thresholding jitter
.\.venv\Scripts\python.exe scripts/auto_annotate_yolo.py
.\.venv\Scripts\python.exe scripts/validate_labels.py
```
*Outputs*: Normalized `.txt` label files, `results/annotation_check.png`

### Stage 5: Darknet Manifest & Configuration Assembly
```powershell
# Create obj.names, train.txt, valid.txt, test.txt and WSL2 paths (/mnt/c/...)
.\.venv\Scripts\python.exe scripts/create_darknet_data.py
```
*Outputs*: `data/obj.names`, `data/obj.data`, `data/obj_wsl.data`, `data/train_wsl.txt`, `data/valid_wsl.txt`, `data/test_wsl.txt`

### Stage 6: YOLOv4 Neural Architecture Customization
```powershell
# Configure 3 classes, filters=24 before YOLO layers, batch=64, subdivisions=64
.\.venv\Scripts\python.exe scripts/create_custom_cfg.py
```
*Outputs*: `cfg/yolov4-custom-potato.cfg`

### Stage 7: Darknet Training on Local RTX GPU
```powershell
# Launch training in WSL2 with background process
wsl -d Ubuntu -e bash -c "cd /mnt/c/Users/vivek/OneDrive/Desktop/AgriBot && ./darknet/darknet detector train data/obj_wsl.data cfg/yolov4-custom-potato.cfg weights/yolov4.conv.137 -dont_show > results/train.log 2>&1"

# Monitor live training iterations and loss:
.\.venv\Scripts\python.exe scripts/monitor_training.py
```
*Outputs*: `results/train.log`, `weights/backup/yolov4-custom-potato_last.weights`

### Stage 8: Evaluation on Untouched Test Set
```powershell
# Evaluates mAP@0.5, per-class AP, image-level Accuracy, Precision, Recall, F1
# Generates confusion matrix and error analysis sheet
.\.venv\Scripts\python.exe scripts/evaluate_test_set.py
```
*Outputs*: `results/metrics.json`, `results/test_metrics.md`, `results/confusion_matrix.png`, `results/errors.png`, `results/map_test.txt`

### Stage 9: Interactive Inference Demo
```powershell
# Predict single leaf image
.\.venv\Scripts\python.exe scripts/infer.py --image "path/to/leaf.jpg"

# Predict entire folder of images
.\.venv\Scripts\python.exe scripts/infer.py --folder "path/to/folder" --output "results/predictions"

# Launch Gradio interactive web application
.\.venv\Scripts\python.exe scripts/app_gradio.py
```

### Stage 10: Environmental Robustness Testing
```powershell
# Benchmark model under noise, blur, lighting, rotation, and hue variations
.\.venv\Scripts\python.exe scripts/test_robustness.py
```
*Outputs*: `results/robustness.csv`, `results/robustness.png`

### Stage 11: DenseNet-121 Baseline Comparison
```powershell
# Train and benchmark DenseNet-121 baseline against YOLOv4
.\.venv\Scripts\python.exe scripts/train_densenet_baseline.py
```
*Outputs*: `results/baseline_comparison.csv`, `results/baseline_comparison.md`
