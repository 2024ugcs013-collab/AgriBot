# 📁 Dataset Directory Structure: AgriBot (DenseNet-121 CNN & YOLOv4)

This directory holds the dataset manifests, annotations, and data partitions for training, validation, and evaluation of both the **DenseNet-121 CNN** classifier and the **YOLOv4** object detector.

> 📖 **Comprehensive Data Engineering Guide**:  
> For an exhaustive mathematical and technical breakdown of the preprocessing, normalization, automated leaf segmentation, and targeted augmentation pipelines, see [`DATA_PROCESSING_README.md`](../DATA_PROCESSING_README.md).

---

### Directory Layout
```
data/
├── manifests/
│   ├── audit.csv            # Perceptual hash audit (pHash) and duplicate component groups
│   └── split.csv            # Leak-free Stratified Group K-Fold split manifest (train/val/test)
├── processed/               # Flat directories with images and YOLO .txt labels
│   ├── train/               # 5,352 images (Targeted train-only augmented) + matching .txt labels
│   ├── valid/               # 194 raw images (Untouched) + matching .txt labels
│   └── test/                # 215 raw images (Untouched held-out) + matching .txt labels
├── raw/                     # Canonical PlantVillage potato image directory (excluded from git)
│   ├── Potato___Early_blight/  (1,000 images, Alternaria solani)
│   ├── Potato___Late_blight/   (1,000 images, Phytophthora infestans)
│   └── Potato___healthy/       (152 images, Disease-free foliage)
├── obj.names                # Darknet class definitions (early_blight, late_blight, healthy)
├── obj.data / obj_wsl.data  # Darknet dataset configuration descriptors
├── train.txt / train_wsl.txt# Darknet manifests pointing to 5,352 training images
├── valid.txt / valid_wsl.txt# Darknet manifests pointing to 194 validation images
└── test.txt / test_wsl.txt  # Darknet manifests pointing to 215 held-out test images
```

---

### Summary of Data Processing by Model

| Processing Dimension | DenseNet-121 CNN | YOLOv4 Object Detector |
|---|---|---|
| **Data Sourcing** | In-memory streaming from `data/manifests/split.csv` referencing `data/raw/` | Flat image & label files in `data/processed/` |
| **Input Resolution** | $224 \times 224 \times 3$ | $416 \times 416 \times 3$ (Letterboxed with gray padding) |
| **Ground Truth Format** | Scalar integer label (`0`, `1`, `2`) | Normalized YOLO bounding box (`<c> <xc> <yc> <w> <h>`) |
| **Label Generation** | Direct directory / CSV mapping | 4-Tier HSV, Otsu, and GrabCut automated leaf segmentation |
| **Augmentation Mode** | In-memory dynamic PyTorch transforms (Flips, Rotations, Jitter) | Offline Y-channel Histogram Equalization (YCrCb) + Online Darknet Mosaic |
| **Normalization** | ImageNet z-score ($\mu=[0.485, 0.456, 0.406]$, $\sigma=[0.229, 0.224, 0.225]$) | 0–1 float scaling ($x / 255.0$) |

---

### Quick Reproduction Runbook
```bash
# 1. Audit duplicates with perceptual hashing
python scripts/audit_dataset.py

# 2. Partition leak-free splits and apply train-only augmentation
python scripts/split_and_augment.py

# 3. Generate YOLO bounding boxes via automated leaf segmentation
python scripts/annotate_dataset.py

# 4. Verify 100% of generated labels
python scripts/validate_labels.py

# 5. Build Darknet manifests and config files
python scripts/create_darknet_data.py
python scripts/create_custom_cfg.py
```
