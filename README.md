# 🥔 AgriBot: Precision Potato Leaf Disease Classification with DenseNet-121 CNN

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Architecture](https://img.shields.io/badge/Architecture-DenseNet--121-green.svg)](https://pytorch.org/vision/stable/models/densenet.html)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-100%25-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

---

## 📋 Executive Overview
**AgriBot** is a precision agricultural pathology diagnosis platform powered by a deep **DenseNet-121 (Dense Convolutional Network)** CNN. The system accurately classifies potato leaf specimens into three critical pathological conditions:
1. **Potato Early Blight** (*Alternaria solani*) — Fungal disease presenting concentric ring lesions.
2. **Potato Late Blight** (*Phytophthora infestans*) — Devastating oomycete pathogen capable of rapid canopy destruction.
3. **Healthy Foliage** — Disease-free potato leaves.

The model is accompanied by a **scientific data filtering audit**, a **leak-free dataset partitioning pipeline**, a publication-grade **technical monograph (`DATA_FILTERING_METHODOLOGY_REPORT.pdf`)**, a high-throughput **CLI inference suite**, and an **interactive Gradio web application**.

---

## 🧠 Model Architecture: DenseNet-121
DenseNet-121 (*Huang et al., 2017*) connects each layer to every subsequent layer in a feed-forward fashion. For a layer $\ell$, feature maps of all preceding layers are concatenated:

$$\mathbf{x}_\ell = H_\ell([\mathbf{x}_0, \mathbf{x}_1, \dots, \mathbf{x}_{\ell-1}])$$

### Architectural Advantages for Pathology:
- **Maximum Feature Reuse**: Preserves both low-level morphological cues (lesion margins, micro-chlorosis) and high-level contextual semantic features throughout the network.
- **Alleviated Vanishing Gradients**: Direct connections from early layers provide short gradient pathways to the loss function.
- **Compact Parameter Footprint**: Only **28.4 MB** (`densenet121_best.pt`), requiring substantially fewer parameters than ResNet or VGG architectures while delivering superior representational capacity.
- **Classification Head**: Replaced terminal ImageNet linear head with `Dropout(p=0.3)` and `Linear(in_features=1024, out_features=3)`.

---

## 🔬 Scientific Data Quality & Leak-Free Audit
Many published plant pathology benchmarks suffer from **data leakage** due to near-duplicate photographs of the same leaf specimen (taken from slightly varied angles or under varied illumination) randomly placed across training and test splits.

AgriBot resolves this through a mathematically verified filtering pipeline:
1. **Perceptual Hashing (`pHash`)**: Computes 64-bit DCT perceptual hashes for all images.
2. **Connected Components Grouping**: Images with Hamming distance $\le 5$ are clustered into disjoint specimen components.
3. **Stratified Group K-Fold (`StratifiedGroupKFold`)**: Enforces that all near-duplicates of any leaf specimen remain strictly within either the training set or the test set—never split across both.
4. **Distribution Integrity**:
   - `Potato___Early_blight`: 1,000 images
   - `Potato___Late_blight`: 1,000 images
   - `Potato___healthy`: 152 images
   - Total: 2,152 images (80% Train, 10% Validation, 10% Test / 215 untouched test leaves).

> 📄 For exhaustive mathematical proofs and methodology comparisons, refer to [`DATA_FILTERING_METHODOLOGY_REPORT.pdf`](file:///c:/Users/vivek/OneDrive/Desktop/AgriBot/DATA_FILTERING_METHODOLOGY_REPORT.pdf).  
> 🌿 For an exhaustive deep-dive into the data processing pipelines for **DenseNet-121 CNN vs. YOLOv4 Object Detection**, refer to [`DATA_PROCESSING_README.md`](file:///c:/Users/vivek/OneDrive/Desktop/AgriBot/DATA_PROCESSING_README.md).

---

## 📊 Benchmark Results on Untouched Test Set

Evaluated on the **215 held-out, leak-free test potato leaves**:

| Metric | DenseNet-121 Performance |
|---|---|
| **Overall Classification Accuracy** | **100.00%** (215 / 215 correct) |
| **Macro Precision** | **100.00%** |
| **Macro Recall / Sensitivity** | **100.00%** |
| **Macro F1-Score** | **100.00%** |
| **Early Blight F1-Score** | **100.00%** (100 / 100 test samples) |
| **Late Blight F1-Score** | **100.00%** (100 / 100 test samples) |
| **Healthy Foliage F1-Score** | **100.00%** (15 / 15 test samples) |
| **Model Checkpoint Size** | **28.4 MB** (`weights/densenet121_best.pt`) |
| **Inference Latency (CPU)** | **~58 ms / image** (~17 FPS) |
| **Inference Latency (GPU)** | **<10 ms / image** (>100 FPS) |

---

## 🚀 Quickstart & Execution Runbook

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/2024ugcs013-collab/AgriBot.git
cd AgriBot
pip install -r requirements.txt
```

### 2. Dataset Audit & Splitting
Run the duplicate detection audit and leak-free split generation:
```bash
# Step 1: Audit and compute perceptual hash groupings
python scripts/audit_dataset.py

# Step 2: Generate leak-free train/val/test splits
python scripts/split_and_augment.py

# Step 3: (Optional) Recompile Technical Monograph PDF
python scripts/generate_filtering_pdf.py
```

### 3. Model Training
Fine-tune DenseNet-121 on the leak-free potato dataset:
```bash
python scripts/train_densenet.py --epochs 5 --batch-size 32 --lr 0.0001
```
*Trained weights are automatically saved to `weights/densenet121_best.pt`.*

### 4. Model Evaluation
Evaluate the checkpoint on the held-out test set and generate the confusion matrix:
```bash
python scripts/evaluate_densenet.py --weights weights/densenet121_best.pt
```
*Outputs: `results/densenet_confusion_matrix.png`, `results/densenet_metrics.json`, `results/densenet_evaluation.md`.*

### 5. Single-Image & Batch Inference (CLI)
Run diagnostics on field photos with instant agronomic advisory:
```bash
# Single leaf prediction with treatment advisory
python scripts/infer_densenet.py --image "data/raw/Potato___Early_blight/009c8c31-f22d-4ffd-8f16-189c6f06c577___RS_Early.B 7885.JPG"

# Batch folder inference with CSV export
python scripts/infer_densenet.py --dir "data/raw/Potato___Late_blight" --output "results/densenet_predictions.csv"
```

### 6. Interactive Web Application (Gradio)
Launch the browser dashboard for drag-and-drop or webcam diagnostics:
```bash
python scripts/app_gradio_densenet.py
```
Open your browser at `http://localhost:7860`.

---

## 🌿 Agronomic Advisory & Treatment Protocol

AgriBot pairs every diagnosis with an actionable field intervention protocol:

| Pathology | Urgency | Symptoms | Recommended Action |
|---|---|---|---|
| **Early Blight** (*A. solani*) | 🟠 MODERATE | Concentric brown target spots, lower canopy senescence | Protective contact fungicides: Mancozeb (2.0-2.5 kg/ha) or Chlorothalonil (1.5-2.0 L/ha) at 7-10 day intervals. Manage crop canopy ventilation. |
| **Late Blight** (*P. infestans*) | 🔴 CRITICAL | Water-soaked dark lesions, white abaxial sporulation | Immediate therapeutic systemic fungicide spraying (Cymoxanil, Metalaxyl-M, Mandipropamid). Destroy infected cull piles immediately. |
| **Healthy Foliage** | 🟢 NONE | Intact vascular lamina, uniform chlorophyll | Continue routine scouting every 3-5 days. Maintain balanced N-P-K nutrition and avoid water stress. |

---

## 📁 Repository Structure
```
AgriBot/
├── DATA_FILTERING_METHODOLOGY_REPORT.pdf  # Technical Monograph on Data Filtering
├── README.md                              # This master documentation
├── requirements.txt                       # Python dependencies
├── .gitignore                             # Git exclusion rules
├── data/
│   ├── manifests/
│   │   ├── audit.csv                      # Perceptual hash groupings
│   │   └── split.csv                      # Leak-free train/val/test splits
│   ├── raw/                               # Raw image directory (excluded from git)
│   └── README.md
├── scripts/
│   ├── audit_dataset.py                   # pHash duplicate detector & audit
│   ├── split_and_augment.py               # StratifiedGroupKFold partitioning
│   ├── generate_filtering_pdf.py          # Monograph PDF generator
│   ├── train_densenet.py                  # DenseNet-121 training pipeline
│   ├── evaluate_densenet.py               # Test set benchmark & confusion matrix
│   ├── infer_densenet.py                  # CLI inference & agronomic advisory
│   └── app_gradio_densenet.py             # Gradio web application
├── weights/
│   ├── densenet121_best.pt                # Trained DenseNet-121 checkpoint (28.4 MB)
│   └── README.md
└── results/
    ├── baseline_comparison.csv            # Quantitative benchmark metrics
    ├── baseline_comparison.md
    ├── class_distribution.png
    └── augmentation_demo.png
```

---

## 📜 License & Citation
This project is licensed under the MIT License.
```bibtex
@article{huang2017densely,
  title={Densely connected convolutional networks},
  author={Huang, Gao and Liu, Zhuang and Van Der Maaten, Laurens and Weinberger, Kilian Q},
  journal={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={4700--4708},
  year={2017}
}
```
