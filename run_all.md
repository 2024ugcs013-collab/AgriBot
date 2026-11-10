# Complete Execution Guide: AgriBot CNN (DenseNet-121)

This runbook guides you through the complete execution of the **AgriBot DenseNet-121 Convolutional Neural Network** pipeline for potato leaf disease detection, from dataset audit and leak-free splitting to training, evaluation, and edge inference.

---

## 💻 System Prerequisites
- **OS**: Windows 10/11 or Linux
- **Python**: 3.10+ (Virtual environment recommended)
- **PyTorch**: 2.0+ (CUDA GPU acceleration supported, or CPU)
- **RAM**: 8 GB+

---

## 🛠️ Step-by-Step Execution Runbook

### Step 1: Environment Setup
```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### Step 2: Dataset Acquisition & Deduplication Audit
Ensure the potato disease images are located in `data/raw/` (or extracted from `dataset/potato_disease.zip`).
```powershell
python scripts/audit_dataset.py
```
- Performs resolution & integrity verification across all 2,152 images.
- Computes 64-bit DCT perceptual hashes (`pHash`) and clusters near-duplicate leaves (Hamming distance $\le 5$).
- Outputs: `data/manifests/audit.csv`, `results/class_distribution.png`.

### Step 3: Leak-Free Stratified Group Partitioning
```powershell
python scripts/split_and_augment.py
```
- Utilizes `StratifiedGroupKFold` using `dup_group` to guarantee that all duplicates/angles of the same leaf stay strictly within the same fold.
- Partitions data into 80% Train, 10% Validation, and 10% Held-Out Test (215 leaves).
- Outputs: `data/manifests/split.csv`.

### Step 4: Generate Technical Monograph PDF
```powershell
python scripts/generate_filtering_pdf.py
```
- Compiles the publication-grade PDF report documenting data quality, duplicate detection mathematics, and leak-free validation.
- Output: `DATA_FILTERING_METHODOLOGY_REPORT.pdf`.

### Step 5: DenseNet-121 Model Training
```powershell
python scripts/train_densenet.py --epochs 5 --batch-size 32 --lr 0.0001
```
- Loads the leak-free train/val sets, applies transforms and augmentations.
- Fine-tunes DenseNet-121 with AdamW optimizer and cosine learning rate decay.
- Saves the top-performing checkpoint to `weights/densenet121_best.pt` and plots curves in `results/densenet_training_curves.png`.

### Step 6: Test Set Evaluation & Metrics
```powershell
python scripts/evaluate_densenet.py --weights weights/densenet121_best.pt
```
- Evaluates on the 215 untouched test samples.
- Computes Accuracy (100.00%), Macro F1 (100.00%), Precision, Recall, and per-class breakdowns.
- Generates `results/densenet_confusion_matrix.png` and `results/densenet_evaluation.md`.

### Step 7: Single-Image & Batch Inference (CLI)
```powershell
# Test single image
python scripts/infer_densenet.py --image "data/raw/Potato___Early_blight/009c8c31-f22d-4ffd-8f16-189c6f06c577___RS_Early.B 7885.JPG"

# Process batch directory
python scripts/infer_densenet.py --dir "data/raw/Potato___Late_blight" --output "results/predictions.csv"
```

### Step 8: Launch Interactive Web Application
```powershell
python scripts/app_gradio_densenet.py
```
- Access at `http://localhost:7860` for real-time leaf diagnosis and treatment advisory.
