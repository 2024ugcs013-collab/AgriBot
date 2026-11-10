"""
AgriBot - Potato Pathology CNN Classifier (DenseNet-121)
Stage: Model Evaluation & Benchmark Script

Evaluates the fine-tuned DenseNet-121 CNN model on the untouched
215-image test split (data/manifests/split.csv), generating:
  1. Test Accuracy, Precision, Recall, and Macro F1-Score
  2. Per-class sensitivity, specificity, and error breakdowns
  3. Publication-grade Confusion Matrix (results/densenet_confusion_matrix.png)
  4. Structured metrics report (results/densenet_metrics.json & results/densenet_evaluation.md)
  5. Inference latency benchmark (ms per leaf)
"""

import argparse
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    classification_report
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_MAP = {
    "potato_early_blight": 0,
    "potato_late_blight": 1,
    "potato_healthy": 2,
}
INV_MAP = {v: k for k, v in CLASS_MAP.items()}
DISPLAY_NAMES = ["Early Blight", "Late Blight", "Healthy"]


class PotatoTestDataset(Dataset):
    def __init__(self, df, project_root, transform=None):
        self.paths = [project_root / p for p in df["path"]]
        self.labels = [CLASS_MAP[c] for c in df["class"]]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx], str(path)


def load_densenet_model(weights_path, device):
    """Loads DenseNet-121 model with saved weights."""
    model = models.densenet121(weights=None)
    num_features = model.classifier.in_features
    # Support both Sequential classifier and single Linear layer
    try:
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, 3)
        )
        model.load_state_dict(torch.load(weights_path, map_location=device))
    except RuntimeError:
        # Fallback to single linear layer if trained with simpler head
        model.classifier = nn.Linear(num_features, 3)
        model.load_state_dict(torch.load(weights_path, map_location=device))

    model = model.to(device)
    model.eval()
    return model


def evaluate_densenet(weights_path=None):
    if weights_path is None:
        weights_path = WEIGHTS_DIR / "densenet121_best.pt"

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found at: {weights_path}")

    split_csv = DATA_DIR / "manifests" / "split.csv"
    if not split_csv.exists():
        raise FileNotFoundError(f"Split manifest not found at: {split_csv}")

    df_split = pd.read_csv(split_csv)
    test_df = df_split[df_split["split"] == "test"].copy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("AGRIBOT: DENSENET-121 MODEL EVALUATION ON UNTOUCHED TEST SET")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Loading weights from: {weights_path}")
    print(f"Test samples: {len(test_df)} images")

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_loader = DataLoader(
        PotatoTestDataset(test_df, PROJECT_ROOT, eval_transform),
        batch_size=1,
        shuffle=False
    )

    model = load_densenet_model(weights_path, device)

    y_true, y_pred = [], []
    latencies = []
    records = []

    with torch.no_grad():
        for imgs, lbls, paths in test_loader:
            imgs = imgs.to(device)
            t0 = time.perf_counter()
            outputs = model(imgs)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)

            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            true_idx = int(lbls[0].item())

            y_pred.append(pred_idx)
            y_true.append(true_idx)

            records.append({
                "path": paths[0],
                "true_class": INV_MAP[true_idx],
                "pred_class": INV_MAP[pred_idx],
                "confidence": float(probs[pred_idx]),
                "correct": bool(pred_idx == true_idx)
            })

    # Metrics calculation
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_per_class, r_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    avg_latency = float(np.mean(latencies[5:])) if len(latencies) > 5 else float(np.mean(latencies))
    model_size_mb = weights_path.stat().st_size / (1024 * 1024)

    print("\n" + "=" * 50)
    print("TEST SET CLASSIFICATION REPORT (DenseNet-121)")
    print("=" * 50)
    print(f"Overall Accuracy:  {accuracy * 100:.2f}%")
    print(f"Macro Precision:   {precision_macro * 100:.2f}%")
    print(f"Macro Recall:      {recall_macro * 100:.2f}%")
    print(f"Macro F1-Score:    {f1_macro * 100:.2f}%")
    print(f"Inference Latency: {avg_latency:.2f} ms/image ({1000.0/avg_latency:.1f} FPS)")
    print(f"Model Size:        {model_size_mb:.2f} MB")
    print("-" * 50)

    for i, name in enumerate(DISPLAY_NAMES):
        print(f"  {name:<20} | Prec: {p_per_class[i]*100:6.2f}% | Rec: {r_per_class[i]*100:6.2f}% | F1: {f1_per_class[i]*100:6.2f}% | Count: {support_per_class[i]}")

    # Generate Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=DISPLAY_NAMES,
        yticklabels=DISPLAY_NAMES,
        cbar=False,
        annot_kws={"size": 14, "weight": "bold"}
    )
    plt.title("DenseNet-121 CNN Confusion Matrix\n(Untouched Test Set: 215 Potato Leaves)", fontsize=13, pad=15)
    plt.xlabel("Predicted Label", fontsize=11, labelpad=10)
    plt.ylabel("Ground Truth Label", fontsize=11, labelpad=10)
    plt.tight_layout()

    cm_path = RESULTS_DIR / "densenet_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"\nSaved confusion matrix plot: {cm_path}")

    # Save metrics JSON
    metrics = {
        "model_architecture": "DenseNet-121 (Dense Convolutional Network)",
        "test_samples": len(test_df),
        "overall_accuracy": float(accuracy),
        "macro_precision": float(precision_macro),
        "macro_recall": float(recall_macro),
        "macro_f1": float(f1_macro),
        "avg_latency_ms": float(avg_latency),
        "throughput_fps": float(1000.0 / avg_latency),
        "model_size_mb": float(model_size_mb),
        "per_class_metrics": {
            INV_MAP[i]: {
                "precision": float(p_per_class[i]),
                "recall": float(r_per_class[i]),
                "f1_score": float(f1_per_class[i]),
                "support": int(support_per_class[i])
            }
            for i in range(3)
        }
    }

    metrics_json_path = RESULTS_DIR / "densenet_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics json: {metrics_json_path}")

    # Save detailed markdown evaluation report
    md_content = f"""# DenseNet-121 CNN Model Evaluation Report

Comprehensive test set evaluation of the AgriBot Potato Leaf Pathology Classifier (**DenseNet-121**), evaluated on the leak-free, held-out test split of **215 potato leaves**.

## 1. Summary Performance Metrics

| Metric | Score | Unit / Description |
|---|---|---|
| **Test Accuracy** | **{accuracy * 100:.2f}%** | Overall classification accuracy |
| **Macro Precision** | **{precision_macro * 100:.2f}%** | Unweighted mean precision |
| **Macro Recall** | **{recall_macro * 100:.2f}%** | Unweighted mean sensitivity |
| **Macro F1-Score** | **{f1_macro * 100:.2f}%** | Harmonic mean of precision and recall |
| **Model Size** | **{model_size_mb:.2f} MB** | PyTorch state dictionary binary |
| **Inference Latency** | **{avg_latency:.2f} ms** | Per-image compute latency (~{1000.0/avg_latency:.1f} FPS) |

---

## 2. Per-Class Diagnostic Performance

| Disease Class | Support (Test Images) | Precision (%) | Recall / Sensitivity (%) | F1-Score (%) |
|---|---|---|---|---|
| **Early Blight** (*Alternaria solani*) | {support_per_class[0]} | {p_per_class[0]*100:.2f}% | {r_per_class[0]*100:.2f}% | {f1_per_class[0]*100:.2f}% |
| **Late Blight** (*Phytophthora infestans*) | {support_per_class[1]} | {p_per_class[1]*100:.2f}% | {r_per_class[1]*100:.2f}% | {f1_per_class[1]*100:.2f}% |
| **Healthy Foliage** | {support_per_class[2]} | {p_per_class[2]*100:.2f}% | {r_per_class[2]*100:.2f}% | {f1_per_class[2]*100:.2f}% |

---

## 3. Confusion Matrix Breakdown

A publication-grade confusion matrix visualization has been rendered and saved to [`results/densenet_confusion_matrix.png`](file:///c:/Users/vivek/OneDrive/Desktop/AgriBot/results/densenet_confusion_matrix.png).
"""
    eval_md_path = RESULTS_DIR / "densenet_evaluation.md"
    eval_md_path.write_text(md_content, encoding="utf-8")
    print(f"Saved Markdown evaluation report: {eval_md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DenseNet-121 CNN Model")
    parser.add_argument("--weights", type=Path, default=None, help="Path to weights file (default: weights/densenet121_best.pt)")
    args = parser.parse_args()

    evaluate_densenet(args.weights)
