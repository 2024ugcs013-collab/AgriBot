"""
AgriBot - Potato Pathology CNN Classifier (DenseNet-121)
Stage: Model Training and Validation Pipeline

Trains a DenseNet-121 convolutional neural network on the leak-free
dataset split (data/manifests/split.csv) for 3-class potato leaf disease classification:
  - Potato Early Blight (Alternaria solani)
  - Potato Late Blight (Phytophthora infestans)
  - Potato Healthy
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
RESULTS_DIR = PROJECT_ROOT / "results"

WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_MAP = {
    "potato_early_blight": 0,
    "potato_late_blight": 1,
    "potato_healthy": 2,
}
INV_MAP = {v: k for k, v in CLASS_MAP.items()}
CLASS_NAMES = ["Potato Early Blight", "Potato Late Blight", "Potato Healthy"]


class PotatoDataset(Dataset):
    """PyTorch Dataset loading images based on manifest DataFrame."""
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
        return img, self.labels[idx]


def build_densenet121(num_classes=3, pretrained=True):
    """Constructs DenseNet-121 with a custom 3-class linear classifier."""
    weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
    model = models.densenet121(weights=weights)
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_features, num_classes)
    )
    return model


def train_densenet(epochs=5, batch_size=32, lr=1e-4, weight_decay=1e-2):
    """Full fine-tuning routine for DenseNet-121."""
    print("=" * 70)
    print("AGRIBOT: DENSENET-121 CNN MODEL TRAINING PIPELINE")
    print("=" * 70)

    split_csv = DATA_DIR / "manifests" / "split.csv"
    if not split_csv.exists():
        raise FileNotFoundError(f"Dataset split manifest not found at: {split_csv}")

    df_split = pd.read_csv(split_csv)
    train_df = df_split[df_split["split"] == "train"].copy()
    val_df = df_split[df_split["split"] == "valid"].copy()
    test_df = df_split[df_split["split"] == "test"].copy()

    print(f"Dataset Partitions:")
    print(f"  - Training Set:   {len(train_df):>5} images")
    print(f"  - Validation Set: {len(val_df):>5} images")
    print(f"  - Test Set:       {len(test_df):>5} images (Held-out untouched)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}")

    # Standard ImageNet normalization and 224x224 input sizing
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_loader = DataLoader(
        PotatoDataset(train_df, PROJECT_ROOT, train_transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        PotatoDataset(val_df, PROJECT_ROOT, eval_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    print("Instantiating pretrained DenseNet-121 architecture...")
    model = build_densenet121(num_classes=3, pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_weights_path = WEIGHTS_DIR / "densenet121_best.pt"

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": []
    }

    print(f"\nCommencing training across {epochs} epochs...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(lbls)
            train_correct += (outputs.argmax(dim=1) == lbls).sum().item()
            train_total += len(lbls)

        scheduler.step()
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total

        # Validation Phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, lbls)
                val_loss += loss.item() * len(lbls)
                val_correct += (outputs.argmax(dim=1) == lbls).sum().item()
                val_total += len(lbls)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {epoch_train_loss:.4f} - Acc: {epoch_train_acc * 100:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f} - Acc: {epoch_val_acc * 100:.2f}%"
        )

        if epoch_val_acc >= best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), best_weights_path)
            print(f"  -> Model checkpoint updated ({best_weights_path.name})")

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f}s. Best Val Accuracy: {best_val_acc * 100:.2f}%")

    # Plot training curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), history["train_loss"], label="Train Loss", marker="o")
    plt.plot(range(1, epochs + 1), history["val_loss"], label="Val Loss", marker="s")
    plt.title("DenseNet-121 Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), [a * 100 for a in history["train_acc"]], label="Train Acc", marker="o")
    plt.plot(range(1, epochs + 1), [a * 100 for a in history["val_acc"]], label="Val Acc", marker="s")
    plt.title("DenseNet-121 Accuracy Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    curves_path = RESULTS_DIR / "densenet_training_curves.png"
    plt.savefig(curves_path, dpi=200)
    plt.close()
    print(f"Training curves saved to: {curves_path}")

    # Save history json
    history_path = RESULTS_DIR / "densenet_training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {history_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DenseNet-121 CNN for Potato Disease Classification")
    parser.add_argument("--epochs", type=int, default=5, help="Number of fine-tuning epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    args = parser.parse_args()

    train_densenet(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
