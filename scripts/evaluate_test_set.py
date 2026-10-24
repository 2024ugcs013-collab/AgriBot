"""
Stage 8: Comprehensive Evaluation on Untouched Test Set
Evaluates YOLOv4 on all 215 untouched test images.
Computes mAP@0.5, per-class AP, Image-Level Accuracy, Precision, Recall, F1,
and produces confusion matrix heatmap and error contact sheets.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["potato_early_blight", "potato_late_blight", "potato_healthy"]
CLASS_MAP = {0: "potato_early_blight", 1: "potato_late_blight", 2: "potato_healthy"}


def run_darknet_map(weights_path: Path):
    """Run AlexeyAB Darknet detector map on test set in WSL2."""
    print("=" * 65)
    print("RUNNING DARKNET DETECTOR MAP ON UNTOUCHED TEST SET")
    print("=" * 65)

    wsl_weights = f"/mnt/c/{weights_path.resolve().as_posix()[3:]}"
    cmd = [
        "wsl",
        "-d",
        "Ubuntu",
        "-e",
        "bash",
        "-c",
        f"cd /mnt/c/Users/vivek/OneDrive/Desktop/AgriBot && "
        f"./darknet/darknet detector map data/obj_test_wsl.data cfg/yolov4-custom-potato.cfg '{wsl_weights}'",
    ]

    print(f"Executing: {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = proc.stdout
        map_file = RESULTS_DIR / "map_test.txt"
        map_file.write_text(output, encoding="utf-8")
        print(f"Saved Darknet mAP output to: {map_file}")

        # Extract summary lines
        for line in output.splitlines():
            if "class_id =" in line or "mean_average_precision" in line or "mAP" in line:
                print("  ", line.strip())
        return output
    except Exception as e:
        print(f"Warning: Darknet mAP via WSL encountered an issue: {e}")
        return None


def run_image_level_evaluation(weights_path: Path, conf_thresh: float = 0.25, nms_thresh: float = 0.45):
    """Evaluate image-level classification on all 215 untouched test images using OpenCV DNN."""
    print("\n" + "=" * 65)
    print("RUNNING IMAGE-LEVEL CLASSIFICATION EVALUATION")
    print("=" * 65)

    cfg_path = PROJECT_ROOT / "cfg" / "yolov4-custom-potato.cfg"
    print(f"Loading network:\n  CFG: {cfg_path}\n  Weights: {weights_path}")

    net = cv2.dnn.readNetFromDarknet(str(cfg_path), str(weights_path))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    output_layer_names = net.getUnconnectedOutLayersNames()

    # Load test manifest
    test_manifest_path = DATA_DIR / "test.txt"
    with open(test_manifest_path, "r", encoding="utf-8") as f:
        test_image_paths = [Path(line.strip()) for line in f if line.strip()]

    print(f"Evaluating {len(test_image_paths)} untouched test images...\n")

    y_true = []
    y_pred = []
    confidences = []
    predictions_record = []
    errors = []

    for img_path in tqdm(test_image_paths, desc="Test Set Inference"):
        # Resolve path
        if not img_path.is_absolute():
            img_path = PROJECT_ROOT / img_path

        # Ground truth from label file
        label_file = img_path.with_suffix(".txt")
        assert label_file.exists(), f"Missing label file: {label_file}"
        label_content = label_file.read_text(encoding="utf-8").strip().splitlines()
        gt_class_id = int(label_content[0].split()[0])
        gt_class_name = CLASS_MAP[gt_class_id]
        y_true.append(gt_class_name)

        # Inference
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Failed to load image: {img_path}")
        h, w = img.shape[:2]

        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layer_names)

        boxes = []
        confs = []
        class_ids = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])
                if confidence > conf_thresh:
                    center_x = int(detection[0] * w)
                    center_y = int(detection[1] * h)
                    width = int(detection[2] * w)
                    height = int(detection[3] * h)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    boxes.append([left, top, width, height])
                    confs.append(confidence)
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confs, conf_thresh, nms_thresh)

        if len(indices) > 0:
            # Flatten indices if needed
            idx_list = indices.flatten() if hasattr(indices, "flatten") else [i[0] for i in indices]
            # Take highest confidence detection
            best_idx = idx_list[np.argmax([confs[i] for i in idx_list])]
            pred_class_id = class_ids[best_idx]
            pred_class_name = CLASS_MAP[pred_class_id]
            pred_conf = confs[best_idx]
            best_box = boxes[best_idx]
        else:
            pred_class_name = "no_detection"
            pred_conf = 0.0
            best_box = None

        y_pred.append(pred_class_name)
        confidences.append(pred_conf)

        record = {
            "image_path": str(img_path),
            "image_name": img_path.name,
            "true_class": gt_class_name,
            "pred_class": pred_class_name,
            "confidence": pred_conf,
            "correct": (gt_class_name == pred_class_name),
        }
        predictions_record.append(record)

        if gt_class_name != pred_class_name:
            errors.append((img_path, gt_class_name, pred_class_name, pred_conf, best_box))

    # Save predictions dataframe
    df_preds = pd.DataFrame(predictions_record)
    df_preds.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)

    # Compute overall accuracy
    acc = accuracy_score(y_true, y_pred)

    # Compute per-class precision, recall, f1, support
    all_eval_labels = CLASS_NAMES.copy()
    if "no_detection" in y_pred:
        all_eval_labels.append("no_detection")

    p, r, f1, s = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_NAMES, zero_division=0
    )

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Build metrics dict (Table 5 format)
    metrics = {
        "overall_accuracy": float(acc),
        "total_test_samples": len(y_true),
        "total_errors": len(errors),
        "macro_average": {
            "precision": float(macro_p),
            "recall": float(macro_r),
            "f1_score": float(macro_f1),
        },
        "weighted_average": {
            "precision": float(weighted_p),
            "recall": float(weighted_r),
            "f1_score": float(weighted_f1),
        },
        "per_class": {},
    }

    for idx, name in enumerate(CLASS_NAMES):
        metrics["per_class"][name] = {
            "precision": float(p[idx]),
            "recall": float(r[idx]),
            "f1_score": float(f1[idx]),
            "support": int(s[idx]),
        }

    metrics_file = RESULTS_DIR / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to: {metrics_file}")

    # Generate Markdown Table 5
    md_table = [
        "# Stage 8 Test Set Evaluation Metrics (Table 5 Reproduction)",
        "",
        f"**Model Weights**: `{weights_path.name}`  ",
        f"**Total Untouched Test Samples**: `{len(y_true)}`  ",
        f"**Overall Classification Accuracy**: **`{acc * 100:.2f}%`**  ",
        f"**Misclassified Images**: `{len(errors)}` / `{len(y_true)}`  ",
        "",
        "| Class | Precision | Recall | F1-Score | Support |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for idx, name in enumerate(CLASS_NAMES):
        md_table.append(
            f"| `{name}` | {p[idx]*100:.2f}% | {r[idx]*100:.2f}% | {f1[idx]*100:.2f}% | {s[idx]} |"
        )
    md_table.append(
        f"| **Macro Avg** | **{macro_p*100:.2f}%** | **{macro_r*100:.2f}%** | **{macro_f1*100:.2f}%** | {len(y_true)} |"
    )
    md_table.append(
        f"| **Weighted Avg** | **{weighted_p*100:.2f}%** | **{weighted_r*100:.2f}%** | **{weighted_f1*100:.2f}%** | {len(y_true)} |"
    )

    md_path = RESULTS_DIR / "test_metrics.md"
    md_path.write_text("\n".join(md_table), encoding="utf-8")
    print(f"Saved Markdown report to: {md_path}")
    print("\n" + "\n".join(md_table))

    # Confusion Matrix Heatmap
    plot_labels = CLASS_NAMES + (["no_detection"] if "no_detection" in y_pred else [])
    cm = confusion_matrix(y_true, y_pred, labels=plot_labels)

    plt.figure(figsize=(8, 6), dpi=300)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=plot_labels,
        yticklabels=plot_labels,
        cbar=True,
    )
    plt.title(f"YOLOv4 Test Set Confusion Matrix (Accuracy: {acc*100:.2f}%)", fontsize=13, weight="bold")
    plt.xlabel("Predicted Class", fontsize=11, weight="bold")
    plt.ylabel("True Ground Truth Class", fontsize=11, weight="bold")
    plt.xticks(rotation=20, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_path = RESULTS_DIR / "confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    print(f"\nSaved confusion matrix heatmap to: {cm_path}")

    # Plot Error Contact Sheet if errors exist
    if errors:
        print(f"\nFound {len(errors)} misclassifications. Creating error contact sheet...")
        n_err = len(errors)
        cols = min(4, n_err)
        rows = int(np.ceil(n_err / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5), squeeze=False, dpi=200)

        for i, (err_img_path, gt, pred, conf, box) in enumerate(errors):
            r_idx, c_idx = divmod(i, cols)
            ax = axes[r_idx, c_idx]
            im = cv2.imread(str(err_img_path))
            im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            if box is not None:
                bx, by, bw, bh = box
                cv2.rectangle(im_rgb, (bx, by), (bx + bw, by + bh), (255, 0, 0), 2)
            ax.imshow(im_rgb)
            ax.set_title(f"GT: {gt}\nPred: {pred} ({conf*100:.1f}%)", fontsize=9, color="red")
            ax.axis("off")

        # Hide any unused subplots
        for j in range(i + 1, rows * cols):
            r_idx, c_idx = divmod(j, cols)
            axes[r_idx, c_idx].axis("off")

        plt.suptitle("YOLOv4 Test Set Misclassified Leaf Images", fontsize=14, weight="bold", color="darkred")
        plt.tight_layout()
        err_path = RESULTS_DIR / "errors.png"
        plt.savefig(err_path)
        plt.close()
        print(f"Saved error gallery to: {err_path}")
    else:
        print("\nPERFECT TEST CLASSIFICATION: 0 errors! Generating perfect score graphic...")
        fig, ax = plt.subplots(figsize=(6, 2), dpi=200)
        ax.text(
            0.5,
            0.5,
            f"100% Test Set Accuracy\n(215 / 215 Correctly Identified)",
            ha="center",
            va="center",
            fontsize=14,
            weight="bold",
            color="forestgreen",
        )
        ax.axis("off")
        err_path = RESULTS_DIR / "errors.png"
        plt.savefig(err_path)
        plt.close()
        print(f"Saved 0-error confirmation to: {err_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv4 Potato Disease Model on Test Set")
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to trained weights file (default: searches backup/last or backup/best)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.15,
        help="Confidence threshold for detection (default: 0.15)",
    )
    args = parser.parse_args()

    # Locate weights
    if args.weights:
        weights_path = Path(args.weights)
    else:
        candidates = [
            PROJECT_ROOT / "weights" / "yolov4-custom-potato_best.weights",
            PROJECT_ROOT / "weights" / "backup" / "yolov4-custom-potato_best.weights",
            PROJECT_ROOT / "weights" / "backup" / "yolov4-custom-potato_last.weights",
            PROJECT_ROOT / "weights" / "backup" / "yolov4-custom-potato_1000.weights",
            PROJECT_ROOT / "weights" / "backup" / "yolov4-custom-potato_2000.weights",
        ]
        weights_path = None
        for cand in candidates:
            if cand.exists():
                weights_path = cand
                break

    if weights_path is None or not weights_path.exists():
        print("ERROR: No trained weights file found in weights/ or weights/backup/!")
        print("Please ensure Stage 7 training has produced at least one checkpoint.")
        sys.exit(1)

    print(f"Found trained weights: {weights_path} ({weights_path.stat().st_size / (1024*1024):.2f} MB)")

    # 1. Darknet test set mAP via WSL
    run_darknet_map(weights_path)

    # 2. Image-level evaluation via OpenCV DNN
    metrics = run_image_level_evaluation(weights_path, conf_thresh=args.conf)
    print("\nStage 8 evaluation complete!")


if __name__ == "__main__":
    main()
