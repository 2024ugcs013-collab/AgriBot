"""
Stage 10: Model Robustness & Perturbation Testing
Evaluates YOLOv4 resilience under adverse environmental conditions
(noise, blur, illumination changes, rotation, hue shifts) on the 215 test images.
"""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_MAP = {0: "potato_early_blight", 1: "potato_late_blight", 2: "potato_healthy"}


def apply_perturbation(img: np.ndarray, pert_type: str) -> np.ndarray:
    """Apply realistic environmental image distortion."""
    perturbed = img.copy()

    if pert_type == "clean":
        return perturbed

    elif pert_type == "gaussian_noise_low":
        noise = np.random.normal(0, 15, perturbed.shape).astype(np.float32)
        perturbed = np.clip(perturbed.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    elif pert_type == "gaussian_noise_high":
        noise = np.random.normal(0, 30, perturbed.shape).astype(np.float32)
        perturbed = np.clip(perturbed.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    elif pert_type == "gaussian_blur_mild":
        perturbed = cv2.GaussianBlur(perturbed, (5, 5), 0)

    elif pert_type == "gaussian_blur_heavy":
        perturbed = cv2.GaussianBlur(perturbed, (9, 9), 0)

    elif pert_type == "high_brightness":
        perturbed = np.clip(perturbed.astype(np.float32) * 1.3 + 20, 0, 255).astype(np.uint8)

    elif pert_type == "low_brightness":
        perturbed = np.clip(perturbed.astype(np.float32) * 0.7 - 20, 0, 255).astype(np.uint8)

    elif pert_type == "high_contrast":
        mean = np.mean(perturbed, axis=(0, 1), keepdims=True)
        perturbed = np.clip((perturbed.astype(np.float32) - mean) * 1.4 + mean, 0, 255).astype(np.uint8)

    elif pert_type == "low_contrast":
        mean = np.mean(perturbed, axis=(0, 1), keepdims=True)
        perturbed = np.clip((perturbed.astype(np.float32) - mean) * 0.6 + mean, 0, 255).astype(np.uint8)

    elif pert_type == "rotation_15deg":
        h, w = perturbed.shape[:2]
        m = cv2.getRotationMatrix2D((w // 2, h // 2), 15, 1.0)
        perturbed = cv2.warpAffine(perturbed, m, (w, h), borderMode=cv2.BORDER_REFLECT)

    elif pert_type == "rotation_minus15deg":
        h, w = perturbed.shape[:2]
        m = cv2.getRotationMatrix2D((w // 2, h // 2), -15, 1.0)
        perturbed = cv2.warpAffine(perturbed, m, (w, h), borderMode=cv2.BORDER_REFLECT)

    elif pert_type == "hue_shift":
        hsv = cv2.cvtColor(perturbed, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + 10) % 180
        perturbed = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return perturbed


def evaluate_condition(net, test_images, pert_type, conf_thresh=0.25, nms_thresh=0.45):
    """Evaluate accuracy on test set under a specific perturbation."""
    out_names = net.getUnconnectedOutLayersNames()
    y_true = []
    y_pred = []

    for img_path in test_images:
        label_file = img_path.with_suffix(".txt")
        gt_cls_id = int(label_file.read_text(encoding="utf-8").strip().splitlines()[0].split()[0])
        gt_cls = CLASS_MAP[gt_cls_id]
        y_true.append(gt_cls)

        img = cv2.imread(str(img_path))
        perturbed = apply_perturbation(img, pert_type)
        h, w = perturbed.shape[:2]

        blob = cv2.dnn.blobFromImage(perturbed, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(out_names)

        boxes = []
        confs = []
        class_ids = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                cid = int(np.argmax(scores))
                conf = float(scores[cid])
                if conf > conf_thresh:
                    boxes.append([0, 0, 10, 10])
                    confs.append(conf)
                    class_ids.append(cid)

        indices = cv2.dnn.NMSBoxes(boxes, confs, conf_thresh, nms_thresh)
        if len(indices) > 0:
            idx_list = indices.flatten() if hasattr(indices, "flatten") else [i[0] for i in indices]
            best_idx = idx_list[np.argmax([confs[i] for i in idx_list])]
            pred_cls = CLASS_MAP[class_ids[best_idx]]
        else:
            pred_cls = "no_detection"

        y_pred.append(pred_cls)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return acc, f1


def main():
    parser = argparse.ArgumentParser(description="YOLOv4 Robustness Testing")
    parser.add_argument("--weights", type=str, default=None)
    args = parser.parse_args()

    candidates = [
        PROJECT_ROOT / "weights" / "yolov4-custom-potato_best.weights",
        PROJECT_ROOT / "weights" / "backup" / "yolov4-custom-potato_best.weights",
        PROJECT_ROOT / "weights" / "backup" / "yolov4-custom-potato_last.weights",
        PROJECT_ROOT / "weights" / "backup" / "yolov4-custom-potato_1000.weights",
    ]
    weights_path = Path(args.weights) if args.weights else None
    if weights_path is None:
        for cand in candidates:
            if cand.exists():
                weights_path = cand
                break
    assert weights_path and weights_path.exists(), "No valid weights file found."

    cfg_path = PROJECT_ROOT / "cfg" / "yolov4-custom-potato.cfg"
    net = cv2.dnn.readNetFromDarknet(str(cfg_path), str(weights_path))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    test_manifest = DATA_DIR / "test.txt"
    with open(test_manifest, "r", encoding="utf-8") as f:
        test_images = [PROJECT_ROOT / line.strip() for line in f if line.strip()]

    conditions = [
        ("Clean Baseline", "clean"),
        ("Gaussian Noise (Low, sigma=15)", "gaussian_noise_low"),
        ("Gaussian Noise (High, sigma=30)", "gaussian_noise_high"),
        ("Gaussian Blur (Mild, 5x5)", "gaussian_blur_mild"),
        ("Gaussian Blur (Heavy, 9x9)", "gaussian_blur_heavy"),
        ("High Brightness (+30%)", "high_brightness"),
        ("Low Brightness (-30%)", "low_brightness"),
        ("High Contrast (x1.4)", "high_contrast"),
        ("Low Contrast (x0.6)", "low_contrast"),
        ("Rotation (+15 deg)", "rotation_15deg"),
        ("Rotation (-15 deg)", "rotation_minus15deg"),
        ("HSV Hue Shift (+10 deg)", "hue_shift"),
    ]

    print("=" * 65)
    print("STAGE 10: ROBUSTNESS & ENVIRONMENTAL PERTURBATION TESTING")
    print(f"Testing {len(conditions)} conditions across {len(test_images)} images each...")
    print("=" * 65)

    results = []
    baseline_acc = None

    for label, code in conditions:
        acc, f1 = evaluate_condition(net, test_images, code)
        if code == "clean":
            baseline_acc = acc
        drop = (baseline_acc - acc) * 100 if baseline_acc is not None else 0.0

        results.append({
            "Condition": label,
            "Accuracy": acc * 100,
            "Macro F1": f1 * 100,
            "Accuracy Drop (%)": drop,
        })
        print(f"  {label:<35} | Acc: {acc*100:6.2f}% | F1: {f1*100:6.2f}% | Drop: {drop:5.2f}%")

    df_rob = pd.DataFrame(results)
    csv_path = RESULTS_DIR / "robustness.csv"
    df_rob.to_csv(csv_path, index=False)
    print(f"\nSaved robustness results to: {csv_path}")

    # Plot Robustness Bar Chart
    plt.figure(figsize=(10, 6), dpi=300)
    colors = ["#2ecc71" if c == "Clean Baseline" else "#3498db" if d < 5 else "#e67e22" if d < 15 else "#e74c3c"
              for c, d in zip(df_rob["Condition"], df_rob["Accuracy Drop (%)"])]
    bars = plt.barh(df_rob["Condition"], df_rob["Accuracy"], color=colors, edgecolor="black", height=0.6)
    plt.axvline(x=baseline_acc * 100, color="gray", linestyle="--", alpha=0.7, label=f"Baseline ({baseline_acc*100:.1f}%)")
    plt.xlim(0, 105)
    plt.xlabel("Test Classification Accuracy (%)", fontsize=11, weight="bold")
    plt.title("YOLOv4 Model Robustness Under Environmental Perturbations", fontsize=13, weight="bold")
    plt.gca().invert_yaxis()

    for bar in bars:
        w = bar.get_width()
        plt.text(w + 1.0, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", fontsize=9, weight="bold")

    plt.tight_layout()
    plot_path = RESULTS_DIR / "robustness.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved robustness plot to: {plot_path}")


if __name__ == "__main__":
    main()
