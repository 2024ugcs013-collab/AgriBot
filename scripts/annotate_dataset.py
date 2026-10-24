"""
Potato Leaf Disease Detection with YOLOv4
Stage 4: Automated YOLO Bounding Box Annotation

Features:
1. Automated leaf localization using HSV/Otsu thresholding + GrabCut fallback + 90% centered fallback.
2. Derivation of exact labels for augmented images:
   - HE: unchanged spatial label.
   - Horizontal Flip / HE-Flip: x_center = 1.0 - x_center.
3. Normalized YOLO format: <class_id> <x_center> <y_center> <width> <height> (0..1, 6 decimals).
4. Validation assertions across 100% of train, valid, and test images.
5. Contact sheet generation (results/annotation_check.png) with green box and red text.
6. Extreme box area audit (smallest & largest 10 boxes).
"""

from pathlib import Path
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

random.seed(42)
np.random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

CLASS_TO_ID = {
    "potato_early_blight": 0,
    "potato_late_blight": 1,
    "potato_healthy": 2
}
ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}

def detect_leaf_bbox(img_bgr):
    """
    Detects the leaf bounding box using a multi-tiered fallback strategy:
    1. HSV thresholding (organic foliage colors)
    2. Grayscale Otsu thresholding
    3. GrabCut segmentation
    4. Centered 90% fallback box
    Returns: (x_center, y_center, width, height, method_name) normalized to [0, 1]
    """
    h, w = img_bgr.shape[:2]
    total_area = float(w * h)

    # --- Tier 1: HSV thresholding ---
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Range covers green leaves, yellowing halos, and necrotic brown blight lesions
    lower_leaf = np.array([10, 30, 20])
    upper_leaf = np.array([95, 255, 245])
    mask_hsv = cv2.inRange(hsv, lower_leaf, upper_leaf)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, kernel, iterations=2)
    cnts, _ = cv2.findContours(mask_hsv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if cnts:
        c = max(cnts, key=cv2.contourArea)
        area_ratio = cv2.contourArea(c) / total_area
        if area_ratio >= 0.20:
            bx, by, bw, bh = cv2.boundingRect(c)
            # Clip within image bounds
            bx = max(0, bx)
            by = max(0, by)
            bw = min(w - bx, bw)
            bh = min(h - by, bh)
            xc = (bx + bw / 2.0) / w
            yc = (by + bh / 2.0) / h
            return xc, yc, bw / w, bh / h, "HSV_contour"

    # --- Tier 2: Grayscale Otsu ---
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask_otsu = cv2.morphologyEx(mask_otsu, cv2.MORPH_CLOSE, kernel, iterations=2)
    cnts, _ = cv2.findContours(mask_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if cnts:
        c = max(cnts, key=cv2.contourArea)
        area_ratio = cv2.contourArea(c) / total_area
        if area_ratio >= 0.20:
            bx, by, bw, bh = cv2.boundingRect(c)
            bx = max(0, bx)
            by = max(0, by)
            bw = min(w - bx, bw)
            bh = min(h - by, bh)
            xc = (bx + bw / 2.0) / w
            yc = (by + bh / 2.0) / h
            return xc, yc, bw / w, bh / h, "Otsu_contour"

    # --- Tier 3: GrabCut ---
    try:
        mask_gc = np.zeros((h, w), np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        # Initialize rectangle inside border
        margin_w = int(w * 0.05)
        margin_h = int(h * 0.05)
        rect = (margin_w, margin_h, w - 2 * margin_w, h - 2 * margin_h)
        cv2.grabCut(img_bgr, mask_gc, rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_RECT)
        fg_mask = np.where((mask_gc == 2) | (mask_gc == 0), 0, 1).astype('uint8')
        cnts, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(c) / total_area >= 0.20:
                bx, by, bw, bh = cv2.boundingRect(c)
                xc = (bx + bw / 2.0) / w
                yc = (by + bh / 2.0) / h
                return xc, yc, bw / w, bh / h, "GrabCut"
    except Exception:
        pass

    # --- Tier 4: Centered 90% Fallback ---
    return 0.500000, 0.500000, 0.900000, 0.900000, "Fallback_90pct"

def annotate_dataset():
    print("=" * 60)
    print("STAGE 4: BOUNDING BOX ANNOTATION IN YOLO FORMAT")
    print("=" * 60)

    splits = ["train", "valid", "test"]
    method_counts = {}
    box_records = []

    # Map of original base stems to detected boxes
    # key: (split, base_stem) -> (xc, yc, w, h, class_id)
    base_box_cache = {}

    # Step 1: Detect bounding boxes for all original images across splits
    print("Detecting bounding boxes on original images...")
    for split in splits:
        split_dir = PROCESSED_DIR / split
        img_paths = sorted(list(split_dir.glob("*.jpg")))
        
        # Original images do not end with _he.jpg, _flip.jpg, or _he_flip.jpg
        orig_paths = [
            p for p in img_paths 
            if not (p.stem.endswith("_he") or p.stem.endswith("_flip") or p.stem.endswith("_he_flip"))
        ]

        for p in tqdm(orig_paths, desc=f"Annotating {split} originals"):
            # Determine class from filename prefix
            class_id = None
            for cls_name, cid in CLASS_TO_ID.items():
                if p.name.startswith(cls_name):
                    class_id = cid
                    break
            assert class_id is not None, f"Could not determine class for {p.name}"

            img_bgr = cv2.imread(str(p))
            xc, yc, bw, bh, method = detect_leaf_bbox(img_bgr)
            method_counts[method] = method_counts.get(method, 0) + 1

            # Clamp coordinates to [0, 1]
            xc = float(np.clip(xc, 0.001, 0.999))
            yc = float(np.clip(yc, 0.001, 0.999))
            bw = float(np.clip(bw, 0.001, 1.0))
            bh = float(np.clip(bh, 0.001, 1.0))

            base_box_cache[(split, p.stem)] = (xc, yc, bw, bh, class_id, method)

            # Write label for original image
            txt_path = p.with_suffix(".txt")
            txt_path.write_text(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

            box_records.append({
                "image": p.name,
                "split": split,
                "class": ID_TO_CLASS[class_id],
                "xc": xc,
                "yc": yc,
                "w": bw,
                "h": bh,
                "area": bw * bh,
                "method": method
            })

    # Step 2: Propagate exact geometric labels to augmented images in train
    print("\nPropagating geometric labels to augmented training images...")
    train_dir = PROCESSED_DIR / "train"
    aug_paths = sorted([
        p for p in train_dir.glob("*.jpg") 
        if p.stem.endswith("_he") or p.stem.endswith("_flip") or p.stem.endswith("_he_flip")
    ])

    for p in tqdm(aug_paths, desc="Labeling augmented images"):
        stem = p.stem
        if stem.endswith("_he_flip"):
            orig_stem = stem[:-8] # remove _he_flip
            is_flip = True
        elif stem.endswith("_flip"):
            orig_stem = stem[:-5] # remove _flip
            is_flip = True
        elif stem.endswith("_he"):
            orig_stem = stem[:-3] # remove _he
            is_flip = False
        else:
            raise ValueError(f"Unexpected filename: {stem}")

        key = ("train", orig_stem)
        assert key in base_box_cache, f"Missing original entry for augmented file: {p.name}"
        orig_xc, orig_yc, orig_w, orig_h, class_id, method = base_box_cache[key]

        if is_flip:
            # Horizontal reflection: center mirrors across vertical axis
            xc = 1.0 - orig_xc
            yc = orig_yc
        else:
            # Histogram equalization: identical spatial layout
            xc = orig_xc
            yc = orig_yc

        bw = orig_w
        bh = orig_h

        # Clamp and write
        xc = float(np.clip(xc, 0.001, 0.999))
        yc = float(np.clip(yc, 0.001, 0.999))
        txt_path = p.with_suffix(".txt")
        txt_path.write_text(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

        box_records.append({
            "image": p.name,
            "split": "train",
            "class": ID_TO_CLASS[class_id],
            "xc": xc,
            "yc": yc,
            "w": bw,
            "h": bh,
            "area": bw * bh,
            "method": method + "_propagated"
        })

    # ---------------------------------------------------------
    # Step 3: Comprehensive Integrity Assertions
    # ---------------------------------------------------------
    print("\nRunning strict label validation checks...")
    total_images = 0
    total_labels = 0

    for split in splits:
        split_dir = PROCESSED_DIR / split
        imgs = sorted(list(split_dir.glob("*.jpg")))
        txts = sorted(list(split_dir.glob("*.txt")))

        total_images += len(imgs)
        total_labels += len(txts)

        assert len(imgs) == len(txts), f"Mismatch in {split}: {len(imgs)} images vs {len(txts)} labels!"

        for img in imgs:
            txt = img.with_suffix(".txt")
            assert txt.exists(), f"Missing label file for {img.name}"
            content = txt.read_text().strip().split()
            assert len(content) == 5, f"Malformed label file in {txt.name}: '{content}'"
            cid = int(content[0])
            xc = float(content[1])
            yc = float(content[2])
            w = float(content[3])
            h = float(content[4])

            assert cid in [0, 1, 2], f"Invalid class ID {cid} in {txt.name}"
            assert 0.0 < xc < 1.0, f"xc out of bounds: {xc} in {txt.name}"
            assert 0.0 < yc < 1.0, f"yc out of bounds: {yc} in {txt.name}"
            assert 0.0 < w <= 1.0, f"w out of bounds: {w} in {txt.name}"
            assert 0.0 < h <= 1.0, f"h out of bounds: {h} in {txt.name}"

    print(f"Validation PASSED: All {total_images:,} images have exactly one valid YOLO label file.")

    # ---------------------------------------------------------
    # Step 4: Method Counts & Outlier Analysis
    # ---------------------------------------------------------
    print("\n--- Detection Method Summary (on 2,152 unique specimens) ---")
    for method, cnt in method_counts.items():
        print(f"  - {method}: {cnt:,} images ({cnt / 2152 * 100:.2f}%)")

    box_df = pd.DataFrame(box_records)
    # Filter to original images for outlier audit
    orig_box_df = box_df[~box_df["method"].str.contains("propagated")].copy()
    
    smallest_10 = orig_box_df.sort_values("area").head(10)
    largest_10 = orig_box_df.sort_values("area", ascending=False).head(10)

    print("\n--- 10 Smallest Detected Boxes (Review for under-segmentation) ---")
    for _, row in smallest_10.iterrows():
        print(f"  {row['image']:<60} | Area: {row['area']:.3f} | Class: {row['class']}")

    print("\n--- 10 Largest Detected Boxes (Full frame / extensive leaves) ---")
    for _, row in largest_10.iterrows():
        print(f"  {row['image']:<60} | Area: {row['area']:.3f} | Class: {row['class']}")

    # ---------------------------------------------------------
    # Step 5: Generate 30-Image Visual Contact Sheet
    # ---------------------------------------------------------
    print("\nGenerating 30-image contact sheet (results/annotation_check.png)...")
    contact_samples = []
    # Pick 10 random images per class across splits
    for cid, cname in ID_TO_CLASS.items():
        cls_rows = box_df[box_df["class"] == cname]
        sampled = cls_rows.sample(10, random_state=42)
        contact_samples.extend(sampled.to_dict("records"))

    # Create 5 rows x 6 columns contact sheet
    fig, axes = plt.subplots(5, 6, figsize=(18, 15))
    axes = axes.flatten()

    for idx, sample in enumerate(contact_samples):
        split = sample["split"]
        img_name = sample["image"]
        img_path = PROCESSED_DIR / split / img_name
        
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        xc = sample["xc"]
        yc = sample["yc"]
        bw = sample["w"]
        bh = sample["h"]

        # Calculate bounding box coordinates
        x1 = int((xc - bw / 2.0) * w)
        y1 = int((yc - bh / 2.0) * h)
        x2 = int((xc + bw / 2.0) * w)
        y2 = int((yc + bh / 2.0) * h)

        # Draw green bounding box (as in paper Figures 4, 6, 10)
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw red label text
        class_label = sample["class"].replace("potato_", "")
        cv2.putText(
            img_rgb, 
            class_label, 
            (max(4, x1), max(18, y1 - 4)), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (255, 0, 0), 
            1, 
            cv2.LINE_AA
        )

        ax = axes[idx]
        ax.imshow(img_rgb)
        ax.set_title(f"ID:{CLASS_TO_ID[sample['class']]} [{sample['split']}]\n{img_name[:18]}...", fontsize=8)
        ax.axis("off")

    plt.suptitle("YOLO Annotation Verification Sheet (Green: Leaf Box, Red: Class Label)", fontsize=16, fontweight="bold")
    plt.tight_layout()
    contact_path = RESULTS_DIR / "annotation_check.png"
    plt.savefig(contact_path, dpi=200)
    plt.close()
    print(f"Saved contact sheet to: {contact_path}")

    print("\n" + "=" * 60)
    print("STAGE 4 COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    annotate_dataset()
