"""
Potato Leaf Disease Detection with YOLOv4
Stage 3: Leak-Free Train/Validation/Test Split & Augmentation

Features:
1. Stratified Group K-Fold split ensuring duplicate leaves never cross splits.
2. 90% (Train+Val) / 10% Test, then 90% Train / 10% Val.
3. Split manifest saved to data/manifests/split.csv.
4. Copy original images into flat split directories with class-prefixed filenames.
5. Apply luminance-based histogram equalization and horizontal flips on TRAIN only.
6. Triple-augmentation on potato_healthy class (HE, Flip, HE+Flip).
7. Generate results/augmentation_demo.png and verification table.
"""

import shutil
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MANIFESTS_DIR = PROJECT_ROOT / "data" / "manifests"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

AUDIT_CSV = MANIFESTS_DIR / "audit.csv"
SPLIT_CSV = MANIFESTS_DIR / "split.csv"

def split_and_augment():
    print("=" * 60)
    print("STAGE 3: LEAK-FREE SPLIT & TARGETED AUGMENTATION")
    print("=" * 60)

    if not AUDIT_CSV.exists():
        raise FileNotFoundError(f"Missing audit CSV: {AUDIT_CSV}. Run Stage 2 first.")

    df = pd.read_csv(AUDIT_CSV)
    print(f"Loaded audit manifest with {len(df):,} images.")

    # ---------------------------------------------------------
    # 1. Stratified Group Split (Seed 42)
    # ---------------------------------------------------------
    # Step 1A: 90% Train+Val vs 10% Test
    sgkf_test = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
    trainval_idx, test_idx = next(sgkf_test.split(df, df["class"], df["dup_group"]))

    test_df = df.iloc[test_idx].copy()
    trainval_df = df.iloc[trainval_idx].copy()

    # Step 1B: 90% Train vs 10% Val from Train+Val
    sgkf_val = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
    train_idx, val_idx = next(sgkf_val.split(trainval_df, trainval_df["class"], trainval_df["dup_group"]))

    train_df = trainval_df.iloc[train_idx].copy()
    val_df = trainval_df.iloc[val_idx].copy()

    # Assign split labels
    train_df["split"] = "train"
    val_df["split"] = "valid"
    test_df["split"] = "test"

    split_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # ---------------------------------------------------------
    # 2. Strict Leak-Free & Disjoint Assertions
    # ---------------------------------------------------------
    print("\nRunning anti-leakage assertions...")
    train_paths = set(train_df["path"])
    val_paths = set(val_df["path"])
    test_paths = set(test_df["path"])

    assert len(train_paths.intersection(val_paths)) == 0, "ERROR: Image path leaked between train and valid!"
    assert len(train_paths.intersection(test_paths)) == 0, "ERROR: Image path leaked between train and test!"
    assert len(val_paths.intersection(test_paths)) == 0, "ERROR: Image path leaked between valid and test!"

    train_groups = set(train_df["dup_group"])
    val_groups = set(val_df["dup_group"])
    test_groups = set(test_df["dup_group"])

    assert len(train_groups.intersection(val_groups)) == 0, "ERROR: Duplicate group leaked between train and valid!"
    assert len(train_groups.intersection(test_groups)) == 0, "ERROR: Duplicate group leaked between train and test!"
    assert len(val_groups.intersection(test_groups)) == 0, "ERROR: Duplicate group leaked between valid and test!"

    print("Assertions PASSED: 0 path overlaps and 0 duplicate-group leaks.")

    # Save split manifest
    out_cols = ["path", "class", "split", "dup_group"]
    split_df[out_cols].to_csv(SPLIT_CSV, index=False)
    print(f"Saved leak-free split manifest to: {SPLIT_CSV}")

    # Print split counts table
    pivot_counts = pd.crosstab(split_df["class"], split_df["split"], margins=True)
    print("\n--- Original Image Counts by Class & Split ---")
    print(pivot_counts.to_string())

    # ---------------------------------------------------------
    # 3. Copy Images to Flat Split Folders with Class Prefix
    # ---------------------------------------------------------
    train_dir = PROCESSED_DIR / "train"
    val_dir = PROCESSED_DIR / "valid"
    test_dir = PROCESSED_DIR / "test"

    for d in [train_dir, val_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)
        # Clear existing files to prevent stale images
        for f in d.glob("*.*"):
            if f.is_file():
                f.unlink()

    print("\nCopying original images into flat processed split directories...")
    # Map for processed files to track augmentation
    copied_records = []

    for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc="Copying images"):
        src_path = PROJECT_ROOT / row["path"]
        class_name = row["class"]
        split_name = row["split"]
        target_dir = PROCESSED_DIR / split_name

        # Clean sanitized stem
        stem = src_path.stem.replace(" ", "_")
        dest_filename = f"{class_name}_{stem}.jpg"
        dest_path = target_dir / dest_filename

        shutil.copy2(src_path, dest_path)
        copied_records.append({
            "dest_path": dest_path,
            "filename": dest_filename,
            "class": class_name,
            "split": split_name,
            "stem": stem
        })

    # ---------------------------------------------------------
    # 4. Augment TRAIN Split Only
    # ---------------------------------------------------------
    print("\nAugmenting training set (Histogram Equalization & Horizontal Flips)...")
    train_records = [r for r in copied_records if r["split"] == "train"]
    demo_images = None

    for r in tqdm(train_records, desc="Augmenting train split"):
        img_bgr = cv2.imread(str(r["dest_path"]))
        if img_bgr is None:
            raise IOError(f"Could not read image: {r['dest_path']}")

        base_stem = f"{r['class']}_{r['stem']}"

        # 4a: Histogram Equalization on Luminance (Y) channel in YCrCb color space
        ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        img_he = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        he_path = train_dir / f"{base_stem}_he.jpg"
        cv2.imwrite(str(he_path), img_he)

        # 4b: Horizontal Flip
        img_flip = cv2.flip(img_bgr, 1)
        flip_path = train_dir / f"{base_stem}_flip.jpg"
        cv2.imwrite(str(flip_path), img_flip)

        # 4c: Additional Flipped HE for healthy minority class
        img_he_flip = None
        if r["class"] == "potato_healthy":
            img_he_flip = cv2.flip(img_he, 1)
            he_flip_path = train_dir / f"{base_stem}_he_flip.jpg"
            cv2.imwrite(str(he_flip_path), img_he_flip)

        # Save sample for 2x2 visualization demo
        if demo_images is None and r["class"] == "potato_early_blight":
            demo_images = {
                "Original": cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                "Y-Channel HE": cv2.cvtColor(img_he, cv2.COLOR_BGR2RGB),
                "Horizontal Flip": cv2.cvtColor(img_flip, cv2.COLOR_BGR2RGB),
                "HE + Flip": cv2.cvtColor(cv2.flip(img_he, 1), cv2.COLOR_BGR2RGB)
            }

    # ---------------------------------------------------------
    # 5. Final Split Inventory & Verification
    # ---------------------------------------------------------
    print("\nCounting final processed dataset files...")
    final_stats = []
    for s_name, s_dir in [("train", train_dir), ("valid", val_dir), ("test", test_dir)]:
        files = list(s_dir.glob("*.jpg"))
        for f in files:
            for cls_name in ["potato_early_blight", "potato_late_blight", "potato_healthy"]:
                if f.name.startswith(cls_name):
                    final_stats.append({"split": s_name, "class": cls_name, "filename": f.name})
                    break

    final_df = pd.DataFrame(final_stats)
    final_pivot = pd.crosstab(final_df["class"], final_df["split"], margins=True)
    print("\n--- Final Image Counts per Split (Post-Augmentation) ---")
    print(final_pivot.to_string())

    # ---------------------------------------------------------
    # 6. Generate 2x2 Augmentation Demo
    # ---------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    keys = ["Original", "Y-Channel HE", "Horizontal Flip", "HE + Flip"]
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for k, (r, c) in zip(keys, positions):
        ax = axes[r, c]
        ax.imshow(demo_images[k])
        ax.set_title(k, fontsize=12, fontweight="bold")
        ax.axis("off")

    plt.suptitle("Paper Augmentation Techniques (Aldakheel et al. 2024)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    demo_path = RESULTS_DIR / "augmentation_demo.png"
    plt.savefig(demo_path, dpi=200)
    plt.close()
    print(f"\nSaved augmentation demo figure to: {demo_path}")

    # Assertions on test and valid: MUST contain NO augmented files
    for split_dir, split_label in [(val_dir, "valid"), (test_dir, "test")]:
        aug_files = [
            f for f in split_dir.glob("*.jpg")
            if f.stem.endswith("_he") or f.stem.endswith("_flip") or f.stem.endswith("_he_flip")
        ]
        assert len(aug_files) == 0, f"ERROR: Found augmented files in untouched {split_label} split!"

    print(f"Validation and Test split purity confirmed: 0 augmented images in valid or test.")

    print("\n" + "=" * 60)
    print("STAGE 3 COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    split_and_augment()
