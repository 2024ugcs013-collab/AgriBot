"""
Potato Leaf Disease Detection with YOLOv4
Stage 2: Dataset Acquisition and Audit Script

Performs:
1. Verification of all images in data/raw
2. Resolution and format checks
3. Perceptual hashing (imagehash.phash) and grouping (Hamming distance <= 5)
4. Audit CSV generation (data/manifests/audit.csv)
5. Visualization: Class distribution chart and 3x3 sample grid
"""

import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import imagehash
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MANIFESTS_DIR = PROJECT_ROOT / "data" / "manifests"
RESULTS_DIR = PROJECT_ROOT / "results"

MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_MAP = {
    "Potato___Early_blight": "potato_early_blight",
    "Potato___Late_blight": "potato_late_blight",
    "Potato___healthy": "potato_healthy"
}

def audit_dataset():
    print("=" * 60)
    print("STAGE 2: DATASET AUDIT & PERCEPTUAL HASHING")
    print("=" * 60)

    records = []
    corrupt_files = []
    sizes = set()
    formats = set()

    all_image_paths = []
    for folder_name in CLASS_MAP.keys():
        folder_path = RAW_DIR / folder_name
        if not folder_path.exists():
            raise FileNotFoundError(f"Missing folder: {folder_path}")
        files = [p for p in folder_path.iterdir() if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        all_image_paths.extend(files)

    print(f"Total raw image files found: {len(all_image_paths):,}")

    hash_list = []
    
    print("\nAuditing image integrity, dimensions, and computing pHash...")
    for p in tqdm(all_image_paths, desc="Auditing images"):
        folder_name = p.parent.name
        class_name = CLASS_MAP[folder_name]

        try:
            with Image.open(p) as img:
                img.verify()
            
            # Reopen to read dimensions and compute hash (verify closes the image stream)
            with Image.open(p) as img:
                w, h = img.size
                fmt = img.format
                sizes.add((w, h))
                formats.add(fmt)
                ph = imagehash.phash(img)
                # Convert phash to 64-bit boolean array
                hash_bits = np.array(ph.hash, dtype=bool).flatten()
                hash_list.append(hash_bits)

            # Store relative path with POSIX forward slashes for cross-platform Darknet compatibility
            rel_path = p.relative_to(PROJECT_ROOT).as_posix()
            records.append({
                "path": rel_path,
                "class": class_name,
                "width": w,
                "height": h,
                "phash": str(ph)
            })

        except Exception as e:
            print(f"Corrupt file detected: {p} ({e})")
            corrupt_files.append((str(p), str(e)))

    if corrupt_files:
        print(f"WARNING: Found {len(corrupt_files)} corrupt files!")
    else:
        print("Image integrity check PASSED: 0 corrupt files found.")

    print(f"Image dimensions detected: {sizes}")
    print(f"Image formats detected: {formats}")

    # Build DataFrame
    df = pd.DataFrame(records)

    # -------------------------------------------------------------
    # Group Near-Duplicates (Hamming distance <= 5)
    # -------------------------------------------------------------
    print("\nDetecting near-duplicate groups using pHash (Hamming distance <= 5)...")
    hash_matrix = np.array(hash_list) # Shape: (N, 64)
    N = len(hash_matrix)

    # Compute pairwise Hamming distance efficiently in chunks
    # distance = sum(hash_i != hash_j)
    # Using boolean XOR / !=
    adj_rows = []
    adj_cols = []

    # Vectorized chunked pairwise comparison to save memory and maximize speed
    chunk_size = 500
    for i_start in range(0, N, chunk_size):
        i_end = min(i_start + chunk_size, N)
        # Shape: (chunk, 1, 64) != (1, N, 64) -> (chunk, N)
        chunk_diff = np.count_nonzero(hash_matrix[i_start:i_end, None, :] != hash_matrix[None, :, :], axis=-1)
        matches = np.argwhere(chunk_diff <= 5)
        for r, c in matches:
            adj_rows.append(i_start + r)
            adj_cols.append(c)

    # Connected components finds all transitive duplicate groups
    adj_data = np.ones(len(adj_rows), dtype=bool)
    graph = csr_matrix((adj_data, (adj_rows, adj_cols)), shape=(N, N))
    n_components, labels = connected_components(graph, directed=False)

    df["dup_group"] = labels
    
    # Save audit manifest
    audit_csv = MANIFESTS_DIR / "audit.csv"
    df.to_csv(audit_csv, index=False)
    print(f"Audit manifest saved to: {audit_csv}")

    # Duplicate statistics
    group_counts = df["dup_group"].value_counts()
    multi_image_groups = group_counts[group_counts > 1]
    total_images_in_dups = group_counts[group_counts > 1].sum()

    print("\n--- Duplicate Audit Summary ---")
    print(f"Total images: {len(df)}")
    print(f"Unique duplicate groups: {n_components}")
    print(f"Clusters containing >= 2 near-duplicate leaves: {len(multi_image_groups)}")
    print(f"Total images belonging to duplicate clusters: {total_images_in_dups}")
    print(f"Largest single duplicate cluster: {group_counts.max()} images")

    # -------------------------------------------------------------
    # Visualizations
    # -------------------------------------------------------------
    print("\nGenerating class distribution chart...")
    class_counts = df["class"].value_counts()
    print("Class counts:")
    for cls_name, cnt in class_counts.items():
        print(f"  - {cls_name}: {cnt} images ({cnt / len(df) * 100:.2f}%)")

    plt.figure(figsize=(8, 5))
    palette = {"potato_early_blight": "#e74c3c", "potato_late_blight": "#8e44ad", "potato_healthy": "#27ae60"}
    sns.barplot(x=class_counts.index, y=class_counts.values, palette=palette)
    plt.title("PlantVillage Potato Class Distribution (Raw Dataset)", fontsize=14, fontweight="bold")
    plt.ylabel("Number of Images", fontsize=12)
    plt.xlabel("Class", fontsize=12)
    for idx, (cls_name, cnt) in enumerate(class_counts.items()):
        plt.text(idx, cnt + 25, f"{cnt:,}", ha="center", fontweight="bold", fontsize=11)
    plt.ylim(0, max(class_counts.values) + 150)
    plt.tight_layout()
    dist_chart_path = RESULTS_DIR / "class_distribution.png"
    plt.savefig(dist_chart_path, dpi=200)
    plt.close()
    print(f"Saved: {dist_chart_path}")

    # Sample Grid: 3 random images per class (3x3 grid)
    print("Generating 3x3 sample image grid...")
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    classes = ["potato_early_blight", "potato_late_blight", "potato_healthy"]

    for row_idx, cls_name in enumerate(classes):
        cls_df = df[df["class"] == cls_name]
        sample_paths = cls_df.sample(3, random_state=42)["path"].tolist()
        for col_idx, rel_p in enumerate(sample_paths):
            full_p = PROJECT_ROOT / rel_p
            img = Image.open(full_p).convert("RGB")
            ax = axes[row_idx, col_idx]
            ax.imshow(img)
            ax.set_title(f"{cls_name}\n({full_p.name})", fontsize=9)
            ax.axis("off")

    plt.suptitle("Sample Leaf Images per Class (PlantVillage Raw Subset)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    grid_path = RESULTS_DIR / "sample_grid.png"
    plt.savefig(grid_path, dpi=200)
    plt.close()
    print(f"Saved: {grid_path}")

    print("\n" + "=" * 60)
    print("STAGE 2 AUDIT COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    audit_dataset()
