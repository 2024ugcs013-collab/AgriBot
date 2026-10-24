"""
Potato Leaf Disease Detection with YOLOv4
Stage 5: Generate Darknet Data Manifests and Config Files

Outputs:
1. data/obj.names: 3 class names
2. data/train.txt: Absolute paths to all 5,352 training images (including augmentations)
3. data/valid.txt: Absolute paths to all 194 validation images
4. data/test.txt:  Absolute paths to all 215 test images
5. data/obj.data:  Darknet data descriptor pointing to obj.names, train.txt, valid.txt, and weights/backup
6. Comprehensive verification asserting every image path exists and has a matching label file.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
WEIGHTS_BACKUP_DIR = PROJECT_ROOT / "weights" / "backup"

WEIGHTS_BACKUP_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = [
    "potato_early_blight",
    "potato_late_blight",
    "potato_healthy"
]

def create_darknet_files():
    print("=" * 60)
    print("STAGE 5: CREATING DARKNET DATA FILES")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Create obj.names
    # ---------------------------------------------------------
    obj_names_path = DATA_DIR / "obj.names"
    with open(obj_names_path, "w", encoding="utf-8") as f:
        for cls_name in CLASSES:
            f.write(f"{cls_name}\n")
    print(f"Created: {obj_names_path}")

    # ---------------------------------------------------------
    # 2. Create train.txt, valid.txt, test.txt
    # ---------------------------------------------------------
    splits = {
        "train": DATA_DIR / "train.txt",
        "valid": DATA_DIR / "valid.txt",
        "test": DATA_DIR / "test.txt"
    }

    # Also generate WSL-formatted txt files for seamless WSL2 Darknet execution
    splits_wsl = {
        "train": DATA_DIR / "train_wsl.txt",
        "valid": DATA_DIR / "valid_wsl.txt",
        "test": DATA_DIR / "test_wsl.txt"
    }

    # Helper to convert Windows path to WSL path
    def to_wsl_path(p: Path) -> str:
        # e.g. C:/Users/... -> /mnt/c/Users/...
        parts = p.resolve().as_posix().split(":")
        drive_letter = parts[0].lower()
        return f"/mnt/{drive_letter}{parts[1]}"

    counts = {}

    for split_name, txt_file in splits.items():
        split_dir = PROCESSED_DIR / split_name
        img_paths = sorted(list(split_dir.glob("*.jpg")))
        counts[split_name] = len(img_paths)

        # Write Windows / standard forward-slash paths
        with open(txt_file, "w", encoding="utf-8") as f:
            for p in img_paths:
                f.write(f"{p.resolve().as_posix()}\n")

        # Write WSL paths
        wsl_file = splits_wsl[split_name]
        with open(wsl_file, "w", encoding="utf-8") as f:
            for p in img_paths:
                f.write(f"{to_wsl_path(p)}\n")

        print(f"Created: {txt_file} ({len(img_paths):,} images)")
        print(f"Created: {wsl_file} (WSL format)")

    # ---------------------------------------------------------
    # 3. Create obj.data (Standard and WSL)
    # ---------------------------------------------------------
    obj_data_path = DATA_DIR / "obj.data"
    obj_data_content = (
        f"classes = {len(CLASSES)}\n"
        f"train = {splits['train'].resolve().as_posix()}\n"
        f"valid = {splits['valid'].resolve().as_posix()}\n"
        f"names = {obj_names_path.resolve().as_posix()}\n"
        f"backup = {WEIGHTS_BACKUP_DIR.resolve().as_posix()}\n"
    )
    obj_data_path.write_text(obj_data_content, encoding="utf-8")
    print(f"\nCreated: {obj_data_path}")

    # WSL obj.data
    obj_data_wsl_path = DATA_DIR / "obj_wsl.data"
    obj_data_wsl_content = (
        f"classes = {len(CLASSES)}\n"
        f"train = {to_wsl_path(splits_wsl['train'])}\n"
        f"valid = {to_wsl_path(splits_wsl['valid'])}\n"
        f"names = {to_wsl_path(obj_names_path)}\n"
        f"backup = {to_wsl_path(WEIGHTS_BACKUP_DIR)}\n"
    )
    obj_data_wsl_path.write_text(obj_data_wsl_content, encoding="utf-8")
    print(f"Created: {obj_data_wsl_path}")

    # ---------------------------------------------------------
    # 4. Verification Check
    # ---------------------------------------------------------
    print("\nRunning path and label verification checks...")
    problems = []

    for split_name, txt_file in splits.items():
        with open(txt_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        for line in lines:
            img_p = Path(line)
            if not img_p.exists():
                problems.append(f"Image not found: {line}")
            label_p = img_p.with_suffix(".txt")
            if not label_p.exists():
                problems.append(f"Matching label not found for: {line}")

    if problems:
        print(f"PROBLEMS FOUND ({len(problems)}):")
        for prob in problems[:20]:
            print(f"  - {prob}")
        raise RuntimeError("Verification failed! See issues above.")
    else:
        print("ALL PATHS OK: 100% of images exist and have matching .txt labels.")

    # ---------------------------------------------------------
    # Print content of obj.data and first 3 lines of each txt file
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("CONTENT OF data/obj.data:")
    print("-" * 60)
    print(obj_data_path.read_text(encoding="utf-8").strip())
    print("=" * 60)

    print("\nCONTENT OF data/obj.names:")
    print("-" * 60)
    print(obj_names_path.read_text(encoding="utf-8").strip())
    print("=" * 60)

    for split_name, txt_file in splits.items():
        print(f"\nFIRST 3 LINES OF {txt_file.name}:")
        print("-" * 60)
        with open(txt_file, "r", encoding="utf-8") as f:
            for _ in range(3):
                print(f.readline().strip())
        print(f"(Total lines: {counts[split_name]:,})")

    print("\n" + "=" * 60)
    print("STAGE 5 COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    create_darknet_files()
