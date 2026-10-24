"""
Script to validate YOLO label files across all splits.
Use after manual edits with LabelImg to verify format integrity.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

def validate_all_labels():
    print("=" * 60)
    print("VALIDATING YOLO LABELS")
    print("=" * 60)
    total_imgs = 0
    errors = []

    for split in ["train", "valid", "test"]:
        split_dir = PROCESSED_DIR / split
        imgs = sorted(list(split_dir.glob("*.jpg")))
        for img in imgs:
            total_imgs += 1
            txt = img.with_suffix(".txt")
            if not txt.exists():
                errors.append(f"Missing label file: {txt.name}")
                continue
            lines = [l.strip() for l in txt.read_text().splitlines() if l.strip()]
            if len(lines) != 1:
                errors.append(f"Expected 1 line in {txt.name}, got {len(lines)}")
                continue
            parts = lines[0].split()
            if len(parts) != 5:
                errors.append(f"Invalid columns in {txt.name}: {parts}")
                continue
            try:
                cid = int(parts[0])
                xc, yc, w, h = [float(x) for x in parts[1:]]
                if cid not in [0, 1, 2]:
                    errors.append(f"Invalid class ID {cid} in {txt.name}")
                if not (0.0 < xc < 1.0 and 0.0 < yc < 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                    errors.append(f"Coordinates out of range in {txt.name}: {parts[1:]}")
            except ValueError:
                errors.append(f"Non-numeric value in {txt.name}: {parts}")

    if errors:
        print(f"Validation FAILED with {len(errors)} errors:")
        for err in errors[:20]:
            print(f"  - {err}")
    else:
        print(f"ALL LABELS VALID: 100% of {total_imgs:,} images have verified YOLO labels.")

if __name__ == "__main__":
    validate_all_labels()
