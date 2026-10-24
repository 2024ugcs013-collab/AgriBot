"""
Training Monitor for YOLOv4 Darknet
Parses results/train.log and displays live statistics (iteration, loss, mAP, ETA).
"""

from pathlib import Path
import re
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_FILE = PROJECT_ROOT / "results" / "train.log"

def monitor():
    if not LOG_FILE.exists():
        print(f"Log file not found: {LOG_FILE}. Training has not started yet.")
        return

    print("=" * 65)
    print("YOLOv4 DARKNET TRAINING MONITOR")
    print("=" * 65)
    print(f"Reading: {LOG_FILE}\n")

    lines = LOG_FILE.read_text(encoding="utf-8", errors="ignore").splitlines()
    iter_lines = [l for l in lines if re.search(r"^\s*\d+:\s+[\d\.]+", l)]

    if not iter_lines:
        print("Training starting up / initializing CUDA buffers...")
        print("Last 10 lines of log:")
        for l in lines[-10:]:
            print("  ", l)
        return

    last_iter_line = iter_lines[-1]
    match = re.search(r"^\s*(\d+):\s+([\d\.]+),\s+([\d\.]+)\s+avg loss.*?([\d\.]+)\s+hours left", last_iter_line)

    map_lines = [l for l in lines if "mean_average_precision (mAP@0.50)" in l or "mAP@0.50" in l]
    last_map = map_lines[-1] if map_lines else "mAP will calculate every 1,000 iterations (or 4 epochs)"

    print(f"Latest Iteration Entry:\n  {last_iter_line.strip()}\n")
    print(f"Validation mAP Status:\n  {last_map.strip()}\n")
    print(f"Total Iteration Records Parsed: {len(iter_lines):,}")

if __name__ == "__main__":
    monitor()
