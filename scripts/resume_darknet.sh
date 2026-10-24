#!/bin/bash
# Script to resume YOLOv4 Darknet training from last checkpoint
set -e

PROJECT_DIR="/mnt/c/Users/vivek/OneDrive/Desktop/AgriBot"
cd "$PROJECT_DIR"

LAST_WEIGHTS="weights/backup/yolov4-custom-potato_last.weights"

if [ ! -f "$LAST_WEIGHTS" ]; then
    echo "ERROR: Checkpoint $LAST_WEIGHTS not found!"
    exit 1
fi

echo "Resuming training from: $LAST_WEIGHTS"

nohup ./darknet/darknet detector train \
    data/obj_wsl.data \
    cfg/yolov4-custom-potato.cfg \
    "$LAST_WEIGHTS" \
    -dont_show \
    >> results/train.log 2>&1 &

echo "Training resumed with PID $!"
