#!/bin/bash
# Script to launch YOLOv4 Darknet training in detached background session
set -e

PROJECT_DIR="/mnt/c/Users/vivek/OneDrive/Desktop/AgriBot"
cd "$PROJECT_DIR"

mkdir -p results
mkdir -p weights/backup

echo "Starting YOLOv4 Darknet training..."
setsid nohup ./darknet/darknet detector train \
    data/obj_wsl.data \
    cfg/yolov4-custom-potato.cfg \
    weights/yolov4.conv.137 \
    -dont_show \
    </dev/null > results/train.log 2>&1 &

sleep 2
echo "Darknet training active."
echo "Tailing log:"
tail -n 15 results/train.log

