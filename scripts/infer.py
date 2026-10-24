"""
Stage 9: YOLOv4 Inference Pipeline
Provides single-image and batch-folder inference functions with visual annotations
and predictions CSV export.
"""

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CFG_PATH = PROJECT_ROOT / "cfg" / "yolov4-custom-potato.cfg"
NAMES_PATH = PROJECT_ROOT / "data" / "obj.names"

CLASS_NAMES = ["potato_early_blight", "potato_late_blight", "potato_healthy"]
if NAMES_PATH.exists():
    loaded_names = [l.strip() for l in NAMES_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(loaded_names) == 3:
        CLASS_NAMES = loaded_names


def load_network(weights_path: Path):
    """Load YOLOv4 network in OpenCV DNN."""
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    if not CFG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CFG_PATH}")

    net = cv2.dnn.readNetFromDarknet(str(CFG_PATH), str(weights_path))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


def predict_image(
    image_input,
    weights_path: Path = None,
    net=None,
    conf_thresh: float = 0.25,
    nms_thresh: float = 0.45,
    output_path: Path = None,
):
    """
    Run YOLOv4 detection on a single image.
    image_input: Path, string, or numpy array (BGR).
    Returns dict: {
        'boxes': list of [x, y, w, h],
        'confidences': list of float,
        'class_ids': list of int,
        'class_names': list of str,
        'top_class': str,
        'top_confidence': float,
        'annotated_image': np.ndarray (BGR)
    }
    """
    if net is None:
        if weights_path is None:
            # Auto-discover weights
            candidates = [
                PROJECT_ROOT / "weights" / "yolov4-custom-potato_best.weights",
                PROJECT_ROOT / "weights" / "backup" / "yolov4-custom-potato_best.weights",
                PROJECT_ROOT / "weights" / "backup" / "yolov4-custom-potato_last.weights",
                PROJECT_ROOT / "weights" / "backup" / "yolov4-custom-potato_1000.weights",
            ]
            for cand in candidates:
                if cand.exists():
                    weights_path = cand
                    break
        assert weights_path is not None and weights_path.exists(), "No valid weights found"
        net = load_network(weights_path)

    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input))
        if img is None:
            raise FileNotFoundError(f"Cannot read image from {image_input}")
    elif isinstance(image_input, np.ndarray):
        img = image_input.copy()
    else:
        raise ValueError("image_input must be a file path or numpy ndarray")

    h, w = img.shape[:2]
    output_layers = net.getUnconnectedOutLayersNames()

    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes = []
    confidences = []
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
                confidences.append(confidence)
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, nms_thresh)

    annotated = img.copy()
    det_boxes = []
    det_confs = []
    det_cls_ids = []
    det_cls_names = []

    if len(indices) > 0:
        idx_list = indices.flatten() if hasattr(indices, "flatten") else [i[0] for i in indices]
        for i in idx_list:
            bx, by, bw, bh = boxes[i]
            conf = confidences[i]
            cid = class_ids[i]
            cname = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"class_{cid}"

            det_boxes.append([bx, by, bw, bh])
            det_confs.append(conf)
            det_cls_ids.append(cid)
            det_cls_names.append(cname)

            # Draw green bounding box (BGR: 0, 255, 0)
            cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)

            # Draw label banner with red text (BGR: 0, 0, 255)
            label = f"{cname}: {conf*100:.1f}%"
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated, (bx, max(0, by - text_h - 6)), (bx + text_w, by), (0, 0, 0), -1)
            cv2.putText(
                annotated,
                label,
                (bx, max(text_h + 2, by - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        top_idx = int(np.argmax(det_confs))
        top_class = det_cls_names[top_idx]
        top_confidence = det_confs[top_idx]
    else:
        top_class = "no_detection"
        top_confidence = 0.0

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), annotated)

    return {
        "boxes": det_boxes,
        "confidences": det_confs,
        "class_ids": det_cls_ids,
        "class_names": det_cls_names,
        "top_class": top_class,
        "top_confidence": top_confidence,
        "annotated_image": annotated,
    }


def predict_folder(
    folder_path: Path,
    output_dir: Path,
    weights_path: Path = None,
    conf_thresh: float = 0.25,
    nms_thresh: float = 0.45,
):
    """Run inference on all images in a folder and export annotated copies and predictions.csv."""
    folder_path = Path(folder_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if weights_path is None:
        candidates = [
            PROJECT_ROOT / "weights" / "yolov4-custom-potato_best.weights",
            PROJECT_ROOT / "weights" / "backup" / "yolov4-custom-potato_best.weights",
            PROJECT_ROOT / "weights" / "backup" / "yolov4-custom-potato_last.weights",
            PROJECT_ROOT / "weights" / "backup" / "yolov4-custom-potato_1000.weights",
        ]
        for cand in candidates:
            if cand.exists():
                weights_path = cand
                break
    net = load_network(weights_path)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    img_files = [f for f in folder_path.rglob("*") if f.suffix.lower() in image_extensions]
    print(f"Found {len(img_files)} images in {folder_path}. Processing...")

    records = []
    for f in img_files:
        save_path = output_dir / f.name
        res = predict_image(f, net=net, conf_thresh=conf_thresh, nms_thresh=nms_thresh, output_path=save_path)
        records.append({
            "filename": f.name,
            "filepath": str(f),
            "top_class": res["top_class"],
            "top_confidence": res["top_confidence"],
            "num_detections": len(res["boxes"]),
            "annotated_path": str(save_path),
        })

    df = pd.DataFrame(records)
    csv_path = output_dir / "predictions.csv"
    df.to_csv(csv_path, index=False)
    print(f"Processed {len(records)} images. Annotated copies and CSV saved to: {output_dir}")
    return df


def main():
    parser = argparse.ArgumentParser(description="YOLOv4 Potato Leaf Disease Inference Demo")
    parser.add_argument("--image", type=str, default=None, help="Path to single input image")
    parser.add_argument("--folder", type=str, default=None, help="Path to input folder with images")
    parser.add_argument("--weights", type=str, default=None, help="Path to YOLOv4 weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--output", type=str, default="results/predictions", help="Output path/folder")
    args = parser.parse_args()

    weights_path = Path(args.weights) if args.weights else None

    if args.image:
        out_file = Path(args.output)
        if out_file.is_dir() or not out_file.suffix:
            out_file = out_file / f"pred_{Path(args.image).name}"
        res = predict_image(args.image, weights_path=weights_path, conf_thresh=args.conf, output_path=out_file)
        print(f"\nResult for {args.image}:")
        print(f"  Diagnosis: {res['top_class']}")
        print(f"  Confidence: {res['top_confidence']*100:.2f}%")
        print(f"  Annotated image saved to: {out_file}")
    elif args.folder:
        predict_folder(Path(args.folder), Path(args.output), weights_path=weights_path, conf_thresh=args.conf)
    else:
        print("Please provide either --image <path> or --folder <path>.")


if __name__ == "__main__":
    main()
