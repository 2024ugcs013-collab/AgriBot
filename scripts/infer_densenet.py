"""
AgriBot - Potato Pathology CNN Classifier (DenseNet-121)
Stage: Production CLI Inference Engine & Agronomic Advisory

Performs high-throughput or single-image inference using the fine-tuned
DenseNet-121 convolutional neural network. Predicts:
  - Potato Early Blight (Alternaria solani)
  - Potato Late Blight (Phytophthora infestans)
  - Potato Healthy Foliage

Generates softmax confidence scores and actionable agronomic advisory
(fungicide schedules, cultural practices, and chemical intervention thresholds).
"""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"
RESULTS_DIR = PROJECT_ROOT / "results"

CLASS_MAP = {
    0: "Potato Early Blight (Alternaria solani)",
    1: "Potato Late Blight (Phytophthora infestans)",
    2: "Potato Healthy Foliage",
}

AGRONOMIC_ADVISORY = {
    0: {
        "disease": "Potato Early Blight",
        "pathogen": "Alternaria solani (Fungal)",
        "symptoms": "Concentric dark brown rings ('target spots') on mature lower leaves.",
        "recommended_action": (
            "Initiate protective foliar fungicide applications: Mancozeb (2.0-2.5 kg/ha) "
            "or Chlorothalonil (1.5-2.0 L/ha) at 7-10 day intervals. Improve nitrogen management "
            "and avoid overhead sprinkler irrigation."
        ),
        "urgency": "MODERATE"
    },
    1: {
        "disease": "Potato Late Blight",
        "pathogen": "Phytophthora infestans (Oomycete)",
        "symptoms": "Water-soaked dark lesions with pale chlorotic margins and white mycelial growth on leaf underside.",
        "recommended_action": (
            "CRITICAL: Immediate therapeutic systemic fungicide spraying required. "
            "Apply Cymoxanil + Mancozeb, Metalaxyl-M, or Mandipropamid. Destroy heavily infected "
            "cull piles immediately to prevent field-wide epidemic defoliation."
        ),
        "urgency": "HIGH / CRITICAL"
    },
    2: {
        "disease": "Healthy Potato Leaf",
        "pathogen": "None detected",
        "symptoms": "Vibrant green lamina with intact vascular tissue and no pathological lesions.",
        "recommended_action": (
            "Continue standard agronomic monitoring. Maintain preventive scouting every 3-5 days. "
            "No chemical fungicide intervention necessary."
        ),
        "urgency": "NONE (Routine Scouting)"
    }
}


def load_densenet_model(weights_path, device):
    """Loads DenseNet-121 model with saved state dict."""
    model = models.densenet121(weights=None)
    num_features = model.classifier.in_features
    try:
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, 3)
        )
        model.load_state_dict(torch.load(weights_path, map_location=device))
    except RuntimeError:
        model.classifier = nn.Linear(num_features, 3)
        model.load_state_dict(torch.load(weights_path, map_location=device))

    model = model.to(device)
    model.eval()
    return model


def predict_single_image(img_path, model, device, transform):
    """Predicts class and returns probabilities for a single image."""
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    latency_ms = (time.perf_counter() - t0) * 1000

    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])

    return pred_idx, confidence, probs, latency_ms


def main():
    parser = argparse.ArgumentParser(description="DenseNet-121 Potato Disease Inference CLI")
    parser.add_argument("--image", type=Path, help="Path to single image file")
    parser.add_argument("--dir", type=Path, help="Path to directory of images to process in batch")
    parser.add_argument("--weights", type=Path, default=WEIGHTS_DIR / "densenet121_best.pt", help="Path to .pt weights file")
    parser.add_argument("--output", type=Path, default=None, help="Path to output JSON/CSV log")
    args = parser.parse_args()

    if not args.image and not args.dir:
        parser.error("Must specify either --image <path> or --dir <folder>")

    if not args.weights.exists():
        print(f"Error: Model weights not found at {args.weights}", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading DenseNet-121 CNN model from {args.weights.name} (Device: {device})...")
    model = load_densenet_model(args.weights, device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if args.image:
        if not args.image.exists():
            print(f"Error: Image {args.image} does not exist.", file=sys.stderr)
            sys.exit(1)

        pred_idx, conf, probs, latency = predict_single_image(args.image, model, device, transform)
        advisory = AGRONOMIC_ADVISORY[pred_idx]

        print("\n" + "=" * 65)
        print("AGRIBOT CNN DIAGNOSTIC RESULT")
        print("=" * 65)
        print(f"Image File:         {args.image.name}")
        print(f"Predicted Disease:  {CLASS_MAP[pred_idx]}")
        print(f"Confidence:         {conf * 100:.2f}%")
        print(f"Inference Latency:  {latency:.2f} ms")
        print(f"Urgency Level:      {advisory['urgency']}")
        print("-" * 65)
        print(f"Class Probabilities:")
        for idx, name in CLASS_MAP.items():
            print(f"  [{idx}] {name:<42}: {probs[idx]*100:6.2f}%")
        print("-" * 65)
        print(f"Agronomic Advisory & Chemical Recommendation:")
        print(f"  {advisory['recommended_action']}")
        print("=" * 65 + "\n")

    elif args.dir:
        if not args.dir.exists():
            print(f"Error: Directory {args.dir} does not exist.", file=sys.stderr)
            sys.exit(1)

        extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
        image_files = []
        for ext in extensions:
            image_files.extend(list(args.dir.glob(ext)))

        if not image_files:
            print(f"No image files found in {args.dir}.")
            sys.exit(0)

        print(f"\nProcessing {len(image_files)} images from {args.dir}...")
        results = []
        for img_p in image_files:
            pred_idx, conf, probs, latency = predict_single_image(img_p, model, device, transform)
            results.append({
                "file": img_p.name,
                "prediction": CLASS_MAP[pred_idx],
                "confidence": conf,
                "early_blight_prob": float(probs[0]),
                "late_blight_prob": float(probs[1]),
                "healthy_prob": float(probs[2]),
                "latency_ms": latency
            })

        print(f"Batch inference complete for {len(results)} images.")
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            import pandas as pd
            df_out = pd.DataFrame(results)
            df_out.to_csv(args.output, index=False)
            print(f"Saved batch inference results to: {args.output}")


if __name__ == "__main__":
    main()
