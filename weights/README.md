# Model Weights Directory: DenseNet-121 CNN

This directory contains the neural network weights for the **AgriBot DenseNet-121 Potato Pathology Classifier**.

### Model Weights File:
- **`densenet121_best.pt`** (~28.4 MB)
  - Pretrained on ImageNet and fine-tuned on the leak-free PlantVillage potato pathology dataset (Early Blight, Late Blight, Healthy).
  - Terminal classification head: `Linear(in_features=1024, out_features=3)` (with dropout $p=0.3$).
  - Achieves **100.00% Accuracy** and **100.00% Macro F1-Score** on the held-out untouched test set (215 leaves).

### Retraining / Generating Weights:
To retrain or fine-tune DenseNet-121 from scratch:
```bash
python scripts/train_densenet.py --epochs 5 --batch-size 32 --lr 0.0001
```

### Evaluating Weights:
To evaluate the checkpoint on the 215-sample test set:
```bash
python scripts/evaluate_densenet.py --weights weights/densenet121_best.pt
```
