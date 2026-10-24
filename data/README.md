# Dataset Directory Structure

This directory holds the dataset manifests, class configurations, and data partitions for training and evaluation.

### Directory Layout:
```
data/
├── manifests/
│   ├── audit.csv            # Perceptual hash audit and duplicate component groups
│   └── split.csv            # Leak-free Stratified Group K-Fold split manifest
├── obj.names                # Class names (potato_early_blight, potato_late_blight, potato_healthy)
├── obj.data                 # Darknet dataset metadata configuration (Windows)
├── obj_wsl.data             # Darknet dataset metadata configuration (WSL2 Linux)
├── obj_test.data            # Darknet test dataset metadata configuration (Windows)
├── obj_test_wsl.data        # Darknet test dataset metadata configuration (WSL2 Linux)
├── raw/                     # Raw PlantVillage image directory (excluded from git)
│   ├── Potato___Early_blight/  (1,000 images)
│   ├── Potato___Late_blight/   (1,000 images)
│   └── Potato___healthy/       (152 images)
└── processed/               # Augmented training images & labels (generated via scripts)
```

### Setup Instructions:
1. Extract or place the raw PlantVillage potato dataset into `data/raw/` (or extract `dataset/potato_disease.zip`).
2. Run data audit and splitting:
   ```bash
   python scripts/audit_dataset.py
   python scripts/split_and_augment.py
   python scripts/annotate_dataset.py
   python scripts/create_darknet_data.py
   ```
