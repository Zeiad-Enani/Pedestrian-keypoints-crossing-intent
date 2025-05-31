# Classification Models

This folder contains models and scripts for classifying pedestrian actions or states using keypoint data.

## Overview
- **Task:** Classify pedestrian actions or states (e.g., crossing intent) from keypoint sequences.
- **Input:** Keypoint data extracted from video frames.

## Models

### pose_classifier_64_big.pth
- **Trained on:** `data_big.json`
- **Purpose:** Classifies pedestrian actions/states using a larger dataset.

### pose_classifier_64_small_drop.pth
- **Trained on:** `data_small.json`
- **Purpose:** Classifies pedestrian actions/states using a smaller dataset with dropout regularization.

### pose_classifier_deep_big.pth
- **Trained on:** `data_big.json`
- **Purpose:** Deep architecture for classifying pedestrian actions/states on the large dataset.

### pose_classifier_deep_small.pth
- **Trained on:** `data_small.json`
- **Purpose:** Deep architecture for classifying pedestrian actions/states on the small dataset.

## Data
- Training and evaluation datasets are in the `data/` folder:
  - `data_big.json`: Large dataset for training/evaluation.
  - `data_small.json`: Small dataset for training/evaluation.
  - `testing_data.json`: Used for model testing/inference.

## Scripts
- Training scripts: `scripts/train/`
- Inference scripts: `scripts/inference/`

## Usage
1. Place your trained model files in the `models/` folder.
2. Use the training scripts to train new models or the inference scripts to run classification.
3. Update paths in scripts as needed to match your setup.

## Notes
- All models are trained and evaluated on keypoint data.
- For more details, see comments in the respective scripts.
