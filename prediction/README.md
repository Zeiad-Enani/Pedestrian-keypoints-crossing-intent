# Prediction Models

This folder contains models and scripts for predicting pedestrian crossing intent using sequences of keypoints.

## Overview
- **Task:** Predict whether a pedestrian will cross (crossing) or not (not crossing).
- **Input:** Sequences of keypoints extracted from video frames.
- **Sequence Details:**
  - Each sequence consists of 10 frames.
  - There are 5 frames between each sampled frame in the sequence.

## Models

### CNN
- **Trained on:**
  - `best_cnn_5k.pth`: Trained on `sequences_5k.json`
- **Purpose:** Predicts crossing intent (crossing/not crossing).
- **Model file location:** `models/cnn/`

### LSTM
- **Trained on:**
  - `best_lstm_5k.pth`: Trained on `sequences_5k.json`
  - `best_lstm_10k.pth`: Trained on `sequences_10k.json`
- **Purpose:** Predicts crossing intent (crossing/not crossing).
- **Model file location:** `models/lstm/`

### STGCN
- **Trained on:**
  - `best_stgcn_5k.pth`: Trained on `sequences_5k.json`
  - `best_stgcn_10k.pth`: Trained on `sequences_10k.json`
  - `best_stgcn_17k.pth`: Trained on `sequences_17k.json`
  - `best_stgcn_v5f.pth`: Trained on `sequences_v5f.json`
  - `best_stgcn_v5f_bal.pth`: Trained on `sequences_v5f_balanced.json`
- **Purpose:** Predicts crossing intent (crossing/not crossing).
- **Model file location:** `models/stgcn/`

## Data
- All models use datasets in the `data/` folder (e.g., `sequences_5k.json`, `sequences_10k.json`, `sequences_17k.json`, `sequences_v5f.json`, `sequences_v5f_balanced.json`).

## Scripts
- Training scripts: `scripts/train/`
- Prediction/inference scripts: `scripts/predict/`

## Usage
1. Place your trained model files in the appropriate subfolders under `models/`.
2. Use the training scripts to train new models or the prediction scripts to run inference.
3. Update paths in scripts as needed to match your setup.

## Notes
- All models are trained and evaluated on the same sequence format and task.
- For more details, see comments in the respective scripts.
