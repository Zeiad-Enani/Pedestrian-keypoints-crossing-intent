# Pedestrian-keypoints-crossing-intent

This project is organized for pedestrian classification and prediction using keypoints and crossing intent analysis. It leverages annotated datasets (PIE), deep learning models, and modular scripts for training and inference.

## Project Structure

```
Pedestrian_Classification_Prediction/
│
├── annotations/             # PIE dataset annotations (XML, CSV)
│   └── setXX/               # Each set contains .xml and .csv files
│
├── classification/          # Classification task
│   ├── data/                # JSON datasets for classification
│   ├── models/              # Trained .pth model files
│   ├── scripts/
│   │   ├── inference/       # Inference scripts
│   │   └── train/           # Training scripts
│   └── utils/               # Helper functions
│
├── prediction/              # Prediction task
│   ├── data/                # JSON datasets for prediction
│   ├── models/              # Trained .pth model files (cnn, lstm, stgcn)
│   ├── scripts/
│   │   ├── predict/         # Prediction scripts
│   │   └── train/           # Training scripts
│   └── utils/               # Helper functions
│
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

## Getting Started

1. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```
2. **Prepare data:**
   - Place PIE annotation files in `annotations/`.
   - Ensure datasets are in the respective `data/` folders.
3. **Train models:**
   - Use scripts in `classification/scripts/train/` or `prediction/scripts/train/`.
4. **Run inference/prediction:**
   - Use scripts in `classification/scripts/inference/` or `prediction/scripts/predict/`.

## Notes
- Update paths in scripts as needed to match your local setup.
- See script docstrings or comments for specific usage instructions.

## Citation
If you use this project or the PIE dataset, please cite the original authors as appropriate.
