import torch
import torch.nn as nn
import numpy as np
import json
import os
from sklearn.metrics import (
    classification_report, 
    accuracy_score, 
    roc_auc_score,
    roc_curve,
    f1_score,
    confusion_matrix
    )
import matplotlib.pyplot as plt

# Match model architecture used during training
class PoseClassifier(nn.Module):
    def __init__(self, input_dim):
        super(PoseClassifier, self).__init__()
        self.model = nn.Sequential(
            ## -------------------- ## 64 2 hidden layers model
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Load your test data
def load_test_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    inputs = []
    frame_ids = []
    labels = []

    for item in data:
        keypoints = np.array(item['keypoints']).astype(np.float32)
        label = float(item['label'])  # 0.0 or 1.0
        inputs.append(keypoints)
        frame_ids.append(item['frame'])
        labels.append(label)

    return torch.tensor(inputs), frame_ids, labels


def plot_metrics(report, thresholds, macro_f1s,
                 fpr, tpr, roc_auc,
                 cm, acc, best_t, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Prepare labels
    classes     = [c for c in report.keys() if c.isdigit()]
    plot_labels = ["not-crossing", "crossing"] + ["macro avg"]
    precs = [report[c]['precision'] for c in classes] + [report["macro avg"]['precision']]
    recs  = [report[c]['recall']    for c in classes] + [report["macro avg"]['recall']]
    f1s   = [report[c]['f1-score']  for c in classes] + [report["macro avg"]['f1-score']]

    # Build the 2x2 dashboard
    fig, axes = plt.subplots(2, 2, figsize=(12,10),
                             gridspec_kw={'hspace':0.3,'wspace':0.3})
    ax0, ax1, ax2, ax3 = axes[0,0], axes[0,1], axes[1,0], axes[1,1]


    # — A) Classification Report Bar Chart —
    x = np.arange(len(plot_labels))
    width = 0.25
    bars_p = ax0.bar(x - width, precs, width, label='Precision')
    bars_r = ax0.bar(x      , recs , width, label='Recall')
    bars_f = ax0.bar(x + width, f1s , width, label='F1-score') 
    for bars in (bars_p, bars_r, bars_f):
        for bar in bars:
            h = bar.get_height()
            ax0.text(bar.get_x()+bar.get_width()/2, h+0.01, f"{h:.2f}",
                     ha='center', va='bottom', fontsize=8)
    ax0.set_xticks(x)
    ax0.set_xticklabels(plot_labels, rotation=15)
    ax0.set_ylim(0,1.05)
    ax0.set_title("Classification Report")
    ax0.legend()

    
    # — B) Macro-F1 vs Threshold —
    ax1.plot(thresholds, macro_f1s, marker='o')
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Macro-F1")
    ax1.set_title("Macro-F1 vs Decision Threshold")
    ax1.grid(True)
    


    # — C) ROC Curve —
    ax2.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    ax2.plot([0,1],[0,1],'--',color='gray')
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve")
    ax2.legend(loc='lower right')
    ax2.grid(True)
    ax2.text(0.95, 0.05, f"AUC = {roc_auc:.3f}",
             ha='right', va='bottom', transform=ax2.transAxes,
             bbox=dict(boxstyle="round,pad=0.3",fc="white",alpha=0.5))

    # — D) Confusion Matrix —
    cm_norm = cm.astype(float)/cm.sum()
    im = ax3.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1, aspect='equal')
    for i in range(2):
        for j in range(2):
            ax3.text(j, i, f"{cm[i,j]}\n{cm_norm[i,j]:.2f}",
                     ha='center', va='center')
    ax3.set_xticks([0,1]); ax3.set_yticks([0,1])
    ax3.set_xticklabels(['Pred 0','Pred 1'])
    ax3.set_yticklabels(['True 0','True 1'])
    ax3.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)


    # — Global annotation —
    fig.text(0.5, 0.97, f"Accuracy @ {best_t:.2f} = {acc:.3f}",
             ha='center', va='top', fontsize=12)

    # Save full dashboard
    dashboard_path = os.path.join(output_dir, "dashboard.png")
    fig.savefig(dashboard_path, dpi=200, bbox_inches='tight')


    plt.close(fig)

def predict_and_evaluate(model_path, data_path, output_dir="inference", save_plots=True):
    x, frame_ids, y_true = load_test_data(data_path)
    input_dim = x.shape[1]

    model = PoseClassifier(input_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        probs = model(x).view(-1).numpy()           # probabilities [0–1]
        preds = (probs > 0.5).astype(int)

    
    # A) Macro-F1 sweep
    thresholds = np.linspace(0.1, 0.9, 81)
    macro_f1s   = []
    best_f1, best_t = 0.0, 0.5
    for t in thresholds:
        ps = [1 if p > t else 0 for p in probs]
        f1 = f1_score(y_true, ps, average='macro')
        macro_f1s.append(f1)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    print(f"\nBest macro-F1 {best_f1:.4f} at threshold={best_t:.2f}")

    # B) Thresholded metrics
    preds = [1 if p > best_t else 0 for p in probs]
    acc   = accuracy_score(y_true, preds)
    print(f"\nTest samples: {len(y_true)}")
    print(f"Accuracy @ {best_t:.2f}: {acc:.4f}\n")
    print("Classification Report:")
    report = classification_report(y_true, preds, output_dict=True, digits=4)
    print("\nClassification Report:\n", classification_report(y_true, preds, digits=4))

    
    # C) ROC–AUC and curve
    auc = roc_auc_score(y_true, probs)
    fpr, tpr, _ = roc_curve(y_true, probs)
    print(f"\nROC AUC: {auc:.4f}")

    # D) Confusion matrix
    cm = confusion_matrix(y_true, preds)
    tn, fp, fn, tp = cm.ravel()
    print("\nConfusion Matrix (counts):")
    print(f"  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    # cm_prop = confusion_matrix(y_true, preds, normalize='all')
    # print("\nConfusion Matrix (proportions):")
    # print(cm_prop)

    # E) Dashboard plot
    if save_plots:
        plot_metrics(report, thresholds, macro_f1s, fpr, tpr, auc, cm, acc, best_t, output_dir)


# Configuration
MODEL = "classification/models/pose_classifier_64_small_drop.pth"
DATA_PATH = "classification/data/testing_data.json"

if __name__ == "__main__":
    predict_and_evaluate(MODEL, DATA_PATH, save_plots=False)
