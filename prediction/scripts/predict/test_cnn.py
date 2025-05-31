# test_cnn.py

import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    roc_curve,
    f1_score,
    confusion_matrix
)

# ─── Config ─────────────────────────────────────────────────────────────────────
WINDOW_SIZE = 10
# If you include deltas, use 17*4=68; if raw only, use 17*2=34
FEATURE_DIM  = 17 * 2  
TEST_JSON    = "prediction/data/sequences_testing_v2.json"
CHECKPOINT   = "prediction/models/cnn/best_cnn_5k.pth"
SAVE_PATH    = "prediction/models/best_cnn_5k_testing_v2.png"
BATCH_SIZE   = 32
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD    = 0.5

# ─── Dataset ────────────────────────────────────────────────────────────────────
class SequenceDataset(Dataset):
    def __init__(self, json_path, img_w=192, img_h=256):
        data = json.load(open(json_path))
        seqs, labels = [], []
        for e in data:
            # load and reshape to (T,17,2)
            raw = np.array(e["sequence"], dtype=np.float32).reshape(WINDOW_SIZE, 17, 2)
            # normalize coordinates
            raw[..., 0] /= img_w
            raw[..., 1] /= img_h
            # flatten to (T,34)
            flat = raw.reshape(WINDOW_SIZE, FEATURE_DIM)
            seqs.append(flat)
            labels.append(float(e["label"]))
        self.X = torch.from_numpy(np.stack(seqs))       # (N, 10, 34)
        self.y = torch.from_numpy(np.array(labels))     # (N,)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ─── Model ──────────────────────────────────────────────────────────────────────
class Conv1DClassifier(nn.Module):
    def __init__(self, feat_dim=FEATURE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            # input now (B, T=10, C=34) → transpose to (B,34,10)
            nn.Conv1d(feat_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),  # extra depth
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),  # → (B,128,1)
            nn.Flatten(),             # → (B,128)
            nn.Dropout(0.5),
            nn.Linear(128, 1)         # → (B,1)
        )

    def forward(self, x):
        # x: (B, T, C) = (B, 10, 34)
        x = x.transpose(1, 2)  # → (B, 34, 10)
        return self.net(x).view(-1)  # → (B,)
    
# ─── Plot metrics ──────────────────────────────────────────────────────────────
def plot_metrics(report, thresholds, macro_f1s, fpr, tpr, roc_auc, cm, acc, best_t, output_path):

    # 1) Prepare labels for the bar chart: each class + a "macro avg" group
    classes     = [c for c in report.keys() if c.isdigit()]
    plot_labels = ["not-crossing", "crossing"] + ["macro avg"]

    # 2) Gather metrics: for each class + macro avg
    precs = [report[c]['precision'] for c in classes] + [report["macro avg"]['precision']]
    recs  = [report[c]['recall']    for c in classes] + [report["macro avg"]['recall']]
    f1s   = [report[c]['f1-score']  for c in classes] + [report["macro avg"]['f1-score']]

    # 3) The rest of your metrics
    macro_precision = report["macro avg"]['precision']

    # 4) Build subplots
    fig, axes = plt.subplots(2, 2, figsize=(12,10) , gridspec_kw={'hspace': 0.3, 'wspace': 0.3})
    ax0, ax1, ax2, ax3 = axes[0,0], axes[0,1], axes[1,0], axes[1,1]

    # ——————————————————————————————————————————————
    # A) Per‐class + macro bar chart
    x = np.arange(len(plot_labels))
    width = 0.25

    bars_p = ax0.bar(x - width, precs, width, label='Precision')
    bars_r = ax0.bar(x      , recs , width, label='Recall')
    bars_f = ax0.bar(x + width, f1s , width, label='F1-score')

    # Annotate each bar
    for bars in (bars_p, bars_r, bars_f):
        for bar in bars:
            h = bar.get_height()
            ax0.text(
                bar.get_x() + bar.get_width()/2,
                h + 0.01,
                f"{h:.2f}",
                ha='center', va='bottom', fontsize=8
            )

    ax0.set_xticks(x)
    ax0.set_xticklabels(plot_labels, rotation=15)
    ax0.set_ylim(0, 1.05)
    ax0.set_title("Classification Report")
    ax0.legend()

    # ——————————————————————————————————————————————
    # B) Macro-F1 vs Threshold
    ax1.plot(thresholds, macro_f1s, marker='o')
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Macro-F1")
    ax1.set_title("Macro-F1 vs Decision Threshold")
    ax1.grid(True)

    # ——————————————————————————————————————————————
    # C) ROC Curve
    ax2.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    ax2.plot([0,1], [0,1], '--', color='gray')
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve")
    ax2.legend(loc='lower right')
    ax2.grid(True)
    

    # ——————————————————————————————————————————————
    # D) Confusion matrix heatmap
    cm_norm = cm.astype(float) / cm.sum()
    cm_row = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    im = ax3.imshow(cm, cmap='Blues', vmin=0, vmax=cm.max())
    for i in range(2):
        for j in range(2):
            ax3.text(
                j, i,
                # f"{cm[i,j]}\n{cm[i,j]:.2f}",
                f"{cm[i,j]}\n{cm_norm[i,j]:.2f}",
                ha='center', va='center'
            )
    ax3.set_xticks([0,1]); ax3.set_yticks([0,1])
    ax3.set_xticklabels(['Pred 0','Pred 1'])
    ax3.set_yticklabels(['True 0','True 1'])
    ax3.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    # ——————————————————————————————————————————————
    # E) Global annotation for macro precision
    fig.text(
        0.5, 0.97,
        f"Accuracy @ {best_t:.2f} = {acc:.3f}",
        ha='center', va='top',
        fontsize=12
    )

    # plt.tight_layout(rect=[0,0,1,0.95])

    # ——————————————————————————————————————————————
    # F) Save the figure
    fig.savefig(output_path, dpi=300, bbox_inches='tight')

    # plt.show()



# ─── Evaluation ────────────────────────────────────────────────────────────────
def evaluate():
    # 1) Data
    ds     = SequenceDataset(TEST_JSON)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    # 2) Model
    model = Conv1DClassifier().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.eval()

    # 3) Inference
    probs, y_true = [], []
    with torch.no_grad():
        for seq, lbl in loader:
            seq = seq.to(DEVICE)
            logits = model(seq)
            probs.extend(logits.sigmoid().cpu().tolist())
            y_true.extend(lbl.tolist())

    # 3) evaluation

    # A) Macro-F1 sweep
    thresholds = np.linspace(0.1, 0.9, 81)
    macro_f1s   = []
    best_f1, best_t = 0.0, THRESHOLD
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
    cm_prop = confusion_matrix(y_true, preds, normalize='all')
    print("\nConfusion Matrix (proportions):")
    print(cm_prop)

    # E) Dashboard plot
    plot_metrics(report, thresholds, macro_f1s, fpr, tpr, auc, cm, acc, best_t, SAVE_PATH)


    

if __name__ == "__main__":
    evaluate()
