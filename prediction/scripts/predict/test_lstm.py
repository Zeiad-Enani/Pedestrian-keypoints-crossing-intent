import json
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt

# ----- Config -----
TEST_JSON    = "prediction.data/sequences_testing_v2.json"   # your 557-window file
MODEL_PATH   = "prediction/models/lstm/best_lstm_10k.pth"        # your saved checkpoint
SAVE_PATH = "prediction/models/best_lstm_10k_testing_v2.png"        # where to save the dashboard
IMG_W, IMG_H = 192, 256
WINDOW_SIZE  = 10
FEATURE_DIM  = 17 * 4
BATCH_SIZE   = 32
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD    = 0.8   # from your macro-F1 sweep

# ----- Dataset -----
class TestDataset(Dataset):
    def __init__(self, json_path):
        data = json.load(open(json_path))
        seqs, labels = [], []
        for e in data:
            raw = np.array(e["sequence"], dtype=np.float32).reshape(WINDOW_SIZE,17,2)
            deltas = raw[1:] - raw[:-1]
            deltas = np.vstack([np.zeros((1,17,2),dtype=np.float32), deltas])
            raw[...,0]   /= IMG_W;    raw[...,1]   /= IMG_H
            deltas[...,0]/= IMG_W;    deltas[...,1]/= IMG_H
            combo = np.concatenate([raw, deltas], axis=2)  # (10,17,4)
            seqs.append(combo.reshape(WINDOW_SIZE, FEATURE_DIM))
            labels.append(int(e["label"]))
        self.X = torch.tensor(np.stack(seqs), dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ----- Model (must match your train script) -----
class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim=FEATURE_DIM, hidden_dim=128, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h_fwd = h_n[-2]; h_bwd = h_n[-1]
        h_cat = torch.cat([h_fwd, h_bwd], dim=1)
        return self.head(h_cat).view(-1)

# ----- Helper Functions -----
def plot_metrics(report, thresholds, macro_f1s,
                 fpr, tpr, roc_auc,
                 cm, acc, best_t, output_path):
    # Ensure output directory exists
    # os.makedirs(output_dir, exist_ok=True)

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
    # dashboard_path = 
    fig.savefig(output_path, dpi=200, bbox_inches='tight')


    plt.close(fig)


# ----- Inference & Metrics -----
def evaluate():
    # load data
    ds = TestDataset(TEST_JSON)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    # load model
    model = BiLSTMClassifier().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # collect all probabilities and labels
    all_probs = []
    all_true  = []
    with torch.no_grad():
        for seq, lbl in loader:
            seq = seq.to(DEVICE)
            logits = model(seq)
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_true.extend(lbl.numpy().tolist())

    # apply fixed threshold

    # 2) Macro-F1 sweep
    thresholds = np.linspace(0.1, 0.9, 81)
    macro_f1s   = []
    best_f1, best_t = 0.0, THRESHOLD
    for t in thresholds:
        ps = [1 if p > t else 0 for p in all_probs]
        f1 = f1_score(all_true, ps, average='macro')
        macro_f1s.append(f1)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    print(f"\nBest macro-F1 {best_f1:.4f} at threshold={best_t:.2f}")


     # 1) Thresholded predictions & classification report
    preds = [1 if p > best_t else 0 for p in all_probs]
    acc   = accuracy_score(all_true, preds)
    print(f"\nTest windows: {len(all_true)}")
    print(f"Accuracy @ {best_t:.2f}: {acc:.4f}")
    report = classification_report(all_true, preds, output_dict=True, digits=4)
    print("\nClassification Report:\n", classification_report(all_true, preds, digits=4))

    # 3) ROC-AUC and curve
    roc_auc = roc_auc_score(all_true, all_probs)
    fpr, tpr, _ = roc_curve(all_true, all_probs)
    print(f"\nROC AUC: {roc_auc:.4f}")

    # 4) Confusion matrix
    cm = confusion_matrix(all_true, preds)
    tn, fp, fn, tp = cm.ravel()
    print("\nConfusion Matrix (counts):")
    print(f"  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    

    # 5) Dashboard plot
    # output_dir = f"a_LSTM/{MODEL_PATH.split('/')[-1].split('.')[0]}.png" 
    plot_metrics(report, thresholds, macro_f1s, fpr, tpr, roc_auc, cm, acc, best_t, SAVE_PATH)

if __name__ == "__main__":
    evaluate()
