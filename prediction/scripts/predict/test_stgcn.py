import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score, roc_curve, confusion_matrix

# ----- Config -----
TEST_JSON    = "prediction/data/sequences_testing_v5f.json"
MODEL_PATH   = "prediction/models/stgcn/best_stgcn_v5f_enh_bal_enh.pth"
SAVE_PATH    = "prediction/models/best_stgcn_v5f_enh_testing_v5f_enh_bal.png"
IMG_W, IMG_H = 192, 256
WINDOW_SIZE  = 10
NUM_JOINTS   = 17
IN_CHANNELS  = 4    # raw(x,y) + delta(x,y)
BATCH_SIZE   = 32
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD    = 0.59

# ----- Adjacency (needed if constructing model here) -----
# But we'll import A_hat from train_stgcn if available.
# Here we assume A_hat is saved or we can rebuild it.
A = np.zeros((NUM_JOINTS, NUM_JOINTS), dtype=np.float32)
# old edges set
# EDGE_LIST = [
#     (0,1),(1,2),(2,3),(3,4),
#     (0,5),(5,6),(6,7),(7,8),
#     (0,9),(9,10),(10,11),
#     (0,12),(12,13),(13,14),
#     (0,15),(0,16)
# ]
EDGE_LIST = [
    (0,1),(1,2),(2,3),(3,4),
    (1,5),(2,6),
    (5,7),(7,9),
    (6,8),(8,10),
    (5,11),(6,12),
    (11,13),(13,15),
    (12,14),(14,16)
]


for i,j in EDGE_LIST:
    A[i,j] = 1; A[j,i] = 1
np.fill_diagonal(A, 1)
D = np.sum(A, axis=1)
D_inv_sqrt = np.diag(1.0 / np.sqrt(D))
A_hat = torch.tensor(D_inv_sqrt @ A @ D_inv_sqrt, dtype=torch.float32)

# ----- Dataset -----

class STGCNTestDatasetWithMeta(Dataset):
    def __init__(self, json_path):
        raw_data = json.load(open(json_path))
        self.data = []
        for e in raw_data:
            # --- build raw and deltas ---
            raw = np.array(e['sequence'], dtype=np.float32) \
                     .reshape(WINDOW_SIZE, NUM_JOINTS, 2)
            deltas = raw[1:] - raw[:-1]
            deltas = np.vstack([
                np.zeros((1, NUM_JOINTS, 2), dtype=np.float32),
                deltas
            ])

            # --- normalize ---
            raw[..., 0]    /= IMG_W;    raw[..., 1]    /= IMG_H
            deltas[..., 0] /= IMG_W;    deltas[..., 1] /= IMG_H

            combo = np.concatenate([raw, deltas], axis=2)  # shape (T, V, 4)

            # --- metadata you need ---
            meta = {
                'person_id':   e.get('person_id'),
                'start_frame': e.get('start_frame'),
                'label_frame': e.get('label_frame'),
            }

            self.data.append((
                torch.from_numpy(combo),                  # X: [T, V, C]
                torch.tensor(int(e['label']), dtype=torch.int64),  # y
                meta
            ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y, meta = self.data[idx]
        return x, y, meta
    

class STGCNTestDataset(Dataset):
    def __init__(self, json_path):
        data = json.load(open(json_path))
        seqs, labels = [], []
        for e in data:
            raw = np.array(e['sequence'], dtype=np.float32).reshape(WINDOW_SIZE, NUM_JOINTS, 2)
            deltas = raw[1:] - raw[:-1]
            deltas = np.vstack([np.zeros((1,NUM_JOINTS,2),dtype=np.float32), deltas])
            raw[...,0]   /= IMG_W;    raw[...,1]   /= IMG_H
            deltas[...,0]/= IMG_W;    deltas[...,1]/= IMG_H
            combo = np.concatenate([raw, deltas], axis=2)  # (T, V, C)
            

            seqs.append(combo)
            labels.append(int(e['label']))
        self.X = torch.tensor(np.stack(seqs), dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.int64)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# ----- ST-GCN Model Definition -----
class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, kernel_size=9, stride=1):
        super().__init__()
        self.register_buffer('A_hat', A)
        self.gcn = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        pad = (kernel_size - 1) // 2
        self.tcn = nn.Sequential(
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=(kernel_size,1), padding=(pad,0), stride=(stride,1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = torch.einsum('nctv,vw->nctw', x, self.A_hat)
        x = self.gcn(x)
        return self.tcn(x)

class STGCN(nn.Module):
    def __init__(self, A, in_channels=IN_CHANNELS, base_channels=64):
        super().__init__()
        self.data_bn = nn.BatchNorm1d(in_channels * NUM_JOINTS)
        self.layers = nn.ModuleList([
            STGCNBlock(in_channels, base_channels, A),
            STGCNBlock(base_channels, base_channels, A),
            STGCNBlock(base_channels, base_channels*2, A, stride=2),
            STGCNBlock(base_channels*2, base_channels*2, A)
        ])
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(base_channels*2, 1)
    def forward(self, x):
        # x: (N, T, V, C)
        x = x.permute(0,3,1,2).contiguous()  # (N,C,T,V)
        N,C,T,V = x.size()
        x = x.view(N, C*V, T)
        x = self.data_bn(x)
        x = x.view(N,C,T,V)
        for blk in self.layers:
            x = blk(x)
        x = self.pool(x).view(N,-1)
        return torch.sigmoid(self.fc(x)).view(-1)

# ----- Evaluation -----
# def evaluate():
#     ds = STGCNTestDataset(TEST_JSON)
#     loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

#     model = STGCN(A_hat).to(DEVICE)
#     model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
#     model.eval()

#     all_probs, all_true = [], []
#     with torch.no_grad():
#         for seq, lbl in loader:
#             seq = seq.to(DEVICE)
#             logits = model(seq)
#             all_probs.extend(logits.cpu().tolist())
#             all_true.extend(lbl.tolist())

#     # default threshold
#     thr = THRESHOLD
#     preds = [1 if p>thr else 0 for p in all_probs]

#     print(f"\nTest windows: {len(all_true)}")
#     print(f"Crossing: {sum(all_true)}, Not-crossing: {len(all_true)-sum(all_true)}")
#     print(f"Accuracy @ {THRESHOLD}: {accuracy_score(all_true, preds):.4f}\n")
#     print(classification_report(all_true, preds, digits=4))

#     # optional macro-F1 sweep
#     best_f1, best_t = 0.0, thr
#     for t in np.linspace(0.1,0.9,81):
#         ps = [1 if p>t else 0 for p in all_probs]
#         f1 = f1_score(all_true, ps, average='macro')
#         if f1>best_f1:
#             best_f1, best_t = f1, t
#     print(f"Best macro-F1 {best_f1:.4f} at threshold={best_t:.2f}")

def plot_metrics(report, thresholds, macro_f1s,
                 fpr, tpr, roc_auc,
                 cm, acc, best_t, output_path=SAVE_PATH ):
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


def evaluate():
    ds     = STGCNTestDataset(TEST_JSON)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    model = STGCN(A_hat).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_probs, all_true = [], []
    with torch.no_grad():
        for seq, lbl in loader:
            seq = seq.to(DEVICE)
            probs = model(seq).cpu().tolist()
            all_probs.extend(probs)
            all_true.extend(lbl.tolist())

    

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
    cm_prop = confusion_matrix(all_true, preds, normalize='all')
    print("\nConfusion Matrix (proportions):")
    print(cm_prop)

    # 5) Dashboard plot
    plot_metrics(report, thresholds, macro_f1s, fpr, tpr, roc_auc, cm, acc, best_t)

def collate_fn(batch):
    xs, ys, metas = zip(*batch)
    return default_collate(xs), default_collate(ys), metas

# def evaluate(model, json_path, batch_size, device, threshold=None):
#     # Prepare data loader
#     ds     = STGCNTestDatasetWithMeta(json_path)
#     loader = DataLoader(ds,
#                         batch_size=batch_size,
#                         shuffle=False,
#                         pin_memory=True,
#                         collate_fn=collate_fn)

#     # Load model
#     model = model.to(device)
#     model.eval()

#     results = []

    

#     with torch.no_grad():
#         for x, y_true, metas in loader:
#             x = x.to(device)               # [B, T, V, C]
#             logits = model(x)              # either [B] or [B,1] or [B,2]

#             # --- handle outputs of different shapes ---
#             if logits.dim() == 1 or logits.shape[1] == 1:
#                 # single-logit case → sigmoid for P(class=1)
#                 probs1 = torch.sigmoid(logits.view(-1))  # [B]
#                 probs0 = 1 - probs1
#             else:
#                 # two-logit case → softmax
#                 probs = F.softmax(logits, dim=1)         # [B,2]
#                 probs0 = probs[:, 0]
#                 probs1 = probs[:, 1]

#             # bring to CPU
#             probs0 = probs0.cpu()
#             probs1 = probs1.cpu()
#             preds  = (probs1 >= threshold).long()        # [B]

            

#             for i in range(len(preds)):
#                 # if threshold is not None and "6_2_1769" == metas[i].get('person_id', -1):
#                         # continue
#                     meta = metas[i] or {}
#                     entry = {
#                         'person_id':   meta.get('person_id',   -1),
#                         'start_frame': meta.get('start_frame', -1),
#                         'label_frame': meta.get('label_frame', -1),
#                         'true_label':  int(y_true[i].item()),
#                         'pred_label':  int(preds[i].item()),
#                         'prob_0':      float(probs0[i].item()),
#                         'prob_1':      float(probs1[i].item()),
#                     }
#                     results.append(entry)

#     return results

if __name__ == '__main__':
    evaluate()
    # model = STGCN(A_hat)
    # model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    # pos_windows = evaluate(model, json_path=TEST_JSON, batch_size=BATCH_SIZE, device=DEVICE, threshold=0.59)

    # # Save the results to a JSON file
    # with open("a_LSTM/pos_windows.json", "w") as f:
    #     json.dump(pos_windows, f, indent=2)

