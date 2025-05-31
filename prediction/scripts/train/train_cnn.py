# projects/crossing/train_cnn.py

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# Config
WINDOW_SIZE = 10
FEATURE_DIM = 17 * 2   # raw(2) + delta(2) per joint
BATCH_SIZE  = 32
LR          = 1e-3
EPOCHS      = 40
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET = "prediction/data/sequences_5k.json"  # your 5k-window file
OUTPUT_MODEL_PATH =  "prediction/models/cnn/best_cnn_5k.pth"


# Dataset (same preprocessing of raw+Δ as before)
class SequenceDataset(Dataset):
    # def __init__(self, json_path, img_w=192, img_h=256):
    #     data = json.load(open(json_path))
    #     seqs, labels = [], []
    #     for e in data:
    #         raw = np.array(e["sequence"], dtype=np.float32).reshape(WINDOW_SIZE,17,2)
    #         # deltas
    #         deltas = raw[1:] - raw[:-1]
    #         deltas = np.vstack([np.zeros((1,17,2),dtype=np.float32), deltas])
    #         # normalize
    #         raw[...,0]   /= img_w;   raw[...,1]   /= img_h
    #         deltas[...,0]/= img_w;  deltas[...,1]/= img_h
    #         combo = np.concatenate([raw, deltas], axis=2)    # (10,17,4)
    #         seqs.append(combo.reshape(WINDOW_SIZE, FEATURE_DIM))  # (10,68)
    #         labels.append(float(e["label"]))
    #     self.X = np.stack(seqs)            # (N,10,68)
    #     self.y = np.array(labels, dtype=np.float32)

    # def __len__(self):
    #     return len(self.y)

    # def __getitem__(self, idx):
    #     return (
    #         torch.from_numpy(self.X[idx]),  # (10,68)
    #         torch.tensor(self.y[idx])       # scalar
    #     )
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

# 1D-CNN model
class Conv1DClassifier(nn.Module):
    # def __init__(self, feat_dim=FEATURE_DIM):
    #     super().__init__()
    #     self.net = nn.Sequential(
    #         nn.Conv1d(feat_dim, 128, kernel_size=3, padding=1),
    #         nn.ReLU(inplace=True),
    #         nn.Conv1d(128, 128, kernel_size=3, padding=1),
    #         nn.ReLU(inplace=True),
    #         nn.Conv1d(128, 128, kernel_size=3, padding=1),  # extra depth
    #         nn.ReLU(inplace=True),
    #         nn.AdaptiveAvgPool1d(1),
    #         nn.Flatten(),
    #         nn.Dropout(0.5),
    #         nn.Linear(128, 1)
    #      )

    # def forward(self, x):
    #     # x: (B,10,68)
    #     x = x.transpose(1,2)  # → (B,68,10)
    #     return self.net(x).view(-1)  # → (B,)
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

def train_and_evaluate():
    # 1) data
    ds = SequenceDataset(DATASET)
    n = len(ds)
    train_ds, val_ds = random_split(ds, [int(0.8*n), n-int(0.8*n)])
    tr = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    vl = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 2) model
    model = Conv1DClassifier().to(DEVICE)

    # 3) weighted BCE (same pos/neg as before)
    labels = ds.y
    pos = (labels == 1).sum()
    neg = (labels == 0).sum()
    pos_weight = torch.tensor([neg/pos], device=DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-2)

    best_acc = 0.0

    # 4) train
    for epoch in range(1, EPOCHS+1):
        model.train()
        running_loss = running_corr = running_tot = 0

        for seq, lbl in tr:
            seq, lbl = seq.to(DEVICE), lbl.to(DEVICE)
            optimizer.zero_grad()
            logits = model(seq)
            loss = criterion(logits, lbl)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * seq.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            running_corr += (preds == lbl).sum().item()
            running_tot  += seq.size(0)

        train_loss = running_loss / running_tot
        train_acc  = running_corr / running_tot

        # validation
        model.eval()
        y_true = []; y_pred = []
        with torch.no_grad():
            for seq, lbl in vl:
                seq = seq.to(DEVICE)
                logits = model(seq)
                preds = (torch.sigmoid(logits)>0.5).int().cpu().numpy()
                y_pred.extend(preds.tolist())
                y_true.extend(lbl.numpy().astype(int).tolist())
        val_acc = accuracy_score(y_true, y_pred)

        # save best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), OUTPUT_MODEL_PATH)
            print(f"→ New best val_acc={best_acc:.4f}, saving checkpoint")

        print(f"Epoch {epoch}/{EPOCHS}  "
              f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  "
              f"Val Acc: {val_acc:.4f}")

    # final eval
    print("\nLoading best model for final evaluation…")
    model.load_state_dict(torch.load(OUTPUT_MODEL_PATH))
    model.eval()
    print(classification_report(y_true, y_pred, digits=4))

if __name__ == "__main__":
    train_and_evaluate()
