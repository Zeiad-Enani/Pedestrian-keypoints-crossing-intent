import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

# ----- Config -----
IMG_W, IMG_H    = 192, 256
WINDOW_SIZE     = 10
NUM_JOINTS      = 17
IN_CHANNELS     = 4   # raw (x,y) + delta (dx,dy)
BATCH_SIZE      = 32
LR              = 1e-3
WEIGHT_DECAY    = 1e-2
EPOCHS          = 40
SEED            = 42
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET    = "prediction/data/sequences_v5f_enh_bal.json"  # your 10-window file
OUTPUT_MODEL_PATH = "prediction/models/stgcn/best_stgcn_v5f_enh_bal_enh.pth"

# ----- Skeleton Graph -----
EDGE_LIST = [
    (0,1),(1,2),(2,3),(3,4),
    (1,5),(2,6),
    (5,7),(7,9),
    (6,8),(8,10),
    (5,11),(6,12),
    (11,13),(13,15),
    (12,14),(14,16)
]

A = np.zeros((NUM_JOINTS, NUM_JOINTS), dtype=np.float32)
for i,j in EDGE_LIST:
    A[i,j] = A[j,i] = 1
np.fill_diagonal(A, 1)
D = np.sum(A, axis=1)
D_inv_sqrt = np.diag(1.0 / np.sqrt(D))
A_hat = torch.tensor(D_inv_sqrt @ A @ D_inv_sqrt, dtype=torch.float32)  # (V, V)

# ----- Dataset -----
class STGCNDataset(Dataset):
    def __init__(self, json_path):
        data = json.load(open(json_path))
        seqs, labels = [], []
        for e in data:
            raw = np.array(e['sequence'], dtype=np.float32).reshape(WINDOW_SIZE, NUM_JOINTS, 2)
            deltas = raw[1:] - raw[:-1]
            deltas = np.vstack([np.zeros((1,NUM_JOINTS,2), dtype=np.float32), deltas])
            raw[...,0]   /= IMG_W; raw[...,1]   /= IMG_H
            deltas[...,0]/= IMG_W; deltas[...,1]/= IMG_H
            combo = np.concatenate([raw, deltas], axis=2)  # (T, V, C)
            seqs.append(combo)
            labels.append(int(e['label']))
        self.X = torch.tensor(np.stack(seqs), dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ----- ST-GCN Block -----
class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, kernel_size=9, stride=1):
        super().__init__()
        self.register_buffer('A_hat', A)
        self.gcn = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        pad = (kernel_size - 1) // 2
        self.tcn = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size,1),
                      padding=(pad,0), stride=(stride,1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x: (N, C, T, V)
        x = torch.einsum('nctv,vw->nctw', x, self.A_hat)
        x = self.gcn(x)
        x = self.tcn(x)
        return x

# ----- ST-GCN Model -----
class STGCN(nn.Module):
    def __init__(self, A, in_channels=IN_CHANNELS, base_channels=64, num_classes=1):
        super().__init__()
        self.data_bn = nn.BatchNorm1d(in_channels * NUM_JOINTS)
        self.layers = nn.ModuleList([
            STGCNBlock(in_channels,       base_channels,    A),
            STGCNBlock(base_channels,     base_channels,    A),
            STGCNBlock(base_channels,     base_channels*2,  A, stride=2),
            STGCNBlock(base_channels*2,   base_channels*2,  A),
        ])
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc   = nn.Linear(base_channels*2, num_classes)

    def forward(self, x):
        # x: (N, T, V, C)
        N, T, V, C = x.size()
        x = x.permute(0,3,1,2).contiguous()   # (N, C, T, V)
        x = x.view(N, C * V, T)
        x = self.data_bn(x)
        x = x.view(N, C, T, V)
        for blk in self.layers:
            x = blk(x)
        x = self.pool(x).view(N, -1)
        return self.fc(x).view(-1)           # raw logits

# ----- Training Loop -----
def train():
    # reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    ds = STGCNDataset(DATASET)
    n = len(ds)
    train_ds, val_ds = random_split(ds, [int(0.8*n), n-int(0.8*n)])
    tr_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    vl_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = STGCN(A_hat).to(DEVICE)
    pos = (ds.y == 1).sum().item()
    neg = (ds.y == 0).sum().item()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg/pos], device=DEVICE))
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_acc = 0.0
    for epoch in range(1, EPOCHS+1):
        # -- Train --
        model.train()
        total_corr = total_loss = total = 0
        for x, y in tr_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            total_corr += (preds == y).sum().item()
            total += x.size(0)
        train_acc = total_corr / total

        # -- Validate --
        model.eval()
        all_true, all_preds = [], []
        with torch.no_grad():
            for x, y in vl_loader:
                x = x.to(DEVICE)
                logits = model(x)
                probs  = torch.sigmoid(logits).cpu().numpy()
                preds  = (probs > 0.5).astype(int)
                all_preds.extend(preds)
                all_true.extend(y.numpy().astype(int))

        val_acc = accuracy_score(all_true, all_preds)
        scheduler.step(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), OUTPUT_MODEL_PATH)
            print(f"→ New best val_acc={val_acc:.4f}, saving checkpoint")
        print(f"Epoch {epoch}/{EPOCHS}  Train Acc: {train_acc:.4f}  Val Acc: {val_acc:.4f}")

    # -- Final evaluation of best model --
    print("\nBest ST-GCN evaluation on val set:")
    model.load_state_dict(torch.load(OUTPUT_MODEL_PATH))
    model.eval()
    best_true, best_preds = [], []
    with torch.no_grad():
        for x, y in vl_loader:
            x = x.to(DEVICE)
            logits = model(x)
            probs  = torch.sigmoid(logits).cpu().numpy()
            preds  = (probs > 0.5).astype(int)
            best_preds.extend(preds)
            best_true.extend(y.numpy().astype(int))

    print(classification_report(best_true, best_preds, digits=4))

if __name__ == '__main__':
    train()
