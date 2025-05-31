import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
import numpy as np

# ----- Config -----
IMG_W, IMG_H    = 192, 256
WINDOW_SIZE     = 10
FEATURE_DIM     = 17 * 4   # raw + delta
BATCH_SIZE      = 32
INITIAL_LR      = 1e-3
WEIGHT_DECAY    = 1e-2
EPOCHS          = 40
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET    = "prediction/data/sequences_10_filtered.json"  # your 10-window file
OUTPUT_MODEL_PATH = "prediction/models/lstm/best_lstm_10k_filtered.pth"

# ----- Dataset -----
class SequenceDataset(Dataset):
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
            labels.append(float(e["label"]))
        self.X = np.stack(seqs)
        self.y = np.array(labels, dtype=np.float32)

    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])

# ----- Model -----
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

# ----- Train & Eval -----
def train_and_evaluate():
    # Data
    ds = SequenceDataset(DATASET)
    n_train = int(0.8 * len(ds))
    train_ds, val_ds = random_split(ds, [n_train, len(ds)-n_train])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    # Model
    model = BiLSTMClassifier().to(DEVICE)

    # Loss with updated pos_weight
    pos = (ds.y == 1).sum()
    neg = (ds.y == 0).sum()
    pos_weight = torch.tensor([neg/pos], device=DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer + scheduler
    optimizer = optim.Adam(model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    best_acc = 0.0

    for epoch in range(1, EPOCHS+1):
        # Train
        model.train()
        total_loss = total_corr = total_samples = 0
        for seq, lbl in train_loader:
            seq, lbl = seq.to(DEVICE), lbl.to(DEVICE)
            optimizer.zero_grad()
            logits = model(seq)
            loss   = criterion(logits, lbl)
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss    += loss.item() * seq.size(0)
            preds          = (torch.sigmoid(logits) > 0.5).float()
            total_corr    += (preds == lbl).sum().item()
            total_samples += seq.size(0)

        train_loss = total_loss / total_samples
        train_acc  = total_corr / total_samples

        # Validate
        model.eval()
        all_true, all_probs = [], []
        with torch.no_grad():
            for seq, lbl in val_loader:
                seq = seq.to(DEVICE)
                logits = model(seq)
                all_probs.extend(torch.sigmoid(logits).cpu().tolist())
                all_true .extend(lbl.tolist())

        # Default val_acc at threshold 0.5
        preds_05 = [p>0.5 for p in all_probs]
        val_acc  = accuracy_score(all_true, preds_05)

        # Scheduler step
        scheduler.step(val_acc)

        # Checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), OUTPUT_MODEL_PATH)
            print(f"→ New best val_acc={best_acc:.4f}, saving checkpoint")

        print(
            f"Epoch {epoch}/{EPOCHS}  "
            f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  "
            f"Val Acc: {val_acc:.4f}"
        )

    # Final evaluation + threshold sweep
    print("\nLoading best model for final evaluation…")
    model.load_state_dict(torch.load(OUTPUT_MODEL_PATH))
    model.eval()

    best_macro_f1 = 0.0
    best_thr = 0.5
    for thr in np.linspace(0.1, 0.9, 81):
        preds = np.array(all_probs) > thr
        f1_0  = f1_score(all_true, preds, pos_label=0)
        f1_1  = f1_score(all_true, preds, pos_label=1)
        macro = (f1_0 + f1_1) / 2
        if macro > best_macro_f1:
            best_macro_f1, best_thr = macro, thr

    print(f"Best threshold (macro-F1) = {best_thr:.2f}, macro-F1 = {best_macro_f1:.4f}")
    print(classification_report(all_true, [p>best_thr for p in all_probs], digits=4))

if __name__ == "__main__":
    train_and_evaluate()
