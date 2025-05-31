import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# Configuration
DATASET = 'classification/data/data_small.json' 
OUTPUT_MODEL = 'classification/model/pose_classifier_64_small_drop.pth'


# Dataset class
class PoseDataset(Dataset):
    def __init__(self, data_file):
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        self.samples = []
        for item in self.data:
            keypoints = np.array(item['keypoints']).flatten().astype(np.float32)
            label = float(item['label'])
            self.samples.append((keypoints, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)

# Deeper model
class PoseClassifier(nn.Module):
    def __init__(self, input_dim):
        super(PoseClassifier, self).__init__()
        self.model = nn.Sequential(
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

# Modified training function to return loss
def train(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0
    for x, y in dataloader:
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs.view(-1), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# Evaluation
def evaluate(model, dataloader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in dataloader:
            outputs = model(x)
            preds = (outputs.view(-1) > 0.5).float()
            y_true.extend(y.numpy())
            y_pred.extend(preds.numpy())

    acc = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy Score: {acc}")
    print("\nClassification Report:\n", classification_report(y_true, y_pred))


def validate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in dataloader:
            outputs = model(x)
            loss = criterion(outputs.view(-1), y)
            total_loss += loss.item()
    return total_loss / len(dataloader)

# Main
def main():
    dataset = PoseDataset(DATASET)
    input_dim = len(dataset[0][0])

    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32)

    model = PoseClassifier(input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    val_losses = []

    for epoch in range(50):
        train_loss = train(model, train_loader, criterion, optimizer)
        val_loss = validate(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/50 - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Final evaluation
    evaluate(model, val_loader)
    torch.save(model.state_dict(), OUTPUT_MODEL)

    # Plot loss curves
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
