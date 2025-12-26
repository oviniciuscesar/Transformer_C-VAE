import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==========================================
# CONFIGURAÇÕES
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS = 40
LR = 1e-3
VAL_SPLIT = 0.2
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# ==========================================
# CARREGAMENTO DOS DADOS
# ==========================================
# Substitua pelas variáveis reais do seu dataset
# src: tensor [N, frames, n_mels]
# labels: tensor [N]
data = torch.load("train_src.pt")   # ou o nome do arquivo que você salvou
labels = torch.load("train_labels.pt")

# Flatten temporal + espectral (ex.: 10x64 → 640)
N, T, F = data.shape
X = data.reshape(N, T * F)
y = labels.long()

print(f"Input shape: {X.shape}, Labels: {len(y)} classes={y.unique().numel()}")

# Normalização por feature
X = (X - X.mean(0)) / (X.std(0) + 1e-8)

# ==========================================
# DATASET SPLIT
# ==========================================
dataset = TensorDataset(X, y)
n_val = int(len(dataset) * VAL_SPLIT)
n_train = len(dataset) - n_val
train_ds, val_ds = random_split(dataset, [n_train, n_val])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# MODELO MLP
# ==========================================
class MLPClassifier(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)

num_classes = y.unique().numel()
model = MLPClassifier(X.shape[1], num_classes).to(DEVICE)

# ==========================================
# TREINAMENTO
# ==========================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
train_losses, val_losses = [], []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
    train_loss = running_loss / len(train_loader.dataset)

    # validação
    model.eval()
    val_loss = 0.0
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = model(xb)
            loss = criterion(preds, yb)
            val_loss += loss.item() * xb.size(0)
            correct += (preds.argmax(1) == yb).sum().item()
            total += yb.size(0)
    val_loss /= len(val_loader.dataset)
    acc = correct / total

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {acc:.3f}")

# ==========================================
# AVALIAÇÃO FINAL
# ==========================================
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for xb, yb in val_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        preds = model(xb).argmax(1)
        all_preds.append(preds.cpu())
        all_labels.append(yb.cpu())

all_preds = torch.cat(all_preds)
all_labels = torch.cat(all_labels)

print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds, digits=3))

# Matriz de confusão
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Curvas de perda
plt.figure()
plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Validation')
plt.legend()
plt.title("Loss Curves")
plt.show()
