# Train_Regression.py
# (Baseado em Train.py)

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import Tuple, Optional
from torch.utils.data import DataLoader, Dataset
from TransformerModel import TransformerModel


# directories
DIRECTORY = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(DIRECTORY, "TorchScript")
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "Checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Configs
EPOCHS = 100
BATCH_SIZE = 32
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
LR = 1e-3

# parameters
INPUT_FEATURES = 64  # (inharmonicity, spectral flux, zcr, energy entropy)
TARGET_FEATURES = 5  # (pitch midi cents, amp midi, grain_size ms, 4 metro ms, event duration ms)
SEQ_LEN = 10 # 10 frames/events as input
MAX_POS = 100 # Maximum positional encoding (must be >= SEQ_LEN)


# Dummy Dataset class for testing
class DummyFeatureDataset(Dataset):
    """
    Gera dados aleatórios para simular seu dataset de features.
    Cada amostra é um par (src, tgt).
    - src: (SEQ_LEN, INPUT_FEATURES)
    - tgt: (SEQ_LEN, TARGET_FEATURES)
    """
    def __init__(self, num_samples: int, seq_len: int, in_features: int, out_features: int):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.in_features = in_features
        self.out_features = out_features

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Gera um gesto de flauta aleatório
        # (Em um dataset real, você carregaria seus dados pré-processados aqui)
        src = torch.randn(self.seq_len, self.in_features)
        
        # Gera a resposta eletrônica "alvo" correspondente
        tgt = torch.randn(self.seq_len, self.out_features)
        
        return src, tgt


# loss function calculation
def _calculate_loss(real: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """
    Calcula a perda de regressão (MSE) entre predições e alvos.
    real: (B, T, F_out) float (alvos)
    pred: (B, T, F_out) float (predições)
    (NOTA: Uma implementação real precisaria de um 'mask' 
     para ignorar posições de padding, caso suas sequências
     tenham comprimentos variáveis.)
    """
    loss_fn = nn.MSELoss() 
    return loss_fn(pred, real)


def train_one_epoch(dataloader: DataLoader, model: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device):
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for batch_idx, (src, tgt) in enumerate(dataloader):
        src = src.to(device)   # (B, SEQ_LEN, INPUT_FEATURES)
        tgt = tgt.to(device)   # (B, SEQ_LEN, TARGET_FEATURES)

        # prepare the input and target (teacher forcing)
        # tgt_input: (B, SEQ_LEN-1, TARGET_FEATURES) - events 0 a 9
        tgt_input = tgt[:, :-1, :]   

        # tgt_real: (B, SEQ_LEN-1, TARGET_FEATURES) - events 1 a 10
        # the model will try to predict 'tgt_real'
        tgt_real = tgt[:, 1:, :]     

        optimizer.zero_grad()
        
        # 'src' feature tensor (B, 10, 8)
        # O 'tgt_input' é o tensor de targets (B, 9, 8)
        predictions = model(src, tgt_input)   # Saída: (B, 9, 8)

        # MSE Loss
        loss = _calculate_loss(tgt_real, predictions)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu().item())
        n_batches += 1
    return total_loss / max(1, n_batches)

# Main training loop
def train(train_loader: DataLoader, model: torch.nn.Module, epochs: int, device: torch.device):
    optimizer = Adam(model.parameters(), lr=LR)
    for epoch in range(1, epochs + 1):
        avg_loss = train_one_epoch(train_loader, model, optimizer, device)
        print(f"Epoch {epoch}/{epochs}  avg_loss={avg_loss:.6f}")


if __name__ == "__main__":
    # 1. prepare datasets (dummy for now)
    print(f"Preparando dataset fictício (DummyFeatureDataset)...")
    dataset = DummyFeatureDataset(
        num_samples=1000, # 1000 train samples
        seq_len=SEQ_LEN,
        in_features=INPUT_FEATURES,
        out_features=TARGET_FEATURES
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Dataset criado com {len(dataset)} amostras.")
 
    # 2. model
    model = TransformerModel(
        num_layers=2,
        d_model=64,
        num_heads=2,
        d_ff=128,
        input_features=INPUT_FEATURES,
        target_features=TARGET_FEATURES,
        max_pos=MAX_POS,
        dropout=0.1,
    ).to(DEVICE)
    
    print(f"Modelo Transformer criado em {DEVICE}.")
    print(f"Input features: {INPUT_FEATURES}, Target features: {TARGET_FEATURES}")

    # 3. training
    print("Iniciando treinamento...")
    train(dataloader, model, EPOCHS, DEVICE)

    # 4. Save weights
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    save_path = os.path.join(CHECKPOINT_DIR, "model.pt")
    torch.save(model.state_dict(), save_path)
    print(f"State dict saved to {save_path}")