import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import Tuple, Optional
from torch.utils.data import DataLoader, Dataset, TensorDataset
# Importa o modelo C-VAE
from TCVAEmodel import TransformerCVAE


# directories
DIRECTORY = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(DIRECTORY, "TorchScript")
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "Checkpoints")
DATASET_DIR = os.path.join(DIRECTORY, 'dataset') 

# Configs
EPOCHS = 100
BATCH_SIZE = 32
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
LR = 1e-3

# parameters
INPUT_FEATURES = 64  # features de entrada
TARGET_FEATURES = 20  # features alvo
SEQ_LEN = 10 # comprimento da sequência
MAX_POS = 100 # número máximo de passos temporais

# hiperparâmetros para o C-VAE
LATENT_DIM = 32   # Dimensão do espaço latente do VAE
BETA = 1.0        # Ponderação da perda KL (pode ser ajustado depois)


# Dummy Dataset class para geração de dados aleatórios para teste 
class DummyFeatureDataset(Dataset):
    """
    Gera dados aleatórios para simular um dataset de features.
    """
    def __init__(self, num_samples: int, seq_len: int, in_features: int, out_features: int):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.in_features = in_features
        self.out_features = out_features

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        src = torch.randn(self.seq_len, self.in_features)
        tgt = torch.randn(self.seq_len, self.out_features)
        return src, tgt
    
# carrega o dataset salvo em arquivos .pt
def load_pytorch_dataset(data_dir):
    """Carrega os tensores .pt e cria um TensorDataset."""
    print(f"Carregando dataset de {data_dir}...")
    src_path = os.path.join(data_dir, 'train_src.pt')
    tgt_path = os.path.join(data_dir, 'train_tgt.pt')
    # label_path = os.path.join(data_dir, 'train_labels.pt') # se precisar dos labels

    try:
        # Carrega os tensores para a CPU primeiro para evitar problemas de memória GPU
        src_tensor = torch.load(src_path, map_location='cpu')
        tgt_tensor = torch.load(tgt_path, map_location='cpu')
        # labels_tensor = torch.load(label_path, map_location='cpu') # Carrega se precisar

        print("Tensores carregados com sucesso.")
        print(f"  SRC shape: {src_tensor.shape}")
        print(f"  TGT shape: {tgt_tensor.shape}")

        # Verifica se os shapes são consistentes
        if src_tensor.shape[0] != tgt_tensor.shape[0]:
            raise ValueError("Erro: SRC e TGT têm número diferente de amostras!")
        if src_tensor.shape[1] != SEQ_LEN or src_tensor.shape[2] != INPUT_FEATURES:
             print(f"Aviso: Shape do SRC {src_tensor.shape} não bate com SEQ_LEN/INPUT_FEATURES ({SEQ_LEN}, {INPUT_FEATURES})")
        if tgt_tensor.shape[1] != SEQ_LEN or tgt_tensor.shape[2] != TARGET_FEATURES:
             print(f"Aviso: Shape do TGT {tgt_tensor.shape} não bate com SEQ_LEN/TARGET_FEATURES ({SEQ_LEN}, {TARGET_FEATURES})")


        # Cria o TensorDataset
        # Nota: O DataLoader moverá os batches para o DEVICE correto durante o treino
        dataset = TensorDataset(src_tensor, tgt_tensor)
        return dataset

    except FileNotFoundError as e:
        print(f"Erro Crítico: Arquivo .pt não encontrado. Verifique o DATASET_DIR.")
        print(f"Tentativa de carregar: {e.filename}")
        print("Certifique-se que o script TCVAEdataset.py foi executado com sucesso.")
        exit(1)
    except Exception as e:
        print(f"Erro ao carregar ou criar TensorDataset: {e}")
        exit(1)

# calculo da função de perda ELBO - Evidence Lower Bound
def _calculate_loss(real: torch.Tensor, pred: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float) -> torch.Tensor:
    """
    Calcula a perda ELBO (Reconstruction + KL-Divergence) para o C-VAE.
    real: (batch_size, T, features alvo) float (dados reais: dados de entrada)
    pred: (batch_size, T, features alvo) float (predições do modelo)
    mu: (batch_size, latent_dim) float (média do espaço latente)
    logvar: (batch_size, latent_dim) float (log-variância do espaço latente)
    beta: ponderação da perda KL (float)
    return: perda total ELBO (float)
    """
    # 1. Perda de Reconstrução (MSE)
    # Usar 'sum' (comum em VAEs para balancear as perdas)
    loss_fn = nn.MSELoss(reduction='sum')
    recon_loss = loss_fn(pred, real)

    # 2. Perda KL-Divergence (regularização do espaço latente)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Perda total = (Recon + beta * KL) / N_elementos_para_média
    # Dividimos pelo número de elementos na predição (batch_size*T*features alvo)
    total_loss = (recon_loss + (beta * kl_loss)) / pred.numel() 
    # Retorna a perda total d
    return total_loss

# Treinamento por época 
def train_one_epoch(dataloader: DataLoader, model: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device):
    model.train()
    total_loss = 0.0
    n_batches = 0

    # 1 - Loop sobre os batches
    for batch_idx, (src, tgt) in enumerate(dataloader):
        src = src.to(device)   # (batch, SEQ_LEN, INPUT_FEATURES)
        tgt = tgt.to(device)   # (batch, SEQ_LEN, TARGET_FEATURES)

        # prepare the input and target (teacher forcing)
        # tgt_input: (batch, SEQ_LEN-1, TARGET_FEATURES)
        tgt_input = tgt[:, :-1, :]   

        # tgt_real: (batch, SEQ_LEN-1, TARGET_FEATURES)
        tgt_real = tgt[:, 1:, :]     

        optimizer.zero_grad()

        # o modelo retorna 3 tensores: predictions [batch, SEQ_LEN-1, TARGET_FEATURES], mu [batch, latent_dim], logvar [batch, latent_dim]
        # passa 'src' (para ConditionalEncoder), 'tgt' (para VAEEncoder)
        predictions, mu, logvar = model(src=src, tgt=tgt)

        # Calcula a perda ELBO
        loss = _calculate_loss(tgt_real, predictions, mu, logvar, BETA)

        # Backpropagation e atualização dos pesos
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu().item())
        n_batches += 1
    return total_loss / max(1, n_batches)


# Função principal de treinamento- recebe: dataloader, modelo, número de épocas e dispositivo
def train(train_loader: DataLoader, model: torch.nn.Module, epochs: int, device: torch.device):
    
    # 1 - Otimizador Adam
    optimizer = Adam(model.parameters(), lr=LR)

    # 2 - Loop de treinamento
    for epoch in range(1, epochs + 1):
        # Treina uma época e obtém a perda média
        avg_loss = train_one_epoch(train_loader, model, optimizer, device)
        print(f"Epoch {epoch}/{epochs}  avg_loss={avg_loss:.6f}")


if __name__ == "__main__":
    # 1. prepare datasets (dummy for now)
    # print(f"Preparando dataset fictício (DummyFeatureDataset)...")
    # dataset = DummyFeatureDataset(
    #     num_samples=1000, # 1000 train samples
    #     seq_len=SEQ_LEN,
    #     in_features=INPUT_FEATURES,
    #     out_features=TARGET_FEATURES
    #)

    print(f"Carregando dataset dos arquivos .pt em '{DATASET_DIR}'...")
    dataset = load_pytorch_dataset(DATASET_DIR)
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Dataset carregado com {len(dataset)} amostras.")

    # 2. model (Instancia TransformerCVAE)
    model = TransformerCVAE(
        num_layers_enc=2, # Camadas para os encoders
        num_layers_dec=2, # Camadas para o decoder
        d_model=64, # Dimensão do modelo
        num_heads=2, # Número de cabeças de atenção
        d_ff=128, # Dimensão da camada de feedforward
        input_features=INPUT_FEATURES, # Dimensão das features de entrada
        target_features=TARGET_FEATURES, # Dimensão das features de saída (alvo)
        latent_dim=LATENT_DIM, # dimensão do espaço latente do VAE 
        max_pos=MAX_POS, # Posição máxima para embeddings
        dropout=0.1, # Dropout
    ).to(DEVICE) # Move para o dispositivo CPU/GPU
    
    print(f"Modelo Transformer C-VAE criado em {DEVICE}.")
    print(f"Input features: {INPUT_FEATURES}, Target features: {TARGET_FEATURES}, Latent dim: {LATENT_DIM}")

    # 3. treina o modelo pelo número de épocas definido
    print("Iniciando treinamento...")
    train(dataloader, model, EPOCHS, DEVICE)

    # 4. Salva os pesos do modelo treinado
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    save_path = os.path.join(CHECKPOINT_DIR, "model.pt")
    torch.save(model.state_dict(), save_path)
    print(f"State dict saved to {save_path}")