import os
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import Tuple, Optional, Dict, List
from torch.utils.data import DataLoader, Dataset, TensorDataset
# Importa o modelo C-VAE
from TCVAEmodel import TransformerCVAE

SEED = 42

# directories
DIRECTORY = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(DIRECTORY, "TorchScript")
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "Checkpoints")
DATASET_DIR = os.path.join(DIRECTORY, 'dataset')
PLOTS_DIR = os.path.join(DIRECTORY, "plots")

# Configs
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# parameters do modelo
ENCODER_LAYERS = 2
DECODER_LAYERS = 2
D_MODEL = 64
D_FF = 128
NUM_HEADS = 2
INPUT_FEATURES = 80  # features de entrada
TARGET_FEATURES = 20  # features alvo
SEQ_LEN = 10 # comprimento da sequência
MAX_POS = 100 # número máximo de passos temporais

# parâmetros de treinamento
EPOCHS = 100
BATCH_SIZE = 192
LR = 1e-4
CONDITION_DROPOUT_RATE = 0.1 # taxa de dropout para a condição (SRC)
LATENT_DIM = 64
BETA_START_EPOCH = 20
BETA_WARMUP_EPOCHS = 50
BETA_MAX = 0.2
FREE_BITS_PER_DIM = 0.02


def set_random_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # Tenta ativar algoritmos determinísticos no PyTorch (pode lançar se alguma operação não suportar)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

def _worker_init_fn(worker_id: int) -> None:
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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
def _calculate_loss(real: torch.Tensor, pred: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float, free_bits_per_dim: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    loss_fn = nn.MSELoss(reduction='mean')
    recon_loss = loss_fn(pred, real) # média sobre batch*seq*feat

    # 2. Perda KL-Divergence (regularização do espaço latente)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1) # (B)

    #3. Free bits regularization
    free_bits_nats = free_bits_per_dim * torch.log(torch.tensor(2.0))
    free_bits_nats = free_bits_nats.to(kl_loss_per_sample.device)  # move para o mesmo dispositivo
    kl_loss_clamped = torch.clamp(kl_loss_per_sample, min=free_bits_nats)
    # kl_loss_sum = torch.sum(kl_loss_clamped) # Soma sobre dims latentes

    # 4. Calcular perdas médias por elemento (para logging consistente)
    # num_elements = pred.numel() # B * (L-1) * F_out
    # recon_loss_avg = recon_loss_sum / num_elements
    # kl_loss_weighted_avg = (beta * kl_loss_sum) / num_elements   
    
    # # 5. Perda total ELBO
    # total_loss_avg = recon_loss_avg + kl_loss_weighted_avg

    kl_loss_avg = torch.mean(kl_loss_clamped) 

    # 4. Perda total ELBO (agora ambas são médias)
    total_loss = recon_loss + (beta * kl_loss_avg)

    # Retorna as médias (KL agora está PONDERADO por beta)
    return total_loss, recon_loss, (beta * kl_loss_avg)


# Treinamento por época 
def train_one_epoch(dataloader: DataLoader, model: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device, current_beta: float, free_bits_per_dim: float, condition_dropout_rate: float) -> Tuple[float, float, float, float, float]:
    model.train()
    total_loss_accum = 0.0
    recon_loss_accum = 0.0 
    kl_loss_accum = 0.0
    mu_mean_accum = 0.0  
    logvar_mean_accum = 0.0
    n_batches = 0

    # 1 - Loop sobre os batches
    for batch_idx, (src, tgt) in enumerate(dataloader):
        src = src.to(device)   # (batch, SEQ_LEN, INPUT_FEATURES)
        tgt = tgt.to(device)   # (batch, SEQ_LEN, TARGET_FEATURES)

        # prepare the input and target (teacher forcing)
        # tgt_input: (batch, SEQ_LEN-1, TARGET_FEATURES)
        tgt_input = tgt[:, :-1, :]   

        if model.training and torch.rand(1).item() < condition_dropout_rate:
            # Em 20% das vezes, "desliga" src e tgt_input (substitui por zeros)
            src = torch.zeros_like(src)
            tgt_input = torch.zeros_like(tgt_input)


        # tgt_real: (batch, SEQ_LEN-1, TARGET_FEATURES)
        tgt_real = tgt[:, 1:, :]     

        optimizer.zero_grad()

        # o modelo retorna 3 tensores: predictions [batch, SEQ_LEN-1, TARGET_FEATURES], mu [batch, latent_dim], logvar [batch, latent_dim]
        # passa 'src' (para ConditionalEncoder), 'tgt' (para VAEEncoder)
        predictions, mu, logvar = model(src=src, tgt=tgt)

        # Obtém os componentes da perda
        total_loss, recon_loss, kl_loss_w = _calculate_loss(
            tgt_real, predictions, mu, logvar, 
            current_beta, free_bits_per_dim
        )

        # Usa a perda total para backpropagation
        total_loss.backward()
        optimizer.step()
        
        # Acumula as perdas para a média da época
        total_loss_accum += total_loss.detach().cpu().item() # Acumula a perda total
        recon_loss_accum += recon_loss.detach().cpu().item() # Acumula a perda de reconstrução
        kl_loss_accum += kl_loss_w.detach().cpu().item()  # Acumula a perda KL ponderada
        mu_mean_accum += mu.detach().mean().cpu().item() # acumula média de mu sobre batch e dim latente
        logvar_mean_accum += logvar.detach().mean().cpu().item() # acumula média de logvar sobre batch e dim latente
        n_batches += 1
        
    # Retorna as médias das três perdas e médias de mu/logvar sobre batches
    num_batches_safe = max(1, n_batches)
    return (total_loss_accum / num_batches_safe, recon_loss_accum / num_batches_safe, kl_loss_accum / num_batches_safe, mu_mean_accum / num_batches_safe,
            logvar_mean_accum / num_batches_safe)


def plot_losses(history: Dict[str, List[float]], save_path: str):
    """Gera e salva um gráfico das perdas e médias de mu/logvar."""
    epochs = range(1, len(history['total_loss']) + 1)
    
    # fig, ax1 = plt.subplots(figsize=(14, 7)) 
    plt.style.use("ggplot")
    fig, ax1 = plt.subplots(figsize=(14, 7), dpi=200)

    # Eixo Y Primário (Esquerda) - Perdas
    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Average Loss (Log Scale)', color=color)
    ax1.plot(epochs, history['total_loss'], color=color, linestyle='-', label='Total Loss (Avg ELBO)')
    ax1.plot(epochs, history['recon_loss'], color='tab:green', linestyle='--', label='Reconstruction Loss (Avg MSE)')
    ax1.plot(epochs, history['kl_loss'], color='tab:red', linestyle=':', label='Weighted KL Loss (Avg Beta*KL)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_yscale('log') # Escala Log para perdas
    ax1.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)
    ax1.legend(loc='upper left')

    # Eixo Y Secundário (Direita) - Médias de mu e logvar
    ax2 = ax1.twinx()  # Cria um segundo eixo compartilhando o mesmo eixo x
    color = 'tab:purple'
    ax2.set_ylabel('Average mu / logvar', color=color)  
    ax2.plot(epochs, history['mu_mean'], color=color, linestyle='-.', label='Avg mu')
    ax2.plot(epochs, history['logvar_mean'], color='tab:orange', linestyle='-.', label='Avg logvar')
    ax2.tick_params(axis='y', labelcolor=color)
    # Linha de referência em y=0 para mu e logvar
    ax2.axhline(0, color='grey', linestyle='--', linewidth=0.8, label='Zero Reference') 
    ax2.legend(loc='upper right')

    fig.tight_layout()  # Ajusta o layout para evitar sobreposição
    plt.title('Training Metrics per Epoch')
    
    try:
        base, _ = os.path.splitext(save_path)
        png_path = base + ".png"
        svg_path = base + ".svg"
        pdf_path = base + ".pdf"
        fig.savefig(png_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
        fig.savefig(svg_path, format='svg', bbox_inches='tight')
        fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
        print(f"Gráfico de métricas salvo em: {save_path}")
    except Exception as e:
        print(f"Erro ao salvar gráfico: {e}")


# Função principal de treinamento- recebe: dataloader, modelo, número de épocas e dispositivo
def train(train_loader: DataLoader, model: torch.nn.Module, epochs: int, device: torch.device):
    # 1 - Otimizador Adam
    optimizer = Adam(model.parameters(), lr=LR)
   
    print(f"Iniciando Beta Annealing: Start={BETA_START_EPOCH}, Warmup={BETA_WARMUP_EPOCHS}, Max={BETA_MAX}")
    print(f"Usando Free Bits por Dimensão: {FREE_BITS_PER_DIM} nats")
    print(f"Usando Condition Dropout (SRC & TGT_in): {CONDITION_DROPOUT_RATE * 100:.0f}%")

    # Dicionário para armazenar o histórico das perdas
    history: Dict[str, List[float]] = {
        'total_loss': [],
        'recon_loss': [],
        'kl_recon_ratio': [],
        'kl_loss': [],
        'mu_mean': [],  
        'logvar_mean': []
    }

    # 2 - Loop de treinamento
    for epoch in range(1, epochs + 1):
        
        # lógica de beta annealing
        if epoch < BETA_START_EPOCH:
            current_beta = 0.0
        else:
            # Cálculo linear simples de warmup
            progress = (epoch - BETA_START_EPOCH) / BETA_WARMUP_EPOCHS
            current_beta = min(progress * BETA_MAX, BETA_MAX)

        # Treina uma época e obtém as três perdas médias e médias de mu/logvar
        avg_total_loss, avg_recon_loss, avg_kl_loss, avg_mu_mean, avg_logvar_mean = train_one_epoch(
            train_loader, model, optimizer, device, 
            current_beta, FREE_BITS_PER_DIM, CONDITION_DROPOUT_RATE)
        
        kl_recon_ratio = (avg_kl_loss / avg_recon_loss) if avg_recon_loss > 0 else 0.0
        
        # Armazena as perdas no histórico
        history['total_loss'].append(avg_total_loss)
        history['recon_loss'].append(avg_recon_loss)
        history['kl_recon_ratio'].append(kl_recon_ratio)
        history['kl_loss'].append(avg_kl_loss)
        history['mu_mean'].append(avg_mu_mean)
        history['logvar_mean'].append(avg_logvar_mean)

        # Imprime as perdas e métricas da época
        print(f"Epoch {epoch}/{epochs} | Loss={avg_total_loss:.6f} | Recon={avg_recon_loss:.6f} | KL={avg_kl_loss:.6f} | KL/Recon ratio={kl_recon_ratio:.4f} | Mu={avg_mu_mean:.4f} | LogVar={avg_logvar_mean:.4f} | beta={current_beta:.4f}")

    # Após o treino, gera o gráfico
    plot_save_path = os.path.join(PLOTS_DIR, "training_metrics.png") # Salva em plots
    plot_losses(history, plot_save_path)


if __name__ == "__main__":
    # Define a seed para reprodutibilidade
    set_random_seed(SEED)

    print(f"Usando dispositivo: {DEVICE}")

    print(f"Carregando dataset dos arquivos .pt em '{DATASET_DIR}'...")
    dataset = load_pytorch_dataset(DATASET_DIR)

    g = torch.Generator()
    g.manual_seed(SEED)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, worker_init_fn=_worker_init_fn, generator=g)
    print(f"Dataset carregado com {len(dataset)} amostras.")

    # 2. model (Instancia TransformerCVAE)
    model = TransformerCVAE(
        num_layers_enc=ENCODER_LAYERS, # Camadas para os encoders
        num_layers_dec=DECODER_LAYERS, # Camadas para o decoder
        d_model=D_MODEL, # Dimensão do modelo
        num_heads=NUM_HEADS, # Número de cabeças de atenção
        d_ff=D_FF, # Dimensão da camada de feedforward
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
