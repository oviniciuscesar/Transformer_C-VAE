import os
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
import torchaudio
import torchaudio.transforms as T
import numpy as np
from pathlib import Path
import librosa
import json 
import random
import matplotlib.pyplot as plt

# --- Importações para t-SNE ---
try:
    from sklearn.manifold import TSNE
    import pandas as pd
    _PLOT_TSNE_ENABLED = True
except ImportError:
    _PLOT_TSNE_ENABLED = False
    print("Erro Crítico: 'sklearn' ou 'pandas' não encontrados.")
    print("Estes são necessários para a plotagem t-SNE.")
    print("Instale com: pip install scikit-learn pandas")
    exit(1)

# --- Importa o modelo ---
try:
    from TCVAEmodel import TransformerCVAE
except ImportError:
    print("Erro Crítico: Não foi possível encontrar 'TCVAEmodel.py'.")
    print("Este script (TCVAEtest.py) deve estar no mesmo diretório que TCVAEmodel.py.")
    exit(1)


# --- diretorios ---
SEED = 42
DIRECTORY = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(DIRECTORY, "TorchScript")
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "Checkpoints")
PLOTS_DIR = os.path.join(DIRECTORY, "plots")
DATASET_DIR = os.path.join(DIRECTORY, "dataset") # Diretório do dataset
os.makedirs(PLOTS_DIR, exist_ok=True)


DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'


# parameters do modelo
ENCODER_LAYERS = 2
DECODER_LAYERS = 1
ENCODER_DROPOUT = 0.1
DECODER_DROPOUT = 0.1
FINAL_PROJ_DROPOUT = 0.1
D_MODEL = 128
ENCODER_D_FF = 256
DECODER_D_FF = 64
NUM_HEADS = 4

 # features de entrada
# --- Parâmetros do Modelo (DEVE SER IDÊNTICO AO TREINAMENTO) ---
INPUT_FEATURES = 80
TARGET_FEATURES = 20
SEQ_LEN = 10 
MAX_POS = 10
LATENT_DIM = 64
N_FRAMES = 10 # Mantido para consistência (embora SEQ_LEN seja usado)

# --- Funções Auxiliares ---
def set_random_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_full_dataset(data_dir):
    """Carrega os tensores .pt completos (src, tgt, labels)."""
    print(f"Carregando dataset de {data_dir}...")
    src_path = os.path.join(data_dir, 'train_src.pt')
    tgt_path = os.path.join(data_dir, 'train_tgt.pt')
    label_path = os.path.join(data_dir, 'train_labels.pt')

    try:
        src_tensor = torch.load(src_path, map_location='cpu')
        tgt_tensor = torch.load(tgt_path, map_location='cpu')
        labels_tensor = torch.load(label_path, map_location='cpu')

        print("Tensores carregados com sucesso.")
        print(f"  SRC shape: {src_tensor.shape}")
        print(f"  TGT shape: {tgt_tensor.shape}")
        print(f"  LABELS shape: {labels_tensor.shape}")

        if not (src_tensor.shape[0] == tgt_tensor.shape[0] == labels_tensor.shape[0]):
            raise ValueError("Erro: SRC, TGT e LABELS têm número diferente de amostras!")

        dataset = TensorDataset(src_tensor, tgt_tensor, labels_tensor)
        return dataset

    except FileNotFoundError as e:
        print(f"Erro Crítico: Arquivo .pt não encontrado. Verifique o DATASET_DIR.")
        print(f"Tentativa de carregar: {e.filename}")
        exit(1)
    except Exception as e:
        print(f"Erro ao carregar ou criar TensorDataset: {e}")
        exit(1)


def plot_latent_space(model: TransformerCVAE, dataloader: DataLoader, device: torch.device, save_path_base: str, num_samples_max: int = 5000):
    """
    Executa t-SNE no espaço latente (mu) e plota os resultados coloridos por classe.
    """
    print("Iniciando plotagem do espaço latente (t-SNE)...")
    model.eval() # Coloca o modelo em modo de avaliação

    if not isinstance(dataloader, DataLoader):
        raise TypeError("plot_latent_space: 'dataloader' deve ser um torch.utils.data.DataLoader")

    all_mu = []
    all_labels = []

    print("Coletando vetores 'mu' do espaço latente...")
    with torch.no_grad():
        for i, (src, tgt, labels) in enumerate(dataloader):
            if i * dataloader.batch_size > num_samples_max:
                print(f"Limite de {num_samples_max} amostras atingido para t-SNE.")
                break
                
            src = src.to(device)
            # TGT ainda é necessário para o 'teacher forcing' do 'forward'
            # mesmo que o vae_encoder não o use mais (o decoder usa)
            tgt = tgt.to(device) 
            
            # Passa pelo modelo para obter mu (com a nova arquitetura)
            _ , mu, _ = model(src, tgt)

            all_mu.append(mu.cpu())
            all_labels.append(labels.cpu())

    mu_tensor = torch.cat(all_mu, dim=0).numpy()
    labels_vector = torch.cat(all_labels, dim=0).numpy()
    
    # Lida com o caso de termos menos amostras que o perplexity padrão (30)
    n_samples = float(mu_tensor.shape[0])
    perplexity_value = max(2.0, min(30.0, n_samples - 1.0))
    
    print(f"Executando t-SNE em {mu_tensor.shape[0]} amostras (Dim: {mu_tensor.shape[1]} -> 2)...")
    tsne = TSNE(n_components=2, perplexity=perplexity_value, max_iter=1000, learning_rate='auto', init='pca', random_state=SEED)
    z_tsne = tsne.fit_transform(mu_tensor)

    print("Plotando t-SNE...")
    df_tsne = pd.DataFrame({
        'tsne-1': z_tsne[:, 0],
        'tsne-2': z_tsne[:, 1],
        'label': labels_vector
    })

    plt.figure(figsize=(16, 10), dpi=200)
    plt.style.use("ggplot")
    
    # Mapeia labels (números) para um colormap
    unique_labels_in_plot = np.unique(labels_vector)
    num_classes = len(unique_labels_in_plot)
    cmap = plt.cm.get_cmap('jet', num_classes) # 'jet' ou 'tab20'
    
    # Mapeia labels de 0 a 13 para 0 a N-1 (para o colormap)
    label_to_color_index = {label: i for i, label in enumerate(sorted(unique_labels_in_plot))}
    colors = [label_to_color_index[l] for l in labels_vector]

    scatter = plt.scatter(
        df_tsne['tsne-1'], 
        df_tsne['tsne-2'], 
        c=colors, 
        cmap=cmap, 
        alpha=0.7,
        s=10 # Tamanho do ponto
    )
    
    plt.title('Visualização do Espaço Latente (Z) com t-SNE')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    
    # Tenta adicionar uma legenda com nomes (se o CSV estiver disponível)
    try:
        csv_path = os.path.join(DATASET_DIR, 'FluteMetadata.csv') 
        df_meta = pd.read_csv(csv_path)
        label_map_df = df_meta[['label', 'folder']].drop_duplicates().sort_values('label')
        label_names_map = dict(zip(label_map_df['label'], label_map_df['folder']))
        
        handles = []
        
        for label_id in sorted(unique_labels_in_plot):
            name = label_names_map.get(label_id, f"Label {label_id}")
            color_index = label_to_color_index[label_id]
            color = cmap(color_index / (num_classes - 1)) if num_classes > 1 else cmap(0.5)
            handles.append(plt.Line2D([0], [0], marker='o', color='w', label=f'{label_id}: {name}', 
                              markerfacecolor=color, markersize=8))
        
        plt.legend(handles=handles, title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='12', title_fontsize='13', frameon=True, borderaxespad=0.5)
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Ajusta para a legenda externa
    
    except Exception as e:
        print(f"Não foi possível carregar nomes de classes para a legenda ({e}). Usando legenda numérica.")
        # Legenda numérica padrão
        cbar = plt.colorbar(scatter, ticks=np.unique(colors))
        cbar.ax.set_yticklabels(sorted(unique_labels_in_plot)) # Mostra os IDs de label reais
        cbar.set_label('Classe Label ID')
        plt.tight_layout()

    plt.grid(True)
    
    try:
        png_path = save_path_base + ".png"
        svg_path = save_path_base + ".svg"
        pdf_path = save_path_base + ".pdf"
        plt.savefig(png_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
        plt.savefig(svg_path, format='svg', bbox_inches='tight')
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
        print(f"Gráfico do Espaço Latente salvo em: {png_path}, {svg_path}, {pdf_path}")
    except Exception as e:
        print(f"Erro ao salvar gráfico do espaço latente: {e}")


# --- Análise de Dimensões Ativas ---
def analyze_active_latent_dims(
    model: TransformerCVAE,
    dataloader: DataLoader,
    device: torch.device,
    var_threshold: float = 1e-3,
    max_samples: int = 50000,
    save_path: str | None = None,
):
    """
    Coleta mu do espaço latente ao longo do dataset e reporta:
    - variância por dimensão latente
    - número de dimensões ativas (variância > var_threshold)
    Opcionalmente salva um JSON com os resultados.
    """
    model.eval()
    mus = []
    n_collected = 0

    print("Analisando dimensões ativas do espaço latente...")   
    with torch.no_grad():
        for src, tgt, _ in dataloader:
            if n_collected >= max_samples:
                break
            src = src.to(device)
            tgt = tgt.to(device)
            # usa forward para obter mu/logvar (compatível com o seu modelo)
            _, mu, logvar = model(src, tgt)  # mu: (B, latent_dim)
            mus.append(mu.cpu())
            n_collected += mu.size(0)

            print(f"mu mean: {mu.mean().item():.4f} mu std: {mu.std().item():.4f}")
            print(f"logvar mean: {logvar.mean().item():.4f} logvar std: {logvar.std().item():.4f}")

    if not mus:
        print("Nenhum mu coletado para análise das dimensões ativas.")
        return None

    mus_all = torch.cat(mus, dim=0)  # (N, latent_dim) em CPU
    variances = mus_all.var(dim=0, unbiased=False)  # estável mesmo com poucos samples
    active_mask = variances > var_threshold
    active_count = int(active_mask.sum().item())
    latent_dim = mus_all.size(1)

    print("Variância por dimensão latente:")
    print(", ".join(f"{v:.6f}" for v in variances.tolist()))
    print(f"Dimensões ativas (> {var_threshold:g}): {active_count} / {latent_dim}")

    results = {
        "var_threshold": float(var_threshold),
        "latent_dim": int(latent_dim),
        "samples": int(mus_all.size(0)),
        "variances": [float(v) for v in variances.tolist()],
        "active_mask": [bool(b) for b in active_mask.tolist()],
        "active_count": int(active_count),
    }

    if save_path:
        try:
            import json
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Resultados salvos em: {save_path}")
        except Exception as e:
            print(f"Falha ao salvar resultados em {save_path}: {e}")

    return results


def main():
    set_random_seed(SEED)
    print(f"Usando dispositivo: {DEVICE}")

    # 1. Carregar o Dataset Completo
    dataset = load_full_dataset(DATASET_DIR)
    
    # Usar um batch size maior para acelerar a inferência
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
    print(f"Dataset carregado com {len(dataset)} amostras.")

    # 2. Inicializar o Modelo (com a arquitetura exata do treino)
    model = TransformerCVAE(
        num_layers_enc=ENCODER_LAYERS,
        num_layers_dec=DECODER_LAYERS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        encoder_d_ff=ENCODER_D_FF,
        decoder_d_ff=DECODER_D_FF,
        input_features=INPUT_FEATURES,
        target_features=TARGET_FEATURES,
        latent_dim=LATENT_DIM,
        max_pos=MAX_POS,
        encoder_dropout=ENCODER_DROPOUT,
        decoder_dropout=DECODER_DROPOUT,
        final_proj_dropout=FINAL_PROJ_DROPOUT,
    ).to(DEVICE)
    
    print("Arquitetura do modelo C-VAE criada.")

    # 3. Carregar os Pesos Treinados (o .pt, não o .ts)
    weights_path = os.path.join(CHECKPOINT_DIR, "model.pt")
    try:
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        print(f"Pesos do modelo carregados com sucesso de: {weights_path}")
    except FileNotFoundError:
        print(f"ERRO: Arquivo de pesos '{weights_path}' não encontrado.")
        print("Certifique-se de que o treinamento foi concluído e o 'model.pt' foi salvo.")
        return
    except Exception as e:
        print(f"Erro ao carregar state_dict do modelo: {e}")
        return
    
    # 4. Análise das dimensões ativas do espaço latente
    active_dims_path = os.path.join(PLOTS_DIR, "latent_active_dims.json")
    analyze_active_latent_dims(
        model=model,
        dataloader=dataloader,
        device=DEVICE,
        var_threshold=1e-3,
        max_samples=50000,
        save_path=active_dims_path,
    )

    # 5. Chamar a função de plotagem
    plot_save_path_base = os.path.join(PLOTS_DIR, "latent_Z")
    plot_latent_space(model, dataloader, DEVICE, plot_save_path_base)
    
    print("\nAnálise t-SNE concluída.")

if __name__ == "__main__":
    main()