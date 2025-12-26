import torch
import os
import numpy as np
from tqdm import tqdm
from TCVAEmodel import TransformerCVAE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# --- CONFIGURAÇÕES ---
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
MODEL_PATH = "TorchScript/Checkpoints/model.pt"

DIRECTORY = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(DIRECTORY, "TorchScript")
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "Checkpoints")
DATA_DIR = os.path.join(DIRECTORY, "dataset") 
PLOTS_DIR = os.path.join(DIRECTORY, "plots")
RAW_AUDIO_DIR = "Flute" 

# Fator de expansão: 2.5 garante que os vetores tenham força (~ -2.5 a +2.5)
# similar a um vetor aleatório torch.randn()
EXPANSION_FACTOR = 2.5

os.makedirs(PLOTS_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(CHECKPOINT_DIR, "latent_centroids.txt")

# Arquitetura
MODEL_CONFIG = {
    'num_layers_enc': 4, 'num_layers_vae': 2, 'num_layers_dec': 4, 
    'd_model': 128, 'num_heads': 4, 'decoder_num_heads': 4,
    'encoder_d_ff': 256, 'vaencoder_d_ff': 256, 'decoder_d_ff': 256, 
    'input_features': 80, 'target_features': 20, 'latent_dim': 64, 'max_pos': 10
}

def load_model():
    print(f"Carregando modelo em {DEVICE}...")
    model = TransformerCVAE(**MODEL_CONFIG).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

def get_class_names():
    if os.path.exists(RAW_AUDIO_DIR):
        return sorted([d for d in os.listdir(RAW_AUDIO_DIR) if os.path.isdir(os.path.join(RAW_AUDIO_DIR, d))])
    return [f"class_{i}" for i in range(20)]

def main():
    model = load_model()
    
    src_path = os.path.join(DATA_DIR, "train_src.pt")
    labels_path = os.path.join(DATA_DIR, "train_labels.pt")
    
    print(f"Carregando dataset de: {src_path}")
    try:
        full_src = torch.load(src_path, map_location='cpu') 
        full_labels = torch.load(labels_path, map_location='cpu')
    except FileNotFoundError:
        print(f"ERRO: Arquivos não encontrados.")
        return

    if full_labels.dim() > 1 and full_labels.shape[1] == 1:
        full_labels = full_labels.squeeze()

    # --- SEM CONVERSÃO DB (Respeitando o treinamento Linear) ---
    print("Usando dados RAW (Linear) para extração...")
    
    # Verifica dimensões e transpõe se necessário [Batch, Time, Feat]
    if full_src.shape[1] == 80 and full_src.shape[2] == 10:
        full_src = full_src.transpose(1, 2)
    
    unique_labels = sorted([int(x) for x in torch.unique(full_labels).tolist() if x >= 0])
    class_names = get_class_names()

    # Dicionário temporário para guardar os vetores brutos (microscópicos)
    raw_centroids = {}
    
    with torch.no_grad():
        for label_idx in tqdm(unique_labels, desc="Extraindo Centróides"):
            indices = (full_labels == label_idx).nonzero(as_tuple=True)[0]
            if len(indices) == 0: continue
            
            class_src = full_src[indices].to(DEVICE)
            
            z_accum = []
            # Processa em batches
            for i in range(0, len(class_src), 512):
                batch = class_src[i : i + 512]
                C = model.conditional_encoder(batch, None)
                mu, _ = model.prior(C)
                z_accum.append(mu.cpu())
            
            all_z = torch.cat(z_accum, dim=0)
            centroid = torch.mean(all_z, dim=0).numpy()
            
            name = class_names[label_idx] if label_idx < len(class_names) else f"class_{label_idx}"
            raw_centroids[label_idx] = (name, centroid)

    # --- EXPANSÃO DO ESPAÇO LATENTE ---
    print("\n--- Aplicando Expansão Estatística ---")
    
    # Prepara dados para o Scaler
    # Ordena por label_idx para consistência
    sorted_items = sorted(raw_centroids.items())
    vectors_raw = np.array([vec for _, (_, vec) in sorted_items])
    names_list = [name for _, (name, _) in sorted_items]
    ids_list = [idx for idx, _ in sorted_items]

    std_raw = np.std(vectors_raw)
    print(f"Desvio Padrão Original (Microscópico): {std_raw:.10f}")

    # Aplica StandardScaler (Média 0, Std 1)
    scaler = StandardScaler()
    vectors_expanded = scaler.fit_transform(vectors_raw)
    
    # Aplica fator de multiplicação (Força Extra)
    vectors_expanded = vectors_expanded * EXPANSION_FACTOR
    
    std_new = np.std(vectors_expanded)
    print(f"Novo Desvio Padrão (Expandido): {std_new:.4f}")
    print(f"Fator de Expansão aplicado: {EXPANSION_FACTOR}x (sobre Std=1)")

    # 3. Salvar (Agora salvamos os vetores EXPANDIDOS)
    print(f"Salvando em {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        for i, idx in enumerate(ids_list):
            name = names_list[i]
            vec = vectors_expanded[i]
            # Formato: ID NOME V1 V2 ...
            vec_str = " ".join([f"{v:.6f}" for v in vec])
            # f.write(f"{idx} {name} {vec_str};\n")
            f.write(f"{vec_str};\n")

    # 4. Plotagem (Para o PD)
    print("Gerando gráfico de navegação...")
    
    # PCA para 2D (preserva a geometria relativa dos vetores expandidos)
    pca = PCA(n_components=2)
    coords_pca = pca.fit_transform(vectors_expanded)
    
    # Normalização Forçada para 0-1 (Independente da expansão, o PD precisa de 0-1)
    minmax = MinMaxScaler(feature_range=(0, 1))
    coords_norm = minmax.fit_transform(coords_pca)

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(coords_norm[:, 0], coords_norm[:, 1], c=range(len(names_list)), cmap='jet', s=250, edgecolors='black', alpha=0.8)
    
    for i, name in enumerate(names_list):
        plt.text(coords_norm[i, 0]+0.02, coords_norm[i, 1], f"{i}: {name}", fontsize=10, weight='bold')

    plt.title(f"Mapa de Navegação Expandido (Fator {EXPANSION_FACTOR}x)\nUse estas coordenadas no [nodes]", fontsize=14)
    plt.xlabel("X (0-1)")
    plt.ylabel("Y (0-1)")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plot_path = os.path.join(PLOTS_DIR, "mapa2d_z.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nGráfico salvo em: {plot_path}")
    
    print("\n" + "="*50)
    print("COORDENADAS PARA O PURE DATA [nodes]")
    print("(A geometria é a mesma, mas os vetores no arquivo .txt agora têm força!)")
    print("="*50)
    for i, name in enumerate(names_list):
        # Imprime com bastante precisão
        print(f"Classe {i} ({name}): \tX = {coords_norm[i, 0]:.4f} \tY = {coords_norm[i, 1]:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()