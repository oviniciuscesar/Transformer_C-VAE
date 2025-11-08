import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import argparse

def load_dataset(path=None):
    """Carrega dataset salvo em formato .pt (dict com src, tgt, label)."""
    src = torch.load('train_src.pt')
    tgt = torch.load('train_tgt.pt')
    labels = torch.load('train_labels.pt')
    return src, tgt, labels

def flatten(tensor):
    """Transforma (N, T, F) -> (N, T*F) para PCA/t-SNE."""
    if len(tensor.shape) > 2:
        tensor = tensor.view(tensor.shape[0], -1)
    return tensor

def reduce_features(x, method='pca', n_components=2, seed=42):
    """Aplica PCA ou t-SNE nas features."""
    x_np = x.detach().cpu().numpy()
    if method == 'pca':
        reducer = PCA(n_components=n_components, random_state=seed)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=seed, perplexity=30, learning_rate=200)
    else:
        raise ValueError("Método deve ser 'pca' ou 'tsne'")
    return reducer.fit_transform(x_np)

def plot_reduction(src_2d, tgt_2d, labels, method='pca', save=None):
    """Plota resultados lado a lado."""
    num_classes = len(torch.unique(labels))
    labels_np = labels.cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cmap = plt.cm.get_cmap('tab10', num_classes)

    for i in range(num_classes):
        mask = labels_np == i
        axes[0].scatter(src_2d[mask, 0], src_2d[mask, 1], s=8, color=cmap(i), label=f'class {i}')
        axes[1].scatter(tgt_2d[mask, 0], tgt_2d[mask, 1], s=8, color=cmap(i))

    axes[0].set_title(f'{method.upper()} – SRC features')
    axes[1].set_title(f'{method.upper()} – TGT features')
    for ax in axes: 
        ax.set_xticks([]); ax.set_yticks([])
    axes[0].legend(loc='best', fontsize=8)
    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=300)
    plt.show()


def separability_analysis(X, y, name="dataset"):
    # Reduz para 10 dimensões no máximo (evita ruído)

    if isinstance(X, torch.Tensor):
        if X.dim() > 2:
            X = X.view(X.size(0), -1)
        X_np = X.detach().cpu().numpy()
    else:
        X_np = np.asarray(X)

    if isinstance(y, torch.Tensor):
        y_np = y.detach().cpu().numpy().ravel()
    else:
        y_np = np.asarray(y).ravel()
    n_components = min(10, X.shape[1])
    X_red = PCA(n_components=n_components).fit_transform(X)

    # Calcular métricas de separabilidade
    silhouette = silhouette_score(X_red, y)
    calinski = calinski_harabasz_score(X_red, y)
    davies = davies_bouldin_score(X_red, y)

    # Mostrar resultados
    print(f"\n{name.upper()} separability metrics:")
    print(f"  • Silhouette Score:       {silhouette:.3f}  (↑ melhor, entre -1 e 1)")
    print(f"  • Calinski–Harabasz:      {calinski:.3f}  (↑ melhor)")
    print(f"  • Davies–Bouldin:         {davies:.3f}  (↓ melhor)")

    # Interpretação qualitativa simples
    if silhouette < 0.2:
        print("  → Clusters muito sobrepostos (baixa separabilidade).")
    elif silhouette < 0.5:
        print("  → Clusters parcialmente separados.")
    else:
        print("  → Clusters bem definidos e separados.")

    # Retornar métricas
    return silhouette, calinski, davies

def stats(src: torch.Tensor, tgt: torch.Tensor, labels: torch.Tensor):
    """
    Estatísticas de tgt:
    - média e std globais por feature
    - média e std por classe (agregando no tempo)
    Retorna um dict com tensores (no device de tgt).
    """
    assert tgt.dim() in (2, 3), "tgt deve ter shape (N,T,F) ou (N,F)"
    if tgt.dim() == 2:
        tgt_time = tgt.unsqueeze(1)  # (N,1,F)
    else:
        tgt_time = tgt

    # Globais (achata tempo)
    tgt_flat = tgt_time.reshape(-1, tgt_time.size(-1))  # (N*T, F)
    global_mean = tgt_flat.mean(dim=0)
    global_std = tgt_flat.std(dim=0, unbiased=False)  # robusto quando N*T=1

    print("Média global por feature:", [f"{v:.4f}" for v in global_mean.detach().cpu().tolist()])
    print("Std global por feature:  ", [f"{v:.4f}" for v in global_std.detach().cpu().tolist()])

    results = {"global": {"mean": global_mean, "std": global_std}, "per_class": {}}

    # Por classe
    for c in torch.unique(labels).tolist():
        mask = (labels == c)
        tsel = tgt_time[mask]  # (Nc, T, F)
        if tsel.numel() == 0:
            continue
        tsel_flat = tsel.reshape(-1, tsel.size(-1))  # (Nc*T, F)
        mean_c = tsel_flat.mean(dim=0)
        std_c = tsel_flat.std(dim=0, unbiased=False)

        print(f"Classe {int(c)} -> mean:", [f"{v:.4f}" for v in mean_c.detach().cpu().tolist()])
        print(f"Classe {int(c)} -> std: ", [f"{v:.4f}" for v in std_c.detach().cpu().tolist()])

        results["per_class"][int(c)] = {"mean": mean_c, "std": std_c}

    return results
    


def main():
    parser = argparse.ArgumentParser(description="Visualiza SRC/TGT via PCA ou t-SNE")
    parser.add_argument('--method', type=str, default='pca', choices=['pca', 'tsne'], help='Método de redução')
    parser.add_argument('--save', type=str, default=None, help='Arquivo de saída opcional (.png)')
    args = parser.parse_args()

    src, tgt, labels = load_dataset(None)
    analise = stats(src, tgt, labels)

    src_flat = flatten(src)
    tgt_flat = flatten(tgt)

    src_2d = reduce_features(src_flat, method=args.method)
    tgt_2d = reduce_features(tgt_flat, method=args.method)
    plot_reduction(src_2d, tgt_2d, labels, method=args.method, save=args.save)

    src_scores = separability_analysis(src_flat, labels, name="SRC")
    tgt_scores = separability_analysis(tgt_flat, labels, name="TGT")


    metric_names = ["Silhouette (↑)", "Calinski–Harabasz (↑)", "Davies–Bouldin (↓)"]
    src_vals = [src_scores[0], src_scores[1]/1000, src_scores[2]]  # normalização grosseira
    tgt_vals = [tgt_scores[0], tgt_scores[1]/1000, tgt_scores[2]]

    x = np.arange(len(metric_names))
    width = 0.35

    plt.figure(figsize=(8,4))
    plt.bar(x - width/2, src_vals, width, label="SRC")
    plt.bar(x + width/2, tgt_vals, width, label="TGT")
    plt.xticks(x, metric_names)
    plt.ylabel("Valor (normalizado)")
    plt.title("Comparação de separabilidade entre SRC e TGT")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
