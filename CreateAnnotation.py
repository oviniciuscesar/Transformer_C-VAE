import os
import glob
import numpy as np
import librosa
import torch
import pandas as pd
from pathlib import Path

# Parâmetros principais
audio_dir = os.path.join(os.path.dirname(__file__), 'Flute')
dataset_dir = os.path.join(os.path.dirname(__file__), 'dataset')
np.random.seed(42)
torch.manual_seed(42)

def list_audio_files(audio_dir):
    """Lista todos os arquivos .wav nas subpastas de audio_dir."""
    return sorted(glob.glob(os.path.join(audio_dir, '*', '*.wav')))

def get_folder_labels(files):
    """
    Cria um mapeamento de pastas para labels numéricos.
    Retorna dicionário {nome_pasta: label_numerico}
    """
    folders = list(set([Path(f).parent.name for f in files]))
    folders.sort()  # Garante ordem consistente
    folder_to_label = {folder: idx for idx, folder in enumerate(folders)}
    return folder_to_label

def extract_audio_metadata(file_path):
    """
    Extrai metadados de um arquivo de áudio.
    Retorna: sample_rate, duration_samples, duration_ms, channels
    """
    try:
        # Carrega o áudio para obter informações completas
        audio_data, sr = librosa.load(file_path, sr=None, mono=False)  # mono=False para preservar canais
        
        # Verifica se é mono ou estéreo
        if audio_data.ndim == 1:
            # Áudio mono
            channels = 1
            duration_samples = len(audio_data)
        else:
            # Áudio multicanal
            channels = audio_data.shape[0]
            duration_samples = audio_data.shape[1]
        
        sample_rate = sr
        duration_ms = (duration_samples / sample_rate) * 1000
        
        return sample_rate, duration_samples, duration_ms, channels
    
    except Exception as e:
        print(f"Erro ao processar {file_path}: {e}")
        return None, None, None, None


def create_audio_metadata_csv(audio_dir, output_path=None):
    """
    Cria um CSV com metadados de todos os arquivos de áudio.
    
    Parâmetros:
    - audio_dir: Diretório contendo subpastas com arquivos .wav
    - output_path: Caminho para salvar o CSV (opcional)
    
    Retorna:
    - DataFrame com os metadados
    """
    
    print('Listando arquivos de áudio...')
    files = list_audio_files(audio_dir)
    print(f'Total de arquivos encontrados: {len(files)}')
    
    if len(files) == 0:
        print("Nenhum arquivo de áudio encontrado.")
        return None
    
    # Cria mapeamento de pastas para labels
    folder_to_label = get_folder_labels(files)
    print(f"\nMapeamento de pastas para labels:")
    for folder, label in folder_to_label.items():
        print(f"  {folder} -> {label}")
    
    # Lista para armazenar os dados
    metadata_list = []
    
    print(f"\nProcessando {len(files)} arquivos...")
    
    for i, file_path in enumerate(files):
        if i % 10 == 0:  # Progress indicator
            print(f"Processando arquivo {i+1}/{len(files)}")
        
        # Extrai informações do caminho
        path_obj = Path(file_path)
        filename = path_obj.name
        folder_name = path_obj.parent.name
        label = folder_to_label[folder_name]
        try:
            relative_path = str(path_obj.relative_to(audio_dir))
        except Exception:
            relative_path = os.path.relpath(file_path, start=audio_dir)

        # Extrai metadados do áudio (agora incluindo canais)
        sample_rate, duration_samples, duration_ms, channels = extract_audio_metadata(file_path)
        
        # Adiciona à lista
        metadata_list.append({
            'filename': filename,
            'folder': folder_name,
            'label': label,
            'sample_rate': sample_rate,
            'duration_samples': duration_samples,
            'duration_ms': duration_ms,
            'channels': channels,  # ← Nova coluna
            'path': relative_path
        })

    # Cria DataFrame
    df = pd.DataFrame(metadata_list)

    print("\nIniciando processo de sub-amostragem...")

    # Define os grupos com base na nossa última conversa
    # (baseado na sua distribuição de arquivos)
    GROUP_2_BASELINE = [
        'jet_whistle', 'harmonic_fingering', 'crescendo_to_decrescendo',
        'note_lasting', 'staccato', 'tongue_ram-pizz', 'crescendo',
        'decrescendo', 'flatterzunge_to_ordinario', 
        'ordinario_to_flatterzunge', 'multiphonics'
    ]

    # Calcula as contagens de arquivos por pasta
    class_counts = df['folder'].value_counts()
    
    # Encontra a contagem alvo (target_count)
    # A meta é o menor número de arquivos de uma classe do "Grupo 2"
    baseline_counts = class_counts[class_counts.index.isin(GROUP_2_BASELINE)]
    
    if baseline_counts.empty:
        print("Aviso: Nenhuma das classes da 'Linha de Base' foi encontrada.")
        print("Sub-amostragem não será aplicada.")
        target_count = -1 # Flag para pular amostragem
    else:
        target_count = baseline_counts.min()
        print(f"Contagem alvo para sub-amostragem: {target_count} (baseado em '{baseline_counts.idxmin()}' - a menor classe do Grupo 2)")

    balanced_df_list = []
    
    if target_count > 0:
        for folder, count in class_counts.items():
            folder_df = df[df['folder'] == folder]
            
            # Se a classe tem mais arquivos que o alvo, sub-amostra
            if count > target_count:
                print(f"Sub-amostrando '{folder}' de {count} para {target_count} arquivos...")
                sampled_df = folder_df.sample(n=target_count, random_state=42)
                balanced_df_list.append(sampled_df)
            else:
                # Se for menor ou igual, mantém todos os arquivos
                print(f"Mantendo '{folder}' com {count} arquivos.")
                balanced_df_list.append(folder_df)
        
        # Concatena os DataFrames balanceados
        df = pd.concat(balanced_df_list)
        print("Sub-amostragem concluída.")
    
    
    
    # APLICA SHUFFLE para randomizar a ordem dos arquivos
    print("\nAplicando shuffle aos dados...")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Define caminho de saída se não fornecido
    if output_path is None:
       output_path = os.path.join(dataset_dir, 'FluteMetadata.csv')

    # Salva CSV
    df.to_csv(output_path, index=False)
    print(f"\nCSV salvo em: {output_path}")
    
    return df

def analyze_dataset_statistics(df):
    """
    Analisa e exibe estatísticas do dataset (atualizada para incluir canais).
    """
    print("\n=== ESTATÍSTICAS DO DATASET (PÓS-AMOSTRAGEM) ===")
    
    # Estatísticas gerais
    print(f"Total de arquivos: {len(df)}")
    print(f"Total de pastas/classes: {df['label'].nunique()}")
    
    # Distribuição por classe
    print("\n--- Distribuição por Classe ---")
    class_counts = df.groupby(['folder', 'label']).size().reset_index(name='count')
    for _, row in class_counts.iterrows():
        print(f"  {row['folder']} (label {row['label']}): {row['count']} arquivos")
    
    # Estatísticas de sample rate
    print("\n--- Sample Rates ---")
    sr_counts = df['sample_rate'].value_counts().sort_index()
    for sr, count in sr_counts.items():
        if pd.notna(sr):
            print(f"  {int(sr)} Hz: {count} arquivos")
    
    # Estatísticas de canais ← NOVA SEÇÃO
    print("\n--- Distribuição de Canais ---")
    channel_counts = df['channels'].value_counts().sort_index()
    for channels, count in channel_counts.items():
        if pd.notna(channels):
            channel_type = "mono" if channels == 1 else f"{int(channels)} canais"
            print(f"  {channel_type}: {count} arquivos")
    
    # Estatísticas de duração
    print("\n--- Estatísticas de Duração ---")
    valid_durations = df[df['duration_ms'].notna()]
    if len(valid_durations) > 0:
        print(f"  Duração média: {valid_durations['duration_ms'].mean():.2f} ms")
        print(f"  Duração mínima: {valid_durations['duration_ms'].min():.2f} ms")
        print(f"  Duração máxima: {valid_durations['duration_ms'].max():.2f} ms")
        print(f"  Desvio padrão: {valid_durations['duration_ms'].std():.2f} ms")
    
    # Verifica arquivos com problemas
    problematic_files = df[df['sample_rate'].isna()]
    if len(problematic_files) > 0:
        print(f"\n Arquivos com problemas: {len(problematic_files)}")
        for _, row in problematic_files.iterrows():
            print(f"    {row['filename']} (pasta: {row['folder']})")
            
def main():
    """Função principal"""
    print("=== ANALISADOR DE METADADOS DE ÁUDIO ===")
    print(f"Diretório de áudio: {audio_dir}")
    
    # Verifica se o diretório existe
    if not os.path.exists(audio_dir):
        print(f" Diretório não encontrado: {audio_dir}")
        return
    
    # Cria CSV com metadados
    df = create_audio_metadata_csv(audio_dir)
    
    if df is not None:
        # Analisa estatísticas
        analyze_dataset_statistics(df)
        
        # Exibe primeiras linhas
        print("\n=== PRIMEIRAS 10 LINHAS DO CSV ===")
        print(df.head(10).to_string(index=False))
        
        print("\n Análise concluída com sucesso!")
    else:
        print(" Falha na criação do CSV.")

if __name__ == '__main__':
    main()