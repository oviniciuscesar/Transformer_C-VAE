import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import torchaudio.transforms as T
import numpy as np

try:
    import pycontorchionist as cc
except ImportError:
    print("Erro: Não foi possível importar o módulo pycontorchionist.")
    print("Verifique se o arquivo do módulo (pycontorchionist...so ou .pyd) está no mesmo diretório que este script.")
    exit(1)



# --- Parâmetros de Diretório ---
DIRECTORY = os.path.dirname(__file__)
AUDIO_DIR_ROOT = os.path.join(DIRECTORY, 'Flute')
DATASET_DIR = os.path.join(DIRECTORY, 'dataset')
CSV_PATH = os.path.join(DATASET_DIR, 'FluteMetadata.csv')



# CONSTANTES GLOBAIS PARA IMPORTAÇÃO
MELSPEC_PARAMS = {
    'sample_rate': 44100,
    'num_samples': 88200,
    'n_mels': 64,
    'n_fft': 4096,
    'hop_length': 512,
    'center': False,
    'fmin': 0.0,
    'fmax': 44100//2,
    'norm': 'slaney',
    'window_type': 'HANN',
    'normalization_type': 'POWER',
    'spectrum_data_format': 'POWERPHASE',
    'mel_formula': 'HTK',
    'mel_norm_mode': 'ENERGY_POWER',
    'device': 'mps' if torch.backends.mps.is_available() else 'cpu',
    'verbose': False,
    'MELSPEC_SHAPE': None,
    'N_FRAMES': None
}


# Parâmetros do Modelo (Nossos alvos)
N_FRAMES = 10       # 10 frames de melspectrograma
TARGET_FEATURES = 5 # (cents, amp, grain, metro, dur)
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'


class ContorchionistMelTransform:
    def __init__(
        self,
        sample_rate=MELSPEC_PARAMS['sample_rate'],
        n_fft=MELSPEC_PARAMS['n_fft'],
        hop_length=MELSPEC_PARAMS['hop_length'],
        win_length=None,
        n_mels=MELSPEC_PARAMS['n_mels'],
        fmin=0.0,
        fmax=None,
        norm="slaney", #normalização do banco de filtros
        window_type="HANN", #tipo de janela
        normalization_type="POWER", #normalização da RFFT
        spectrum_data_format="POWERPHASE", #formato de saída do espectro
        mel_formula="HTK", #fórmula de conversão para mel
        mel_norm_mode="ENERGY_POWER", #normalização do banco de filtros mel
        device=None,
        verbose=False
    ):
        self.processor = cc.MelSpectrogramProcessor()

        self.processor.set_sample_rate(int(sample_rate))
        self.processor.set_n_fft(int(n_fft))
        self.processor.set_hop_length(int(hop_length))
        self.processor.set_win_length(int(win_length) if win_length else int(n_fft))
        self.processor.set_n_mels(int(n_mels))
        self.processor.set_fmin_mel(float(fmin))
        self.processor.set_fmax_mel(float(fmax) if fmax else sample_rate // 2)
        self.processor.set_filterbank_norm(norm)
        self.processor.set_window_type(getattr(cc.WindowType, window_type))
        self.processor.set_normalization_type(getattr(cc.NormalizationType, normalization_type))
        self.processor.set_output_format(getattr(cc.SpectrumDataFormat, spectrum_data_format))
        self.processor.set_mel_formula(getattr(cc.MelFormulaType, mel_formula))
        self.processor.set_mel_norm_mode(getattr(cc.MelNormMode, mel_norm_mode))
        self.processor.set_device(torch.device(device) if device else torch.device("mps" if torch.backends.mps.is_available() else "cpu"))
        self.processor.set_verbose(bool(verbose))


    def __call__(self, signal):
        # signal: torch.Tensor [1, num_samples]
        audio = signal.squeeze().cpu().numpy().astype(np.float32)
        frames = []
        block_size = self.processor.get_hop_length()
        for i in range(0, len(audio), block_size):
            chunk = audio[i:i+block_size]
            mel = self.processor.process(chunk)
            if mel is not None:
                frames.append(np.array(mel))
        if frames:
            mel_spec = np.stack(frames, axis=1)
        else:
            mel_spec = np.zeros((self.processor.get_n_mels(), 1))
        mel_spec = torch.tensor(mel_spec, dtype=torch.float32, device=signal.device) # move to the same device as input signal
        if mel_spec.ndim == 2:
            mel_spec = mel_spec.unsqueeze(0)
        return mel_spec # return the channel dimension as well [1, n_mels, n_frames]
    

def calculate_mel_frames(num_samples, n_fft=MELSPEC_PARAMS['n_fft'], hop_length=MELSPEC_PARAMS['hop_length'],  center=False):
    """
    Calcula o número de frames do mel-spectrogram
    Args:
        num_samples: Número de samples de áudio
        n_fft: Tamanho da janela FFT
        hop_length: Salto entre janelas
        center: Se True, adiciona padding
    Returns:
        int: Número de frames temporais
    """
    if center:
        padded_length = num_samples + n_fft
        n_frames = (padded_length - n_fft) // hop_length + 1
    else:
        n_frames = (num_samples - n_fft) // hop_length + 1
    
    return n_frames

def get_melspec_shape(sample_rate=44100, num_samples=88200, n_mels=MELSPEC_PARAMS['n_mels'], 
                     n_fft=MELSPEC_PARAMS['n_fft'], hop_length=MELSPEC_PARAMS['hop_length'],  center=False):
    """
    Retorna o shape completo do mel-spectrogram
    Returns:
        tuple: (n_mels, n_frames)
    """
    n_frames = calculate_mel_frames(num_samples, n_fft, hop_length, center=False)
    return (n_mels, n_frames)


# Cache global para resamplers
RESAMPLER_CACHE = {}

def get_resampler(src_sr, tgt_sr):
    """Cria ou obtém um T.Resample do cache."""
    if (src_sr, tgt_sr) not in RESAMPLER_CACHE:
        RESAMPLER_CACHE[(src_sr, tgt_sr)] = T.Resample(
            orig_freq=src_sr, 
            new_freq=tgt_sr
        ).to(DEVICE)
    return RESAMPLER_CACHE[(src_sr, tgt_sr)]


def get_target_for_label(label: int, n_frames: int, n_features: int) -> torch.Tensor:
    """
    Esta é a função mais importante para sua composição.
    Ela define qual sequência (tgt) o modelo deve aprender para cada classe (src).

    Args:
        label (int): O rótulo da classe (ex: 0 para 'aeolian', 4 para 'flatterzunge').
        n_frames (int): O número de frames (10).
        n_features (int): O número de features de saída (5).

    Returns:
        torch.Tensor: Um tensor de shape (n_frames, n_features), ex: (10, 5)
    """
    
    # Exemplo de lógica que você pode implementar:
    # 
    if label == 4: # 'flatterzunge'
        # Retorna uma sequência (10, 5) que representa "grãos rápidos"
        # (cents, amp, grain_size, metro, dur)
        # (Apenas um exemplo, use seus próprios valores)
        cents = torch.full((n_frames, 1), 7200.0) # pitch
        amps = torch.full((n_frames, 1), 100.0)   # amplitude
        grains = torch.full((n_frames, 1), 50.0)    # grain size (ms)
        metros = torch.full((n_frames, 1), 50.0)    # metro (ms)
        durs = torch.full((n_frames, 1), 50.0)      # duration (ms)
        return torch.cat([cents, amps, grains, metros, durs], dim=1)
    
    elif label == 1: # 'crescendo'
        # Retorna uma sequência com amplitude crescente
        amps = torch.linspace(60.0, 120.0, n_frames).unsqueeze(1)
        # ... preencha os outros 4 parâmetros ...
        # return torch.cat([..., amps, ...], dim=1)
    
    else:
        # Resposta padrão para outras classes
        return torch.zeros(n_frames, n_features)

    # AVISO: Treinar com zeros fará com que o modelo aprenda a gerar silêncio.
    if not hasattr(get_target_for_label, "warned"):
        print("\n*** AVISO IMPORTANTE ***")
        print("A função 'get_target_for_label' está usando um placeholder (zeros).")
        print("Você DEVE editá-la em TCVAEdataset.py com sua lógica composicional.")
        print("*************************\n")
        get_target_for_label.warned = True # Evita spam de avisos

    return torch.zeros(n_frames, n_features)


### --- Lógica Principal de Criação do Dataset ---
def create_dataset(csv_path, audio_root, save_dir, hop_step=1):
    """
    Lê o CSV, processa arquivos de áudio, extrai janelas de melspectrograma (src)
    e gera os alvos (tgt) correspondentes. Salva o dataset em arquivos .pt.
    
    Args:
        hop_step (int): O "pulo" da janela deslizante em frames. 
                        1 = máximo de dados (altamente sobreposto).
    """
    
    print(f"Usando dispositivo: {DEVICE}")
    
    # 1. Carregar o CSV balanceado
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Erro: Arquivo CSV não encontrado em {csv_path}")
        return
        
    print(f"CSV balanceado carregado com {len(df)} arquivos.")

    # 2. Inicializar o transformador pycontorchionist
    print("Inicializando ContorchionistMelTransform...")
    transform = ContorchionistMelTransform(
        device=DEVICE,
        verbose=MELSPEC_PARAMS['verbose']
        # (Usa os outros defaults de MELSPEC_PARAMS)
    )
    
    # 3. Inicializar conversor para dB
    # pycontorchionist retorna POWER, convertemos para dB
    to_db = T.AmplitudeToDB(stype='power').to(DEVICE)

    # 4. Listas para armazenar todas as amostras
    all_src_samples = []
    all_tgt_samples = []
    all_labels = []

    print(f"Iniciando processamento de {len(df)} arquivos...")
    
    # 5. Iterar sobre o CSV
    for idx, row in df.iterrows():
        file_path = os.path.join(audio_root, row['path'])
        label = int(row['label'])
        
        if idx % 20 == 0:
            print(f"  Processando {idx+1}/{len(df)}: {row['filename']}")

        try:
            # Carregar áudio
            waveform, sr = torchaudio.load(file_path)
            waveform = waveform.to(DEVICE)

            # Resample se necessário
            target_sr = MELSPEC_PARAMS['sample_rate']
            if sr != target_sr:
                resampler = get_resampler(sr, target_sr)
                waveform = resampler(waveform)

            # Converter para mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # 6. Calcular Melspectrograma completo com pycontorchionist
            melspec = transform(waveform) # (B, n_mels, L)
            melspec_db = to_db(melspec)    # (B, n_mels, L)
            melspec_db = melspec_db.squeeze(0) # (n_mels, L) -> (64, L)
            
            total_frames = melspec_db.shape[1]
            
            # Pular arquivos muito curtos
            if total_frames < N_FRAMES:
                print(f"    Aviso: Pulando {row['filename']}, muito curto ({total_frames} frames)")
                continue

            # 7. Obter a sequência alvo (TGT) para esta classe
            tgt_sequence = get_target_for_label(label, N_FRAMES, TARGET_FEATURES).to(DEVICE)

            # 8. Aplicar Janela Deslizante (Sliding Window)
            for i in range(0, total_frames - N_FRAMES + 1, hop_step):
                # Extrair a janela SRC
                src_window = melspec_db[:, i : i + N_FRAMES] # (n_mels, n_frames) -> (64, 10)
                
                # Transpor para (n_frames, n_mels) -> (10, 64)
                # Esta é a forma que o modelo espera (SEQ_LEN, FEATURES)
                src_window_transposed = src_window.T
                
                # Adicionar às listas
                all_src_samples.append(src_window_transposed)
                all_tgt_samples.append(tgt_sequence)
                all_labels.append(label)

        except Exception as e:
            print(f"    ERRO ao processar {file_path}: {e}")

    print("\nProcessamento de arquivos concluído.")
    
    if not all_src_samples:
        print("Erro: Nenhuma amostra foi extraída. Verifique seus arquivos de áudio.")
        return

    # 9. Empilhar tudo em tensores grandes
    print("Empilhando tensores...")
    try:
        final_src = torch.stack(all_src_samples).cpu() # (N_samples, 10, 64)
        final_tgt = torch.stack(all_tgt_samples).cpu() # (N_samples, 10, 5)
        final_labels = torch.tensor(all_labels, dtype=torch.long).cpu() # (N_samples,)
    except Exception as e:
        print(f"Erro ao empilhar tensores: {e}")
        print("Verifique se todas as amostras têm shapes consistentes.")
        return

    print("\n--- Estatísticas Finais do Dataset ---")
    print(f"Total de amostras (janelas de 10 frames): {len(final_src)}")
    print(f"Shape do tensor SRC: {final_src.shape}")
    print(f"Shape do tensor TGT: {final_tgt.shape}")
    print(f"Shape do tensor LABELS: {final_labels.shape}")

    # 10. Salvar os tensores em disco
    os.makedirs(save_dir, exist_ok=True)
    src_path = os.path.join(save_dir, 'train_src.pt')
    tgt_path = os.path.join(save_dir, 'train_tgt.pt')
    label_path = os.path.join(save_dir, 'train_labels.pt')

    print(f"\nSalvando dataset em {save_dir}...")
    try:
        torch.save(final_src, src_path)
        torch.save(final_tgt, tgt_path)
        torch.save(final_labels, label_path)
        print("Dataset salvo com sucesso!")
        print(f"  SRC: {src_path}")
        print(f"  TGT: {tgt_path}")
        print(f"  LABELS: {label_path}")
    except Exception as e:
        print(f"Erro ao salvar arquivos .pt: {e}")

### --- Ponto de Entrada Principal ---

def main():
    # Nota: hop_step=1 gerará MUITOS dados (altamente sobrepostos).
    # Comece com um hop_step maior (ex: 10 ou 5) para um teste rápido.
    # Um hop_step menor que N_FRAMES (10) cria dados sobrepostos, o que é bom.
    create_dataset(CSV_PATH, AUDIO_DIR_ROOT, DATASET_DIR, hop_step=5)

if __name__ == '__main__':
    main()