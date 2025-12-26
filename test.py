import os
import random
from pathlib import Path
import json

import torch
import numpy as np
import torchaudio

# Ranges de normalização do dataset
from TCVAEdataset import NORMALIZATION_RANGES

# Biblioteca opcional (Contorchionist). Se não houver, cai para torchaudio.
try:
    import pycontorchionist as cc
    HAS_CC = True
except ImportError:
    print("Aviso: pycontorchionist não encontrado, usando torchaudio para MelSpectrogram.")
    HAS_CC = False

# ====== Paths e seeds ======
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

DIRECTORY = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(DIRECTORY, "TorchScript")
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "Checkpoints")
FLUTE_DIR = os.path.join(DIRECTORY, "Flute")

# Se quiser testar um único arquivo, defina AUDIO_PATH; caso contrário, o script roda batch por classe
# AUDIO_PATH = os.path.join(DIRECTORY, 'Flute/jet_whistle/Fl-jet_wh-N-N-N-N.wav')
# AUDIO_PATH = os.path.join(DIRECTORY, 'Flute/ordinario/Fl-ord-A#q6-ff-N-N.wav')
AUDIO_PATH = os.path.join(DIRECTORY, 'Flute/crescendo_to_decrescendo/Fl-cre_dec-B3-ppmfpp-N-N.wav')
RUN_BATCH_PER_CLASS = False  # defina True para rodar 1 exemplo aleatório por classe
LATENT_TEST = True
CLASS = "aeolian"  # se None, usa AUDIO_PATH; senão, busca um arquivo aleatório dessa classe
LATENT_CLASS = "crescendo"  # se None, usa a classe do áudio; senão, define a classe do vetor latente

# ====== Parâmetros do modelo/dados ======
N_FRAMES = 10       # frames do melspectrograma
N_MELS = 80         # bins Mel
N_FFT = 4096
HOP_LENGTH = 512
SAMPLE_RATE = 44100
FMIN = 0.0
FMAX = SAMPLE_RATE // 2
MEL_NORM = "slaney"  # usado no Contorchionist (filterbank_norm)
MEL_FORMULA = 'HTK'
WINDOW_TYPE = 'HANN'
MEL_MODE = cc.MelNormMode.ENERGY_POWER if HAS_CC else None  # só CC usa

# Targets
STEPS = 1           # passos de saída do modelo
N_PITCHES = 7
N_AMPS = 7
N_TEXTURE_PARAMS = 6  # 4 metros + 1 grão + 1 âmbito
TARGET_FEATURES = N_PITCHES + N_AMPS + N_TEXTURE_PARAMS  # 20

# ====== Desnormalização ======
def denormalize_minus1_1(x_norm: torch.Tensor, xmin: float, xmax: float) -> torch.Tensor:
    # mapeia [-1,1] -> [0,1]
    x01 = torch.clamp((x_norm + 1.0) * 0.5, 0.0, 1.0)
    xmin_t = torch.as_tensor(xmin, dtype=x_norm.dtype, device=x_norm.device)
    xmax_t = torch.as_tensor(xmax, dtype=x_norm.dtype, device=x_norm.device)
    eps = torch.as_tensor(1e-8, dtype=x_norm.dtype, device=x_norm.device)
    # reconstrói no domínio do log e aplica exp
    min_log = torch.log(xmin_t + eps)
    max_log = torch.log(xmax_t)
    x_log = x01 * (max_log - min_log) + min_log
    x = torch.exp(x_log)
    return torch.clamp(x, xmin_t, xmax_t)

def denormalize_pitches_zscore(pitches_norm: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    mean_t = torch.as_tensor(mean, dtype=pitches_norm.dtype, device=pitches_norm.device)
    std_t = torch.as_tensor(std, dtype=pitches_norm.dtype, device=pitches_norm.device).clamp(min=1e-8)
    return pitches_norm * std_t + mean_t

def denormalize_output(flat_out: torch.Tensor) -> torch.Tensor:
    """
    Reverte a normalização do vetor de saída achatado do modelo.
    Aceita shape (20) ou (steps*20). Retorna tensor (steps, 20).
    """
    x = flat_out.view(-1, TARGET_FEATURES)  # (steps, 20)

    # Pitches (Z-score -> cents) – média no meio do range configurado; std típico 1200.0 cents
    pitch_mean = NORMALIZATION_RANGES['pitch']['min'] + (NORMALIZATION_RANGES['pitch']['max'] - NORMALIZATION_RANGES['pitch']['min']) / 2.0
    x[:, 0:7] = denormalize_pitches_zscore(x[:, 0:7], mean=pitch_mean, std=1200.0)

    # Amplitudes (log [-1,1] -> [min,max])
    x[:, 7:14] = denormalize_minus1_1(
        x[:, 7:14],
        NORMALIZATION_RANGES['amp']['min'],
        NORMALIZATION_RANGES['amp']['max'],
    )

    # Metrônomos m1..m4 (log)
    x[:, 14:18] = denormalize_minus1_1(
        x[:, 14:18],
        NORMALIZATION_RANGES['metro']['min'],
        NORMALIZATION_RANGES['metro']['max'],
    )

    # Grain (log)
    x[:, 18:19] = denormalize_minus1_1(
        x[:, 18:19],
        NORMALIZATION_RANGES['grain']['min'],
        NORMALIZATION_RANGES['grain']['max'],
    )

    # Âmbito (log neste dataset)
    x[:, 19:20] = denormalize_minus1_1(
        x[:, 19:20],
        NORMALIZATION_RANGES['ambito']['min'],
        NORMALIZATION_RANGES['ambito']['max'],
    )
    return x

# ====== Utilidades de áudio/mel ======
def load_audio(path: str, target_sr: int) -> np.ndarray:
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform.squeeze().numpy()

def build_cc_processor():
    if not HAS_CC:
        return None
    p = cc.MelSpectrogramProcessor()
    p.set_sample_rate(float(SAMPLE_RATE))
    p.set_n_fft(N_FFT)
    p.set_hop_length(HOP_LENGTH)
    p.set_n_mels(N_MELS)
    p.set_fmin_mel(float(FMIN))
    p.set_fmax_mel(float(FMAX))
    p.set_window_type(getattr(cc.WindowType, WINDOW_TYPE))
    p.set_normalization_type(getattr(cc.NormalizationType, 'POWER'))  # power spectrum
    p.set_filterbank_norm(MEL_NORM)
    p.set_mel_norm_mode(MEL_MODE)
    p.set_mel_formula(getattr(cc.MelFormulaType, MEL_FORMULA))
    p.set_output_format(cc.SpectrumDataFormat.MAGPHASE)
    return p

_CC_PROCESSOR = build_cc_processor()

def compute_mel(audio: np.ndarray, n_frames: int) -> torch.Tensor:
    """
    Retorna tensor [1, n_frames*N_MELS] flatten.
    Usa Contorchionist se disponível; caso contrário, torchaudio.
    """
    if HAS_CC and _CC_PROCESSOR is not None:
        frames = []
        block_size = HOP_LENGTH
        for i in range(0, len(audio), block_size):
            chunk = audio[i:i+block_size].astype(np.float32, copy=False)
            mel = _CC_PROCESSOR.process(chunk)
            if mel is not None:
                frames.append(np.array(mel))
            if len(frames) >= n_frames:
                break
        if not frames:
            mel_stack = np.zeros((n_frames, N_MELS), dtype=np.float32)
        else:
            mel_stack = np.stack(frames[:n_frames], axis=0).astype(np.float32)
        mel_tensor = torch.from_numpy(mel_stack).flatten().unsqueeze(0)  # [1, F*M]
        return mel_tensor
    else:
        # torchaudio MelSpectrogram full, depois recorta para n_frames
        mel_tf = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            f_min=FMIN,
            f_max=FMAX,
            norm=MEL_NORM,
            power=2.0,
        )
        wav_t = torch.from_numpy(audio).float()  # [T]
        mel_spec = mel_tf(wav_t)  # [MELS, frames]
        mel_spec = mel_spec[:, :n_frames] if mel_spec.size(1) >= n_frames else torch.nn.functional.pad(
            mel_spec, (0, n_frames - mel_spec.size(1))
        )
        mel_tensor = mel_spec.transpose(0, 1).contiguous().view(1, -1)  # [1, F*M]
        return mel_tensor

# ====== Seleção de exemplos ======
def pick_one_wav_per_class(root_dir: str) -> dict:
    """
    Percorre subpastas de root_dir e escolhe 1 .wav aleatório por classe.
    Retorna dict {classe: caminho_wav}.
    """
    result = {}
    if not os.path.isdir(root_dir):
        return result
    for entry in os.scandir(root_dir):
        if not entry.is_dir():
            continue
        wavs = [f.path for f in os.scandir(entry.path) if f.is_file() and f.name.lower().endswith(".wav")]
        if not wavs:
            continue
        result[entry.name] = random.choice(wavs)
    return result

# ====== Execução de inferência ======
def load_scripted_model():
    ts_path = os.path.join(MODEL_DIR, "tc-vaeV2.ts")
    if not os.path.isfile(ts_path):
        raise FileNotFoundError(f"Modelo TorchScript não encontrado: {ts_path}")
    model = torch.jit.load(ts_path, map_location='cpu')
    model.eval()
    # define steps de saída = 1
    if hasattr(model, "steps"):
        model.steps(3)
    return model

def run_inference_on_path(model, path: str) -> torch.Tensor:
    audio = load_audio(path, SAMPLE_RATE)
    mel_tensor = compute_mel(audio, N_FRAMES)  # [1, N_FRAMES*N_MELS]
    with torch.no_grad():
        out = model.forward(mel_tensor)  # wrapper usa forward() aleatório com z~N(0,I)
    denorm = denormalize_output(out.squeeze(0).cpu())
    return denorm  # [steps, 20]

def load_centroids_dict(filepath: str, device='cpu') -> dict:
    """
    Lê o arquivo de texto e retorna um dicionário { 'nome_da_classe': Tensor(1, 64) }
    """
    centroids = {}
    print(f"Carregando centróides de: {filepath}")
    
    if not os.path.exists(filepath):
        print("AVISO: Arquivo de centróides não encontrado.")
        return {}

    with open(filepath, 'r') as f:
        for line in f:
            # Limpa espaços e o ponto-e-vírgula final
            line = line.strip().replace(';', '')
            if not line: continue
            
            parts = line.split(' ')
            # Formato esperado: ID CLASSE V1 V2 ... V64
            if len(parts) > 2:
                class_name = parts[1]
                # Pega do índice 2 até o fim e converte para float
                vector_values = [float(x) for x in parts[2:]]
                # Cria tensor (1, Latent_Dim)
                z_tensor = torch.tensor(vector_values, dtype=torch.float32, device=device).unsqueeze(0)
                centroids[class_name] = z_tensor
                
    print(f"Centróides carregados para {len(centroids)} classes.")
    return centroids


def run_inference_on_path_z(model, path: str, centroids: dict, target_class: str = None) -> torch.Tensor:
    """Executa inferência usando um vetor latente específico da classe.
    
    Args:
        model: modelo TorchScript carregado
        path: caminho do arquivo de áudio
        centroids: dicionário {classe: tensor latente} carregado do arquivo
        target_class: nome da classe para usar o vetor latente. Se None, usa a classe do arquivo.
    """
    audio = load_audio(path, SAMPLE_RATE)
    mel_tensor = compute_mel(audio, N_FRAMES)  # [1, N_FRAMES*N_MELS]
    
    # Determina qual classe usar para o vetor latente
    if target_class is not None:
        class_name = target_class
        print(f"Usando vetor latente da classe especificada: {class_name}")
    else:
        # Extrai o nome da classe do caminho (pasta pai do arquivo)
        class_name = Path(path).parent.name
        print(f"Usando vetor latente da classe do arquivo: {class_name}")
    
    # Busca o vetor latente correspondente à classe
    if class_name in centroids:
        z_vector = centroids[class_name]
    else:
        print(f"AVISO: Classe '{class_name}' não encontrada nos centróides. Usando vetor aleatório.")
        z_vector = torch.randn((1, 64), dtype=torch.float32)
    
    with torch.no_grad():
        model.latent(z_vector)  # define o vetor latente
        out = model.forwardz(mel_tensor)  # forward com vetor latente específico
    denorm = denormalize_output(out.squeeze(0).cpu())
    return denorm  # [steps, 20]


def print_denorm_result(denorm: torch.Tensor, cls: str, fname: str, step: int = 0):
    print(f"\n=== Classe: {cls} | Arquivo: {fname} ===")
    print(f"---------- step: {step} -----------")
    print(f"  Pitches (cents): {np.round(denorm[step, 0:7].numpy(), 1)}")
    print(f"  Amps (0-127):   {np.round(denorm[step, 7:14].numpy(), 1)}")
    print(f"  Metros (ms):    {np.round(denorm[step, 14:18].numpy(), 1)}")
    print(f"  Grain (ms):     {np.round(denorm[step, 18:19].numpy(), 1)}")
    print(f"  Âmbito:         {np.round(denorm[step, 19:20].numpy(), 1)}")

def batch_test_random_examples():
    samples = pick_one_wav_per_class(FLUTE_DIR)
    if not samples:
        print(f"Nenhuma classe .wav encontrada em {FLUTE_DIR}")
        return
    model = load_scripted_model()
    print(f"Executando teste aleatório por classe ({len(samples)} classes)...")
    for cls, wav_path in sorted(samples.items()):
        try:
            denorm = run_inference_on_path(model, wav_path)
            print_denorm_result(denorm, cls=cls, fname=os.path.basename(wav_path), step=0)
        except Exception as e:
            print(f"Falha na classe {cls} ({wav_path}): {e}")

def single_test():
    if not os.path.isfile(AUDIO_PATH):
        print(f"Arquivo de áudio não encontrado: {AUDIO_PATH}")
        return
    model = load_scripted_model()
    denorm = run_inference_on_path(model, AUDIO_PATH)
    cls = Path(AUDIO_PATH).parent.name
    print_denorm_result(denorm, cls=cls, fname=os.path.basename(AUDIO_PATH), step=0)


def latent_test(target_class: str = None, latent_class: str = None):
    """
    Teste extra: gera uma saída a partir de um vetor latente específico da classe.
    Carrega os centróides do arquivo de texto e usa o vetor correspondente à classe do áudio.
    
    Args:
        target_class: nome da classe para buscar o arquivo de áudio. Se None, usa AUDIO_PATH. 
                     Se fornecido, busca um arquivo aleatório dessa classe.
        latent_class: nome da classe para usar o vetor latente. Se None, usa a classe do arquivo de áudio.
    """
    # Carrega os centróides do arquivo
    centroids_path = os.path.join(CHECKPOINT_DIR, "latent_centroids.txt")
    centroids = load_centroids_dict(centroids_path)
    
    if not centroids:
        print("AVISO: Nenhum centróide carregado. Verifique o arquivo de centróides.")
        return
    
    # Determina qual arquivo usar
    if target_class is not None:
        # Busca um arquivo aleatório da classe especificada
        class_dir = os.path.join(FLUTE_DIR, target_class)
        if not os.path.isdir(class_dir):
            print(f"ERRO: Diretório da classe '{target_class}' não encontrado: {class_dir}")
            return
        
        wavs = [f.path for f in os.scandir(class_dir) if f.is_file() and f.name.lower().endswith(".wav")]
        if not wavs:
            print(f"ERRO: Nenhum arquivo .wav encontrado na classe '{target_class}'")
            return
        
        audio_path = random.choice(wavs)
        print(f"Classe do áudio selecionada: {target_class}")
        print(f"Arquivo escolhido: {os.path.basename(audio_path)}")
    else:
        # Usa o AUDIO_PATH definido no início do arquivo
        if not os.path.isfile(AUDIO_PATH):
            print(f"Arquivo de áudio não encontrado: {AUDIO_PATH}")
            return
        audio_path = AUDIO_PATH
    
    model = load_scripted_model()
    denorm = run_inference_on_path_z(model, audio_path, centroids, target_class=latent_class)
    cls = Path(audio_path).parent.name
    print_denorm_result(denorm, cls=cls, fname=os.path.basename(audio_path), step=0)



# ====== Main ======
if __name__ == "__main__":
    if RUN_BATCH_PER_CLASS:
        batch_test_random_examples()
    else:
        single_test()
    if LATENT_TEST:
        latent_test(target_class=CLASS, latent_class=LATENT_CLASS) 
        print("\nLatent test desativado.")