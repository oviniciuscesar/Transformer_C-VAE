import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import torchaudio.transforms as T
import numpy as np
from pathlib import Path
import librosa
import json 


# Importa sua biblioteca
try:
    import pycontorchionist as cc
except ImportError:
    print("Erro: Não foi possível importar o módulo pycontorchionist.")
    exit(1)

# --- Parâmetros de Diretório ---
DIRECTORY = os.path.dirname(__file__)
AUDIO_DIR_ROOT = os.path.join(DIRECTORY, 'Flute')
DATASET_DIR = os.path.join(DIRECTORY, 'dataset')
CSV_PATH = os.path.join(DATASET_DIR, 'FluteMetadata.csv')

# --- Constantes Globais ---
MELSPEC_PARAMS = {
    'sample_rate': 44100,
    'n_mels': 80,
    'n_fft': 4096,
    'hop_length': 512,
    'fmin': 0.0,
    'fmax': 44100 // 2,
    'norm': 'slaney',
    'window_type': 'HANN',
    'normalization_type': 'BACKWARD',
    'spectrum_data_format': 'MAGPHASE',
    'mel_formula': 'HTK',
    'mel_norm_mode': 'ENERGY_POWER',
    'verbose': False
}

# Parâmetros do Modelo (targets)
N_FRAMES = 10       # 10 frames de melspectrograma
N_PITCHES = 7       # 7 notas
N_AMPS = 7          # 7 amplitudes
N_TEXTURE_PARAMS = 6 # 4 (metros) + 1 (grão) + 1 (âmbito) = 6
TARGET_FEATURES = N_PITCHES + N_AMPS + N_TEXTURE_PARAMS  # target completo: 7 + 7 + 6 = 20

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

# PRÉ-CÁLCULO DOS BINS MEL EM HZ #
MEL_FREQS_HZ = librosa.mel_frequencies(
    n_mels=MELSPEC_PARAMS['n_mels'],
    fmin=MELSPEC_PARAMS['fmin'],
    fmax=MELSPEC_PARAMS['fmax'],
    htk=(MELSPEC_PARAMS['mel_formula'] == 'HTK')
)

# definição dos ranges globais de normalização
# !!! IMPORTANTE: ajustar com base nas heurísticas dos targets !!!
NORMALIZATION_RANGES = {
    # Features 0-6: Pitches (MIDI Cents)
    'pitch': {'min': 6000.0, 'max': 9600.0}, # três oitavas
    # Features 7-13: Amplitudes (MIDI Velocity)
    'amp': {'min': 0.0, 'max': 127.0},
    # Features 14-17: Metrônomos (ms)
    'metro': {'min': 100.0, 'max': 4083.56}, # 
    # Feature 18: Grain Size (ms)
    'grain': {'min': 50.0, 'max': 243}, # 
    # Feature 19: Âmbito
    'ambito': {'min': 10, 'max': 96.18}
}

# Cria tensores com os valores min/max para normalização
_min_vals = []
_max_vals = []
_min_vals.extend([NORMALIZATION_RANGES['pitch']['min']] * N_PITCHES)
_max_vals.extend([NORMALIZATION_RANGES['pitch']['max']] * N_PITCHES)
_min_vals.extend([NORMALIZATION_RANGES['amp']['min']] * N_AMPS)
_max_vals.extend([NORMALIZATION_RANGES['amp']['max']] * N_AMPS)
_min_vals.extend([NORMALIZATION_RANGES['metro']['min']] * 4) # 4 metros
_max_vals.extend([NORMALIZATION_RANGES['metro']['max']] * 4)
_min_vals.append(NORMALIZATION_RANGES['grain']['min']) # 1 grain
_max_vals.append(NORMALIZATION_RANGES['grain']['max'])
_min_vals.append(NORMALIZATION_RANGES['ambito']['min']) # 1 ambito
_max_vals.append(NORMALIZATION_RANGES['ambito']['max'])

TGT_MIN_VALS = torch.tensor(_min_vals, dtype=torch.float32, device=DEVICE)
TGT_MAX_VALS = torch.tensor(_max_vals, dtype=torch.float32, device=DEVICE)
TGT_RANGES = TGT_MAX_VALS - TGT_MIN_VALS
# Adiciona epsilon para evitar divisão por zero onde min == max
TGT_RANGES[TGT_RANGES == 0] = 1.0

# --- Classe ContorchionistMelTransform  ---
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
        normalization_type="BACKWARD", #normalização da RFFT
        spectrum_data_format="MAGPHASE", #formato de saída do espectro
        mel_formula="HTK", #fórmula de conversão para mel
        mel_norm_mode="ENERGY_POWER", #normalização do banco de filtros mel
        device=None,
        verbose=False
    ):
        self.processor = cc.MelSpectrogramProcessor() #
        self.processor.set_sample_rate(int(sample_rate)) #
        self.processor.set_n_fft(int(n_fft)) #
        self.processor.set_hop_length(int(hop_length)) #
        self.processor.set_win_length(int(win_length) if win_length else int(n_fft)) #
        self.processor.set_n_mels(int(n_mels)) #
        self.processor.set_fmin_mel(float(fmin)) #
        self.processor.set_fmax_mel(float(fmax) if fmax else sample_rate // 2) #
        self.processor.set_filterbank_norm(norm) #
        self.processor.set_window_type(getattr(cc.WindowType, window_type)) #
        self.processor.set_normalization_type(getattr(cc.NormalizationType, normalization_type)) #
        self.processor.set_output_format(getattr(cc.SpectrumDataFormat, spectrum_data_format)) #
        self.processor.set_mel_formula(getattr(cc.MelFormulaType, mel_formula)) #
        self.processor.set_mel_norm_mode(getattr(cc.MelNormMode, mel_norm_mode)) #
        self.device = torch.device(device) if device else torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.processor.set_device(self.device) #
        self.processor.set_verbose(bool(verbose)) #

    def __call__(self, signal):

        
        audio = signal.squeeze().cpu().numpy().astype(np.float32) #
        frames = [] #
        block_size = self.processor.get_hop_length() #
        for i in range(0, len(audio), block_size):
            chunk = audio[i:i+block_size] #
            mel = self.processor.process(chunk) #
            if mel is not None: #
                frames.append(np.array(mel)) #
        if frames: #
            mel_spec = np.stack(frames, axis=1) #
        else:
            mel_spec = np.zeros((self.processor.get_n_mels(), 1)) #
        mel_spec = torch.tensor(mel_spec, dtype=torch.float32, device=self.device) 
        if mel_spec.ndim == 2: #
            mel_spec = mel_spec.unsqueeze(0) #
        # print("primeiros 5 valores do mel_spec:", mel_spec.flatten()[:5].cpu().numpy()) #
        return mel_spec # [1, n_mels, n_frames] #


# --- Funções Auxiliares  ---
RESAMPLER_CACHE = {}
def get_resampler(src_sr, tgt_sr):
    if (src_sr, tgt_sr) not in RESAMPLER_CACHE:
        RESAMPLER_CACHE[(src_sr, tgt_sr)] = T.Resample(
            orig_freq=src_sr, new_freq=tgt_sr
        ).to(DEVICE)
    return RESAMPLER_CACHE[(src_sr, tgt_sr)]

MIN_DB_NORM = -80.0
MAX_DB_NORM = 0.0

# Converte potência para velocity MIDI (0-127)
def power_to_midi_velocity(power_values: torch.Tensor) -> torch.Tensor:
    db_values = 10.0 * torch.log10(power_values + 1e-10)
    normalized = (db_values - MIN_DB_NORM) / (MAX_DB_NORM - MIN_DB_NORM)
    normalized = torch.clamp(normalized, 0.0, 1.0)
    midi_velocity = normalized * 127.0
    return midi_velocity.round()

# calcula entropia espectral
def calculate_spectral_entropy(power_spectrum: torch.Tensor, epsilon=1e-10) -> torch.Tensor:
    power_spectrum = power_spectrum + epsilon 
    pdf = power_spectrum / power_spectrum.sum()
    log2_pdf = torch.log2(pdf)
    entropy = -torch.sum(pdf * log2_pdf)
    max_entropy = torch.log2(torch.tensor(power_spectrum.shape[0], dtype=torch.float32, device=power_spectrum.device))
    normalized_entropy = entropy / max_entropy
    return torch.clamp(normalized_entropy, 0.0, 1.0)

# calcula centróide espectral
def calculate_spectral_centroid(power_spectrum: torch.Tensor, mel_freqs_hz: np.ndarray, epsilon=1e-10) -> torch.Tensor:
    power_spectrum = power_spectrum + epsilon 
    mel_freqs_tensor = torch.tensor(mel_freqs_hz, device=power_spectrum.device, dtype=power_spectrum.dtype)
    weighted_sum = torch.sum(mel_freqs_tensor * power_spectrum)
    total_power = torch.sum(power_spectrum)
    centroid_hz = weighted_sum / total_power
    return centroid_hz

def calculate_spec_kurtosis(power_spectrum: torch.Tensor, mel_freqs_hz: np.ndarray, epsilon=1e-10) -> torch.Tensor:
    power_spectrum = power_spectrum + epsilon 
    mel_freqs_tensor = torch.tensor(mel_freqs_hz, device=power_spectrum.device, dtype=power_spectrum.dtype)
    centroid_hz = calculate_spectral_centroid(power_spectrum, mel_freqs_hz, epsilon)
    diff = mel_freqs_tensor - centroid_hz
    diff4 = diff ** 4
    weighted_diff4 = diff4 * power_spectrum
    kurtosis_numerator = torch.sum(weighted_diff4)
    total_power = torch.sum(power_spectrum)
    kurtosis = kurtosis_numerator / total_power
    return kurtosis

def normalize_minus1_1(x: torch.Tensor, xmin: float = 0.0, xmax: float = 127.0) -> torch.Tensor:
    """
    Normaliza amplitudes (ex.: MIDI 0–127) para o intervalo [-1, 1].
    - amps: tensor shape (..., 7)
    Retorna tensor normalizado com mesmo dtype/device.
    """
    x_min = float(xmin) + 1e-6
    x_max = float(xmax)
    x_clamped = torch.clamp(x, min=x_min, max=x_max)
    xmin_t = torch.as_tensor(xmin, dtype=x.dtype, device=x.device)
    xmax_t = torch.as_tensor(xmax, dtype=x.dtype, device=x.device)
    eps = torch.as_tensor(1e-8, dtype=x.dtype, device=x.device)
    x_log = torch.log(x_clamped)
    min_log = torch.log(xmin_t + eps)
    max_log = torch.log(xmax_t)
    denom = (max_log - min_log).clamp(min=1e-6)
    norm01 = (x_log - min_log) / denom
    return torch.clamp(norm01 * 2.0 - 1.0, -1.0, 1.0)

def normalize_texture_params_log11(texture6: torch.Tensor) -> torch.Tensor:
    """
    Normaliza [m1,m2,m3,m4,grain,âmbito] para [-1, 1]:
    - metrônomos (ms): log -> [-1, 1]
    - grain (ms):      log -> [-1, 1]
    - âmbito:          linear -> [-1, 1]
    """
    assert texture6.numel() == 6, "texture6 deve ter 6 elementos"
    metros = texture6[0:4]
    grain = texture6[4:5]
    ambito = texture6[5:6]
    metros_n = normalize_minus1_1(
        metros, NORMALIZATION_RANGES['metro']['min'], NORMALIZATION_RANGES['metro']['max']
    )
    grain_n = normalize_minus1_1(
        grain, NORMALIZATION_RANGES['grain']['min'], NORMALIZATION_RANGES['grain']['max']
    )
    ambito_n = normalize_minus1_1(
        ambito, NORMALIZATION_RANGES['ambito']['min'], NORMALIZATION_RANGES['ambito']['max']
    )
    return torch.cat([metros_n, grain_n, ambito_n], dim=0)


def denormalize_minus1_1(x_norm: torch.Tensor, xmin: float, xmax: float) -> torch.Tensor:
    """
    Desnormaliza valores em [-1, 1] para [xmin, xmax] usando a inversa da normalização logarítmica.
    Preserva dtype/device.
    """
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


def normalize_zscore(pitches: torch.Tensor, mean: float = 7800.0, std: float = 1200.0) -> torch.Tensor:
    """Normaliza pitches via Z-score """
    mean_t = torch.as_tensor(mean, dtype=pitches.dtype, device=pitches.device)
    std_t = torch.as_tensor(std, dtype=pitches.dtype, device=pitches.device).clamp(min=1e-8)
    return (pitches - mean_t) / std_t

def denormalize_pitches_zscore(pitches_norm: torch.Tensor, mean: float = 7800.0, std: float = 1200.0) -> torch.Tensor:
    """Desnormaliza pitches Z-score"""
    mean_t = torch.as_tensor(mean, dtype=pitches_norm.dtype, device=pitches_norm.device)
    std_t = torch.as_tensor(std, dtype=pitches_norm.dtype, device=pitches_norm.device).clamp(min=1e-8)
    return pitches_norm * std_t + mean_t


def generate_pitches_amps_from_class(seed, class_name, brightness_factor, duration_factor):
    """
    Gera pitches e amplitudes baseado na classe técnica SEM usar SRC
    """
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    
    # Perfis base por classe técnica (exemplos)
    class_profiles = {
        'aeolian': {'base_pitch': 6800, 'pitch_range': 600, 'base_amp': 80, 'amp_range': 30},
        'crescendo': {'base_pitch': 7600, 'pitch_range': 856, 'base_amp': 80, 'amp_range': 40},
        'crescendo_to_decrescendo': {'base_pitch': 6700, 'pitch_range': 500, 'base_amp': 85, 'amp_range': 37},
        'decrescendo': {'base_pitch': 7500, 'pitch_range': 500, 'base_amp': 80, 'amp_range': 20},
        'flatterzunge': {'base_pitch': 7400, 'pitch_range': 900, 'base_amp': 80, 'amp_range': 45},
        'flatterzunge_to_ordinario': {'base_pitch': 7500, 'pitch_range': 700, 'base_amp': 80, 'amp_range': 25},
        'jet_whistle': {'base_pitch': 9000, 'pitch_range': 1000, 'base_amp': 60, 'amp_range': 30},
        'multiphonics': {'base_pitch': 7200, 'pitch_range': 1200, 'base_amp': 70, 'amp_range': 25},
        'ordinario': {'base_pitch': 7600, 'pitch_range': 400, 'base_amp': 90, 'amp_range': 15},
        'ordinario_to_flatterzunge': {'base_pitch': 7500, 'pitch_range': 700, 'base_amp': 80, 'amp_range': 25},
        'sforzato': {'base_pitch': 7600, 'pitch_range': 625, 'base_amp': 80, 'amp_range': 30},
        'staccato': {'base_pitch': 7600, 'pitch_range': 700, 'base_amp': 80, 'amp_range': 20},
        'tongue_ram-pizz': {'base_pitch': 8500, 'pitch_range': 800, 'base_amp': 75, 'amp_range': 30},
        'trill': {'base_pitch': 7800, 'pitch_range': 600, 'base_amp': 80, 'amp_range': 30},
    }
    
    # Default para classes não especificadas
    profile = class_profiles.get(class_name.lower(), 
                               {'base_pitch': 7600, 'pitch_range': 600, 'base_amp': 80, 'amp_range': 20})
    
    # Gera 7 pitches com variação controlada
    base_pitches = profile['base_pitch'] + brightness_factor * profile['pitch_range']
    pitches = base_pitches + torch.randn(7, generator=gen, device=DEVICE) * profile['pitch_range'] * 0.2
    
    # Gera 7 amplitudes com variação controlada  
    base_amps = profile['base_amp'] + duration_factor * profile['amp_range']
    amps = base_amps + torch.randn(7, generator=gen, device=DEVICE) * profile['amp_range'] * 0.3
    
    # Ordena pitches e aplica clamping
    pitches = torch.sort(pitches)[0]
    pitches = torch.clamp(pitches, 6000.0, 9600.0)
    amps = torch.clamp(amps, 0.0, 127.0)
    
    return pitches, amps


def calculate_temporal_targets(melspec_power, folder_name, global_factors, frame_indices):
    """
    Calcula targets ESPECÍFICOS para cada um dos 10 frames temporais
    """
    temporal_targets = []
    
    for frame_idx in frame_indices:
        # 1. Potência APENAS deste frame específico
        frame_power = melspec_power[:, frame_idx]  # [64] - apenas este frame!
        
        # 2. Spectral features APENAS deste frame
        frame_entropy = calculate_spectral_entropy(frame_power)
        frame_centroid = calculate_spectral_centroid(frame_power, MEL_FREQS_HZ)
        frame_kurtosis = calculate_spec_kurtosis(frame_power, MEL_FREQS_HZ)

        fmin_hz=MELSPEC_PARAMS['fmin']; fmax_hz=MELSPEC_PARAMS['fmax']
        centroid_range=max(fmax_hz-fmin_hz,1.0)
        
        # 3. Brightness específico deste frame
        brightness_factor = float(torch.clamp((frame_centroid-fmin_hz)/centroid_range, 0.0, 1.0))  # float puro

        # Gera pitches e amplitudes baseado na classe técnica e brilho
        tgt_cents, tgt_amps = generate_pitches_amps_from_class(
            seed=global_factors['seed'] + frame_idx,
            class_name=folder_name,
            brightness_factor=brightness_factor,
            duration_factor=global_factors['duration_factor']
        )
        
        # # 4. calcula picos de pitch e amp de cada frames
        # top_k_values, top_k_indices = torch.topk(frame_power, N_PITCHES)
        # mel_bins_hz = MEL_FREQS_HZ[top_k_indices.cpu().numpy()]
        # midi_cents = librosa.hz_to_midi(mel_bins_hz) * 100.0
        # tgt_cents = torch.tensor(midi_cents, device=DEVICE, dtype=torch.float32)
        # tgt_amps = power_to_midi_velocity(top_k_values)

        # normaliza pitches e amps
        tgt_cents = normalize_minus1_1(tgt_cents, xmin=NORMALIZATION_RANGES['pitch']['min'], xmax=NORMALIZATION_RANGES['pitch']['max'])
        tgt_amps = normalize_minus1_1(tgt_amps, xmin=0, xmax=127.0)

        # 5. Textura com variação temporal 
        texture_seed = global_factors['seed'] + frame_idx  # Semente por frame
        tgt_texture = get_temporal_texture_params(
            texture_seed, folder_name, 
            global_factors['duration_factor'],
            frame_entropy.item(),
            brightness_factor,
            frame_idx  # Posição temporal
        )

        # normaliza textura
        tgt_texture = normalize_texture_params_log11(tgt_texture)

        frame_target = torch.cat([tgt_cents, tgt_amps, tgt_texture])
        temporal_targets.append(frame_target)
    
    return torch.stack(temporal_targets)  # [seq_len, features]

def get_temporal_texture_params(seed, folder_name, duration_factor, 
                              entropy_factor, brightness_factor, frame_position):
    """
    Gera parâmetros de textura com variação temporal suave
    """
    # Base por classe (como antes)
    base_params = get_process_params_for_label(
        seed, folder_name, duration_factor, entropy_factor, brightness_factor
    ).to(DEVICE) 
    
    # Variação mais natural baseada em walk aleatório
    gen = torch.Generator(device=DEVICE).manual_seed(seed + frame_position)
    
    # Ruído suave que acumula ao longo do tempo
    noise_scale = 0.05 + 0.1 * (frame_position / 10.0)  # Aumenta com o tempo
    temporal_noise = torch.randn(6, generator=gen, device=DEVICE) * noise_scale
    
    # Aplica variação seletivamente
    varied_params = base_params.clone()
    varied_params[0:4] += temporal_noise[0:4] * base_params[0:4]  # Metrônomos
    varied_params[4] += temporal_noise[4] * 0.1 * base_params[4]  # Grain
    varied_params[5] += temporal_noise[5] * 10.0  # Âmbito
    return varied_params


# --- Funções de Geração de metros ---
def _generate_metros(seed: int, min_ms: float, max_ms: float, entropy_factor: float = 0.0, variability: float = 0.0) -> torch.Tensor:
    gen = torch.Generator(); gen.manual_seed(seed) 
    gen_rand = torch.Generator(); gen_rand.manual_seed(seed + 1000)
    gen_noise = torch.Generator(); gen_noise.manual_seed(int(seed) + 2000)

    base_h = torch.rand(1, generator=gen) * (max_ms - min_ms) + min_ms
    m1_h, m2_h, m3_h, m4_h = base_h, base_h * 1.5, base_h * 2.0, base_h * 0.75
    harmonic_metros = torch.tensor([m1_h.item(), m2_h.item(), m3_h.item(), m4_h.item()])

    random_metros = torch.rand(4, generator=gen_rand) * (max_ms - min_ms) + min_ms
    
    factor = float(max(0.0, min(1.0, float(entropy_factor))))
    blended = (1.0 - factor) * harmonic_metros + factor * random_metros

    # ruído adicional para variar localmente cada metro
    v = float(max(0.0, min(1.0, float(variability))))
    if v > 0.0:
        sigma = v * max_ms - v * min_ms
        noise = torch.randn(4, generator=gen_noise, dtype=torch.float32) * sigma
        final_metros = blended + noise
    else:
        final_metros = blended

    final_metros = torch.clamp(final_metros, min_ms, max_ms).to(dtype=torch.float32)
    return final_metros.to(DEVICE)  # <- garante device

#--- Função de Geração de Grão e Âmbito ---
def _generate_grain_ambito(seed: int, grain_min: float, grain_max: float, ambito_min: float, ambito_max: float,
                           duration_factor: float = 0.5, brightness_factor: float = 0.5, variability: float = 0.5) -> torch.Tensor:
    gen = torch.Generator(); gen.manual_seed(seed + 1)

    # Clamp factors
    d_f = float(max(0.0, min(1.0, duration_factor)))
    b_f = float(max(0.0, min(1.0, brightness_factor)))
    v_f = float(max(0.0, min(1.0, variability)))

    # Grain
    grain_base = (grain_min + grain_max) / 2.0
    grain_range = max(grain_max - grain_min, 1e-6)
    det_grain = grain_base + (d_f - 0.5) * grain_range  # deslocamento em ±0.5*range
    sigma_grain = v_f * grain_range
    noise_grain = torch.randn(1, generator=gen).item() * sigma_grain
    final_grain = float(torch.clamp(torch.tensor(det_grain + noise_grain), grain_min, grain_max).item())

    # Âmbito: (centro + deslocamento + ruído)
    ambito_base = (ambito_min + ambito_max) / 2.0
    ambito_range = ambito_max - ambito_min if (ambito_max - ambito_min) > 0 else 0.1
    det_ambito = ambito_base + (b_f - 0.5) * ambito_range
    sigma_ambito = v_f * ambito_range
    noise_ambito = torch.randn(1, generator=gen).item() * sigma_ambito
    final_ambito = float(torch.clamp(torch.tensor(det_ambito + noise_ambito), ambito_min, ambito_max).item())
   
    return torch.tensor([final_grain, final_ambito], dtype=torch.float32, device=DEVICE)

### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ###
# definição heurística dos parâmetros de textura por classe
### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ###
def get_process_params_for_label(seed: int, folder_name: str, duration_factor: float, 
                                 entropy_factor: float, brightness_factor: float) -> torch.Tensor:
    folder_name = folder_name.lower()
    is_dense = False; is_sparse = False; is_dilated = False; is_contracted = False; is_impulse = False; is_noisy = False; 
    is_alure = False; is_evolutive = False;

    # definição das texturas por classe
    if 'aeolian' in folder_name: 
        is_sparse = True
        is_contracted = True
        is_noisy = True
        metro_min, metro_max = 135.3, 452.7; grain_min, grain_max = 50, 123; ambito_min, ambito_max = 10, 21
        print(f"  Classe '{folder_name}': Mapeada para Densa + Contraída")

    elif 'crescendo' in folder_name: 
        is_sparse = True
        is_dilated = True
        is_evolutive = True
        metro_min, metro_max = 442.84, 1230.0; grain_min, grain_max = 50, 75.34; ambito_min, ambito_max = 10, 43
        print(f"  Classe '{folder_name}': Mapeada para Rarefeita + Dilatada") 

    elif 'crescendo_to_decrescendo' in folder_name: 
        is_sparse = True
        is_contracted = True
        is_evolutive = True
        metro_min, metro_max = 442.84, 847.23; grain_min, grain_max = 63.87, 75.34; ambito_min, ambito_max = 10, 64.73
        print(f"  Classe '{folder_name}': Mapeada para Rarefeita + Contraída")

    elif 'decrescendo' in folder_name: 
        is_dense = True
        is_dilated = True
        is_evolutive = True
        metro_min, metro_max = 723.54, 1645.43; grain_min, grain_max = 74.85, 203; ambito_min, ambito_max = 10, 15.64
        print(f"  Classe '{folder_name}': Mapeada para Densa + Dilatada")

    elif 'flatterzunge' in folder_name: 
        is_dense = True
        is_contracted = True
        is_alure = True
        metro_min, metro_max = 236.74, 874.83; grain_min, grain_max = 50, 94.65; ambito_min, ambito_max = 10, 78.21
        print(f"  Classe '{folder_name}': Mapeada para Densa + Contraída")

    elif 'flatterzunge_to_ordinario' in folder_name: 
        is_dense = True
        is_dilated = True
        is_evolutive = True
        metro_min, metro_max = 123.9, 2394.0; grain_min, grain_max = 50, 45; ambito_min, ambito_max = 10, 37.4
        print(f"  Classe '{folder_name}': Mapeada para Densa + Dilatada")  

    elif 'jet_whistle' in folder_name: 
        is_dense = True
        is_dilated = True
        is_noisy = True 
        metro_min, metro_max = 347, 1574.75; grain_min, grain_max = 50, 200; ambito_min, ambito_max = 10, 85.34
        print(f"  Classe '{folder_name}': Mapeada para Densa + Dilatada")

    elif 'multiphonics' in folder_name: 
        is_sparse = True
        is_dilated = True
        is_alure = True
        metro_min, metro_max = 100, 3849; grain_min, grain_max = 50, 77; ambito_min, ambito_max = 10, 32.4
        print(f"  Classe '{folder_name}': Mapeada para rarefeita + Dilatada")

    elif 'ordinario' in folder_name: 
        is_sparse = True
        is_contracted = True
        is_alure = True
        metro_min, metro_max = 847.36, 2656.55; grain_min, grain_max = 50, 95; ambito_min, ambito_max = 10, 94.12
        print(f"  Classe '{folder_name}': Mapeada para rarefeita + Contraída")

    elif 'ordinario_to_flatterzunge' in folder_name: 
        is_dense = True
        is_dilated = True
        is_evolutive = True
        metro_min, metro_max = 1014.6, 4083.56; grain_min, grain_max = 50, 66; ambito_min, ambito_max = 10, 55.32
        print(f"  Classe '{folder_name}': Mapeada para rarefeita + dilatada")

    elif 'sforzato' in folder_name: 
        is_sparse = True
        is_dilated = True
        is_impulse = True
        metro_min, metro_max = 1584.6, 3946.23; grain_min, grain_max = 50, 156; ambito_min, ambito_max = 10, 27
        print(f"  Classe '{folder_name}': Mapeada para rarefeita + dilatada")

    elif 'staccato' in folder_name: 
        is_dense = True
        is_dilated = True
        is_impulse = True
        metro_min, metro_max = 746, 4536.73; grain_min, grain_max = 50, 243; ambito_min, ambito_max = 10, 46.87
        print(f"  Classe '{folder_name}': Mapeada para densa + dilatada")

    elif 'tongue_ram-pizz' in folder_name: 
        is_dense = True
        is_contracted = True
        is_impulse = True
        metro_min, metro_max = 234.56, 3173.8; grain_min, grain_max = 50, 172; ambito_min, ambito_max = 10, 96.18
        print(f"  Classe '{folder_name}': Mapeada para densa + contraída")

    elif 'trill' in folder_name: 
        is_dense = True
        is_dilated = True
        is_alure = True
        metro_min, metro_max = 154, 236; grain_min, grain_max = 50, 88.5; ambito_min, ambito_max = 10, 13.4
        print(f"  Classe '{folder_name}': Mapeada para densa + dilatada")
    
    # Define ranges baseados nas heurísticas
    # metro_min, metro_max = 100, 5000; grain_min, grain_max = 50, 1500; ambito_min, ambito_max = -100, 100
    # if is_impulse: metro_min, metro_max = 100, 700
    # elif is_noisy: metro_min, metro_max = 1500, 3000
    # elif is_alure: metro_min, metro_max = 3000, 6000
    # elif is_evolutive: metro_min, metro_max = 5500, 8000
    # if is_dilated: grain_min, grain_max = 500, 1500; ambito_min, ambito_max = 10, 100
    # elif is_contracted: grain_min, grain_max = 50, 500; ambito_min, ambito_max = 65, 100

    gen_var = torch.Generator(); gen_var.manual_seed(seed + 42)

    MIN_VARIABILITY = 0.1 
    MAX_VARIABILITY = 0.7
    sampled_variability = torch.rand(1, generator=gen_var).item() * (MAX_VARIABILITY - MIN_VARIABILITY) + MIN_VARIABILITY

    # Gera targets de textura
    metros_tensor = _generate_metros(seed, metro_min, metro_max, entropy_factor=entropy_factor, variability=sampled_variability) 
    grain_ambito_tensor = _generate_grain_ambito(seed, grain_min, grain_max, ambito_min, ambito_max, 
                                                duration_factor=duration_factor, brightness_factor=brightness_factor, variability=sampled_variability) 
    final_params = torch.cat([metros_tensor, grain_ambito_tensor]).to(DEVICE)
    return final_params


# --- Lógica Principal de Criação do Dataset ---
def create_dataset(csv_path, audio_root, save_dir, hop_step=1):
    """
    Lê o CSV, processa áudio, extrai janelas (src), gera alvos (tgt) HÍBRIDOS e NORMALIZA.
    """
    print(f"Usando dispositivo: {DEVICE}")
    print(f"Target Features: {TARGET_FEATURES} (Pitches: {N_PITCHES}, Amps: {N_AMPS}, Textura: {N_TEXTURE_PARAMS})")
    
    # 1. Carregar CSV e normalizar durações 
    try: #
        df = pd.read_csv(csv_path)
        all_durations = df['duration_ms'].dropna()
        if len(all_durations) > 1:
            MIN_DURATION_MS = all_durations.min(); MAX_DURATION_MS = all_durations.max()
            DURATION_RANGE = max(MAX_DURATION_MS - MIN_DURATION_MS, 1.0) 
            print(f"Normalização de Duração: Min={MIN_DURATION_MS:.0f}ms, Max={MAX_DURATION_MS:.0f}ms")
        else:
            MIN_DURATION_MS = 0.0; DURATION_RANGE = 1.0; print("Aviso: Não foi possível calcular range de duração.")
    except FileNotFoundError: print(f"Erro: Arquivo CSV não encontrado em {csv_path}"); return
    print(f"CSV balanceado carregado com {len(df)} arquivos.")

    # 2. Inicializar funções de análise de áudio
    print("Inicializando ContorchionistMelTransform...") #
    transform = ContorchionistMelTransform(device=DEVICE, verbose=MELSPEC_PARAMS['verbose']) #
    to_db = T.AmplitudeToDB(stype='power').to(DEVICE) #

    # 3. Listas para armazenar amostras
    all_src_samples = []; all_tgt_samples = []; all_labels = [] #

    print(f"Iniciando processamento de {len(df)} arquivos...")
    
    # 4. Iterar sobre o CSV
    for idx, row in df.iterrows(): #
        file_path = os.path.join(audio_root, row['path']); label = int(row['label']); folder_name = row['folder']; duration_ms = row['duration_ms']
        if pd.notna(duration_ms): duration_factor = np.clip((duration_ms - MIN_DURATION_MS) / DURATION_RANGE, 0.0, 1.0)
        else: duration_factor = 0.5 
        if idx % 20 == 0: print(f"  Processando {idx+1}/{len(df)}: {row['filename']}") #

        try:
            # carrega áudio, ajusta taxa de amostragem e converte para mono
            waveform, sr = torchaudio.load(file_path); waveform = waveform.to(DEVICE)
            target_sr = MELSPEC_PARAMS['sample_rate']
            if sr != target_sr: waveform = get_resampler(sr, target_sr)(waveform)
            if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)

            # 5. Calcula Melspecs
            melspec = transform(waveform).squeeze(0) # (64, L) 
            melspec_power = melspec ** 2
            melspec_db = to_db(melspec_power.unsqueeze(0)).squeeze(0) # (64, L)
            total_frames = melspec_db.shape[1]
            if total_frames < N_FRAMES: continue

            

            # 6. Aplicar Janela Deslizante
            for i in range(0, total_frames - N_FRAMES + 1, hop_step):
                # 6.1. aplica janela deslizante no melspec e transpõe de [64, 10] para [10, 64]
                melspec_window = melspec[:, i : i + N_FRAMES];  melspec_window_transposed = melspec_window.T # transpõe (10, 64)
                # 6.2. aplica janela deslizante no melspec de potência para análise espectral
                src_window_power = melspec_power[:, i : i + N_FRAMES] # (64, 10) #
                mean_spectrum_power = torch.mean(src_window_power, dim=1) # (64) #

                unique_seed = int(idx * 10000 + i) # semente única por arquivo + frame
                
                # 6.3. Pitches e Amps
                # top_k_values, top_k_indices = torch.topk(mean_spectrum_power, N_PITCHES) #
                # mel_bins=top_k_indices.cpu().numpy(); mel_bins_hz=MEL_FREQS_HZ[mel_bins]
                # midi_cents=librosa.hz_to_midi(mel_bins_hz)*100.0
                # tgt_cents_series=torch.tensor(midi_cents,device=DEVICE,dtype=torch.float32) # (7) #
                # tgt_amps_series=power_to_midi_velocity(top_k_values) # (7) #

                # 6.4. Fatores de Modulação 
                entropy_factor=calculate_spectral_entropy(mean_spectrum_power).item() #
                spectral_centroid_hz=calculate_spectral_centroid(mean_spectrum_power,MEL_FREQS_HZ)
                kurtosis_factor=calculate_spec_kurtosis(mean_spectrum_power,MEL_FREQS_HZ).item() #
                fmin_hz=MELSPEC_PARAMS['fmin']; fmax_hz=MELSPEC_PARAMS['fmax']
                centroid_range=max(fmax_hz-fmin_hz,1.0)
                brightness_factor=torch.clamp((spectral_centroid_hz-fmin_hz)/centroid_range,0.0,1.0).item() #

                # # 6.5. Parâmetros de Textura
                # tgt_texture_series = get_process_params_for_label(
                #     unique_seed, folder_name, duration_factor, entropy_factor, brightness_factor
                # ).to(DEVICE) # (6) #

                # # 6.6. Combinar TGT
                # tgt_row_static = torch.cat([tgt_cents_series, tgt_amps_series, tgt_texture_series]) # (20) #

                # # 6.7. Normalizar TGT
                # tgt_row_normalized = normalize_tensor(tgt_row_static) # (20) , valores [0, 1]

                # 6.8. Criar Sequência TGT Normalizada (Repetir linha normalizada)
                # tgt_sequence_normalized = tgt_row_normalized.repeat(N_FRAMES, 1) # (10, 20)

                temporal_targets = calculate_temporal_targets(
                    melspec_power=melspec_power[:, i:i+N_FRAMES],  # Janela temporal
                    folder_name=folder_name,
                    global_factors={
                        'seed': unique_seed,
                        'duration_factor': duration_factor,
                        'entropy_factor': entropy_factor,
                        'brightness_factor': brightness_factor
                    },
                    frame_indices=range(N_FRAMES)  # Todos os frames da janela  # (10, 20    )
                )

            
                # 6.9. Adicionar (SRC, TGT Normalizado, Label)
                all_src_samples.append(melspec_window_transposed)
                all_tgt_samples.append(temporal_targets) # Salva o TGT normalizado
                all_labels.append(label)

        except Exception as e:
            print(f"    ERRO ao processar {file_path}: {e}")

    # 7. salva tensores
    print("\nProcessamento de arquivos concluído.") #
    if not all_src_samples: print("Erro: Nenhuma amostra foi extraída."); return #

    # 8. Análise de Variação do Target de Textura
    print("\n=== ANÁLISE DE VARIAÇÃO DO TARGET (Textura Normalizada [-1,1]) ===")
    if all_tgt_samples:
        temp_tgt_tensor = torch.stack(all_tgt_samples) # Mantém no DEVICE
        temp_labels_tensor = torch.tensor(all_labels, device=DEVICE)
        
        unique_labels = torch.unique(temp_labels_tensor).sort().values
        label_to_folder = df.set_index('label')['folder'].to_dict()

        for unique_label in unique_labels.tolist():
            class_mask = (temp_labels_tensor == unique_label)
            class_tgts = temp_tgt_tensor[class_mask] # Amostras X 10 X 20
            
            # Pega o target do primeiro passo (todos são iguais) e isola textura
            class_texture_targets = class_tgts[:, 0, 14:] # Amostras X 6 
            
            if class_texture_targets.shape[0] > 1:
                # Calcula std dev para as 6 features de textura
                std_dev = torch.std(class_texture_targets, dim=0) 
                folder_name = label_to_folder.get(unique_label, f"Label {unique_label}")
                print(f"\n--- Classe: {folder_name} (Label: {unique_label}) ---")
                print(f"  No. Amostras: {class_texture_targets.shape[0]}")
                print(f"  Desvio Padrão (Metros, Grain, Âmbito): {std_dev.cpu().numpy().round(4)}")
            elif class_texture_targets.shape[0] == 1:
                 print(f"\n--- Classe: {folder_name} (Label: {unique_label}) ---")
                 print(f"  No. Amostras: 1 (Desvio padrão não aplicável)")
            else:
                 print(f"\n--- Classe: {folder_name} (Label: {unique_label}) ---")
                 print(f"  No. Amostras: 0")

    else:
        print("Nenhuma amostra TGT para analisar.")

    print("Empilhando tensores...") #
    try:
        final_src = torch.stack(all_src_samples).cpu() # (N, 10, 64) #
        final_tgt = torch.stack(all_tgt_samples).cpu() # (N, 10, 20) #
        final_labels = torch.tensor(all_labels, dtype=torch.long).cpu() # (N,) #
    except Exception as e: print(f"Erro ao empilhar tensores: {e}"); return #

    print("\n--- Estatísticas Finais do Dataset ---") #
    print(f"Total de amostras (janelas de 10 frames): {len(final_src)}") #
    print(f"Shape do tensor SRC: {final_src.shape}") #
    print(f"Shape do tensor TGT (Normalizado): {final_tgt.shape}") #
    print(f"Shape do tensor LABELS: {final_labels.shape}") 

    print("\n=== EXEMPLO DE AMOSTRAS POR CLASSE (APÓS PROCESSAMENTO) ===")
    printed_labels = set()
    label_to_folder = df.set_index('label')['folder'].to_dict() # Mapeamento reverso

    num_unique_labels = len(label_to_folder)
    
    for i in range(len(final_labels)):
        label = final_labels[i].item()
        if label not in printed_labels:
            folder_name = label_to_folder.get(label, f"Label Desconhecido {label}")
            print(f"\n--- Exemplo para Classe: {folder_name} (Label: {label}) ---")
            
            # SRC (Melspectrograma dB)
            example_src = final_src[i] # Shape (10, 64)
            print(f"  SRC Shape: {example_src.shape}")
            # Mostra uma pequena parte do primeiro frame
            print(f"  SRC (Início Frame 0, dB): {example_src[0, :5].numpy()}...") 
            
            # TGT (Normalizado para [0, 1])
            example_tgt_normalized = final_tgt[i] # Shape (10, 20)
            # Mostra o target do primeiro passo (todos são iguais)
            tgt_step_0_norm = example_tgt_normalized[0] # Shape (20)
            print(f"  TGT Shape (Normalizado): {example_tgt_normalized.shape}")
            print(f"  TGT (Passo 0, Normalizado [-1,1]):")
            # Imprime formatado para melhor leitura
            print(f"    Pitches (7): {tgt_step_0_norm[0:7].numpy().round(3)}")
            print(f"    Amps (7):    {tgt_step_0_norm[7:14].numpy().round(3)}")
            print(f"    Textura (6): {tgt_step_0_norm[14:].numpy().round(3)}") # Metros, Grain, Âmbito
            
            printed_labels.add(label)
            
            # Opcional: parar se já imprimiu um exemplo de cada classe
            if len(printed_labels) == num_unique_labels:
                print("\nTodos os exemplos de classe foram impressos.")
                break

    # 8. Salvar Tensores E Ranges de Normalização
    os.makedirs(save_dir, exist_ok=True) #
    src_path = os.path.join(save_dir, 'train_src.pt') #
    tgt_path = os.path.join(save_dir, 'train_tgt.pt') #
    label_path = os.path.join(save_dir, 'train_labels.pt') #
    ranges_path = os.path.join(save_dir, 'normalization_ranges.json')

    print(f"\nSalvando dataset em {save_dir}...") #
    try:
        torch.save(final_src, src_path) #
        torch.save(final_tgt, tgt_path) #
        torch.save(final_labels, label_path) #
        # Salva os ranges como JSON
        with open(ranges_path, 'w') as f:
            json.dump(NORMALIZATION_RANGES, f, indent=4)
            
        print("Dataset e ranges de normalização salvos com sucesso!")
        print(f"  SRC: {src_path}")
        print(f"  TGT (Normalizado): {tgt_path}")
        print(f"  LABELS: {label_path}")
        print(f"  RANGES: {ranges_path}") 
    except Exception as e:
        print(f"Erro ao salvar arquivos: {e}")


def main():
    # hop_step=5 (sobreposição de 50%) é um bom começo
    create_dataset(CSV_PATH, AUDIO_DIR_ROOT, DATASET_DIR, hop_step=5)

if __name__ == '__main__':
    main()