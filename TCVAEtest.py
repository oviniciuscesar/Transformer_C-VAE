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

DIRECTORY = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(DIRECTORY, "TorchScript")

# Parâmetros do Modelo (targets)
N_FRAMES = 10       # 10 frames de melspectrograma
N_PITCHES = 7       # 7 notas
N_AMPS = 7          # 7 amplitudes
N_TEXTURE_PARAMS = 6 # 4 (metros) + 1 (grão) + 1 (âmbito) = 6
TARGET_FEATURES = N_PITCHES + N_AMPS + N_TEXTURE_PARAMS  # target completo: 7 + 7 + 6 = 20

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'


# definição dos ranges globais de normalização
# !!! IMPORTANTE: ajustar com base nas heurísticas dos targets !!!
NORMALIZATION_RANGES = {
    # Features 0-6: Pitches (MIDI Cents)
    'pitch': {'min': 6000.0, 'max': 9600.0}, # três oitavas
    # Features 7-13: Amplitudes (MIDI Velocity)
    'amp': {'min': 0.0, 'max': 127.0},
    # Features 14-17: Metrônomos (ms)
    'metro': {'min': 10, 'max': 8000.0}, # 
    # Feature 18: Grain Size (ms)
    'grain': {'min': 10.0, 'max': 2500.0}, # 
    # Feature 19: Âmbito
    'ambito': {'min': -100, 'max': 100}
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

# ====== PARÂMETROS MEL SPECTROGRAM ======
audio_dir = os.path.join(os.path.dirname(__file__), 'Flute/multiphonics/')
AUDIO_PATH = os.path.join(audio_dir, 'Fl-mul-A5_G4-mf-N-N.wav')
SAMPLE_RATE = 44100
N_MELS = 64
N_FFT = 2048
HOP_LENGTH = 512
FMIN = 0.0
FMAX = SAMPLE_RATE // 2
MEL_NORM = "slaney"
MEL_MODE = cc.MelNormMode.ENERGY_POWER
NORM = 'N_FFT'  # Normalização da RFFT
MEL_FORMULA = 'HTK'  # Fórmula Mel  
WINDOW_TYPE = 'HANN'  # Tipo de janela
UNIT = 'MAGPHASE'  # Formato de saída do espectro
print(f"Parâmetros: {SAMPLE_RATE}, {N_MELS}, {N_FFT}, {HOP_LENGTH}, {FMIN}, {FMAX}, {MEL_NORM}, {MEL_MODE}, {NORM}, {MEL_FORMULA}, {WINDOW_TYPE}, {UNIT} ({cc.mel_norm_mode_to_string(MEL_MODE)})")


# # ====== CARREGAR ÁUDIO ======
waveform, sr = torchaudio.load(AUDIO_PATH)
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)  # Mono
waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
audio = waveform.squeeze().numpy()

print(f"Áudio carregado: {audio.shape}, SR={SAMPLE_RATE}")

# ====== MEL SPECTROGRAM PYTORCH ======
mel_torch = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    f_min=FMIN,
    f_max=FMAX,
    norm=MEL_NORM,
    power=2.0,
)(torch.tensor(audio)).numpy()
print(f"MelSpectrogram PyTorch: {mel_torch.shape}")


# ====== MEL SPECTROGRAM CONTORCHIONIST ======
processor = cc.MelSpectrogramProcessor()
processor.set_sample_rate(float(SAMPLE_RATE))
processor.set_n_fft(N_FFT)
processor.set_hop_length(HOP_LENGTH)
processor.set_n_mels(N_MELS)
processor.set_fmin_mel(float(FMIN))
processor.set_fmax_mel(float(FMAX))
processor.set_mel_formula(getattr(cc.MelFormulaType, 'SLANEY'))
processor.set_window_type(getattr(cc.WindowType, 'HANN'))
processor.set_normalization_type(getattr(cc.NormalizationType, 'POWER'))
processor.set_filterbank_norm(MEL_NORM)
processor.set_mel_norm_mode(MEL_MODE)
processor.set_mel_formula(getattr(cc.MelFormulaType, MEL_FORMULA))
processor.set_output_format(cc.SpectrumDataFormat.MAGPHASE)


## Processar em blocos
frames = []
block_size = HOP_LENGTH
for i in range(0, len(audio), block_size):
    chunk = audio[i:i+block_size].astype(np.float32)
    mel = processor.process(chunk)
    if mel is not None:
        frames.append(np.array(mel))
mel_contorch = np.stack(frames, axis=0) if frames else np.zeros((N_MELS, 1))
# mel_tensor = mel_contorch[:N_FRAMES]
mel_tensor = torch.from_numpy(mel_contorch[:N_FRAMES]).to(torch.float32)
mel_tensor = mel_tensor.flatten().unsqueeze(0)  # [1, frames*mel_bins]
print(f"MelSpectrogram Contorchionist: {mel_contorch.shape}")
print(f"--- Tensor Melspec: {mel_tensor.shape} ---")
print(mel_tensor)


# carrega modelo torchscript
loaded_model = torch.jit.load(os.path.join(MODEL_DIR, 'tc-vae.ts'))
loaded_model.eval()

# define número de steps de saída
loaded_model.steps(1) 

# z fictício para teste
dummy_z = torch.randn(32)

loaded_model.latent(dummy_z) # seta z controlado
with torch.no_grad():
        output = loaded_model.forwardz(mel_tensor)
print(f"Output modelo: {output}")  # deve ser [1, N_TARGET_FEATURES]

