import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio

import numpy as np

try:
    import pycontorchionist as cc
except ImportError:
    print("Erro: Não foi possível importar o módulo pycontorchionist.")
    print("Verifique se o arquivo do módulo (pycontorchionist...so ou .pyd) está no mesmo diretório que este script.")
    exit(1)


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