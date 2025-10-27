import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import torchaudio.transforms as T
import numpy as np
from pathlib import Path
import librosa
import json ### NOVO IMPORT ###

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
    'n_mels': 64,
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
    'pitch': {'min': 6000.0, 'max': 8400.0}, # duas oitavas
    # Features 7-13: Amplitudes (MIDI Velocity)
    'amp': {'min': 0.0, 'max': 127.0},
    # Features 14-17: Metrônomos (ms)
    'metro': {'min': 100, 'max': 5000.0}, # 
    # Feature 18: Grain Size (ms)
    'grain': {'min': 50.0, 'max': 1500.0}, # 
    # Feature 19: Âmbito
    'ambito': {'min': -1.0, 'max': 1.0}
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
    max_entropy = torch.log2(torch.tensor(power_spectrum.shape[0], dtype=torch.float32))
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

# Normaliza o tensor de target
def normalize_tensor(tensor_1d: torch.Tensor) -> torch.Tensor:
    """Normaliza um tensor 1D (shape 20) para o intervalo [0, 1] usando ranges globais."""
    min_vals = TGT_MIN_VALS.to(tensor_1d.device)
    ranges = TGT_RANGES.to(tensor_1d.device)
    normalized = (tensor_1d - min_vals) / ranges
    return torch.clamp(normalized, 0.0, 1.0) # clampa para [0, 1] por segurança

# --- Funções de Geração de metros ---
def _generate_metros(seed: int, min_ms: float, max_ms: float, entropy_factor: float = 0.0) -> torch.Tensor:
    gen = torch.Generator(); gen.manual_seed(seed) 
    base_h = torch.rand(1, generator=gen) * (max_ms - min_ms) + min_ms
    m1_h, m2_h, m3_h, m4_h = base_h, base_h * 1.5, base_h * 2.0, base_h * 0.75
    harmonic_metros = torch.tensor([m1_h.item(), m2_h.item(), m3_h.item(), m4_h.item()])
    gen_rand = torch.Generator(); gen_rand.manual_seed(seed + 1000)
    random_metros = torch.rand(4, generator=gen_rand) * (max_ms - min_ms) + min_ms
    factor = torch.clamp(torch.tensor(entropy_factor), 0.0, 1.0)
    final_metros = (1.0 - factor) * harmonic_metros + factor * random_metros
    final_metros = torch.clamp(final_metros, min_ms, max_ms)
    return final_metros

#--- Função de Geração de Grão e Âmbito ---
def _generate_grain_ambito(seed: int, grain_min: float, grain_max: float, ambito_min: float, ambito_max: float,
                           duration_factor: float = 0.5, brightness_factor: float = 0.5) -> torch.Tensor:
    gen = torch.Generator(); gen.manual_seed(seed + 1)
    # Grain
    grain_base=(grain_min+grain_max)/2.0; grain_range=grain_max-grain_min
    duration_mod=torch.clamp(torch.tensor(duration_factor),0.0,1.0)-0.5 
    modulated_grain=grain_base+duration_mod*grain_range 
    final_grain=torch.clamp(modulated_grain,grain_min,grain_max)
    # Âmbito
    ambito_base=(ambito_min+ambito_max)/2.0; ambito_range=ambito_max-ambito_min if (ambito_max-ambito_min)>0 else 0.1 
    brightness_mod=torch.clamp(torch.tensor(brightness_factor),0.0,1.0)-0.5
    modulated_ambito=ambito_base+brightness_mod*ambito_range 
    final_ambito=torch.clamp(modulated_ambito,ambito_min,ambito_max)
    return torch.tensor([final_grain.item(), final_ambito.item()])

### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ###
# definição heurística dos parâmetros de textura por classe
### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ###
def get_process_params_for_label(label: int, folder_name: str, duration_factor: float, 
                                 entropy_factor: float, brightness_factor: float) -> torch.Tensor:
    folder_name = folder_name.lower()
    is_dense = False; is_sparse = False; is_dilated = False; is_contracted = False

    # definição das texturas por classe
    if 'aeolian' in folder_name: 
        is_sparse = True
        is_contracted = True
        print(f"  Classe '{folder_name}': Mapeada para Densa + Contraída")

    elif 'crescendo' in folder_name: 
        is_sparse = True
        is_dilated = True
        print(f"  Classe '{folder_name}': Mapeada para Rarefeita + Dilatada") 

    elif 'crescendo_to_decrescendo' in folder_name: 
        is_sparse = True
        is_contracted = True
        print(f"  Classe '{folder_name}': Mapeada para Rarefeita + Contraída")

    elif 'decrescendo' in folder_name: 
        is_dense = True
        is_dilated = True
        print(f"  Classe '{folder_name}': Mapeada para Densa + Dilatada")

    elif 'flatterzunge' in folder_name: 
        is_dense = True
        is_contracted = True
        print(f"  Classe '{folder_name}': Mapeada para Densa + Contraída")

    elif 'flatterzunge_to_ordinario' in folder_name: 
        is_dense = True
        is_dilated = True
        print(f"  Classe '{folder_name}': Mapeada para Densa + Dilatada")  

    elif 'jet_whistle' in folder_name: 
        is_dense = True
        is_dilated = True
        print(f"  Classe '{folder_name}': Mapeada para Densa + Dilatada")

    elif 'multiphonics' in folder_name: 
        is_sparse = True
        is_dilated = True
        print(f"  Classe '{folder_name}': Mapeada para rarefeita + Dilatada")

    elif 'ordinario' in folder_name: 
        is_sparse = True
        is_contracted = True
        print(f"  Classe '{folder_name}': Mapeada para rarefeita + Contraída")

    elif 'ordinario_to_flatterzunge' in folder_name: 
        is_dense = True
        is_dilated = True
        print(f"  Classe '{folder_name}': Mapeada para rarefeita + dilatada")

    elif 'sforzato' in folder_name: 
        is_sparse = True
        is_dilated = True
        print(f"  Classe '{folder_name}': Mapeada para rarefeita + dilatada")

    elif 'staccato' in folder_name: 
        is_dense = True
        is_dilated = True
        print(f"  Classe '{folder_name}': Mapeada para densa + dilatada")

    elif 'tongue_ram-pizz' in folder_name: 
        is_dense = True
        is_contracted = True
        print(f"  Classe '{folder_name}': Mapeada para densa + contraída")

    elif 'trill' in folder_name: 
        is_dense = True
        is_dilated = True
        print(f"  Classe '{folder_name}': Mapeada para densa + dilatada")
    
    # Define ranges baseados nas heurísticas
    metro_min, metro_max = 100, 5000; grain_min, grain_max = 50, 1500; ambito_min, ambito_max = -1, 1
    if is_dense: metro_min, metro_max = 200, 2000.0
    elif is_sparse: metro_min, metro_max = 2000.0, 5000.0
    if is_dilated: grain_min, grain_max = 500.0, 1000.0; ambito_min, ambito_max = 0, 1.0
    elif is_contracted: grain_min, grain_max = 50.0, 300.0; ambito_min, ambito_max = -1.0, 0.0


    # Gera targets de textura
    metros_tensor = _generate_metros(label, metro_min, metro_max, entropy_factor=entropy_factor) 
    grain_ambito_tensor = _generate_grain_ambito(label, grain_min, grain_max, ambito_min, ambito_max, 
                                                duration_factor=duration_factor, brightness_factor=brightness_factor) 
    final_params = torch.cat([metros_tensor, grain_ambito_tensor])
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
                #src_window_db = melspec_db[:, i : i + N_FRAMES] #src_window_transposed = src_window_db.T # (10, 64) #
                # 6.2. aplica janela deslizante no melspec de potência para análise espectral
                src_window_power = melspec_power[:, i : i + N_FRAMES] # (64, 10) #
                mean_spectrum_power = torch.mean(src_window_power, dim=1) # (64) #

                # 6.3. Pitches e Amps
                top_k_values, top_k_indices = torch.topk(mean_spectrum_power, N_PITCHES) #
                mel_bins=top_k_indices.cpu().numpy(); mel_bins_hz=MEL_FREQS_HZ[mel_bins]
                midi_cents=librosa.hz_to_midi(mel_bins_hz)*100.0
                tgt_cents_series=torch.tensor(midi_cents,device=DEVICE,dtype=torch.float32) # (7) #
                tgt_amps_series=power_to_midi_velocity(top_k_values) # (7) #

                # 6.4. Fatores de Modulação 
                entropy_factor=calculate_spectral_entropy(mean_spectrum_power).item() #
                spectral_centroid_hz=calculate_spectral_centroid(mean_spectrum_power,MEL_FREQS_HZ)
                fmin_hz=MELSPEC_PARAMS['fmin']; fmax_hz=MELSPEC_PARAMS['fmax']
                centroid_range=max(fmax_hz-fmin_hz,1.0)
                brightness_factor=torch.clamp((spectral_centroid_hz-fmin_hz)/centroid_range,0.0,1.0).item() #

                # 6.5. Parâmetros de Textura
                tgt_texture_series = get_process_params_for_label(
                    label, folder_name, duration_factor, entropy_factor, brightness_factor
                ).to(DEVICE) # (6) #

                # 6.6. Combinar TGT
                tgt_row_static = torch.cat([tgt_cents_series, tgt_amps_series, tgt_texture_series]) # (20) #

                # 6.7. Normalizar TGT
                tgt_row_normalized = normalize_tensor(tgt_row_static) # (20) , valores [0, 1]

                # 6.8. Criar Sequência TGT Normalizada (Repetir linha normalizada)
                tgt_sequence_normalized = tgt_row_normalized.repeat(N_FRAMES, 1) # (10, 20)

                # 6.9. Adicionar (SRC, TGT Normalizado, Label)
                all_src_samples.append(melspec_window_transposed)
                all_tgt_samples.append(tgt_sequence_normalized) # Salva o TGT normalizado
                all_labels.append(label)

        except Exception as e:
            print(f"    ERRO ao processar {file_path}: {e}")

    # 7. Empilhar e Salvar (Idêntico, mas agora com TGTs normalizados)
    print("\nProcessamento de arquivos concluído.") #
    if not all_src_samples: print("Erro: Nenhuma amostra foi extraída."); return #

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
            print(f"  TGT (Passo 0, Normalizado [0,1]):")
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
        print(f"  RANGES: {ranges_path}") ### NOVO ###
    except Exception as e:
        print(f"Erro ao salvar arquivos: {e}")


def main():
    # hop_step=5 (sobreposição de 50%) é um bom começo
    create_dataset(CSV_PATH, AUDIO_DIR_ROOT, DATASET_DIR, hop_step=5)

if __name__ == '__main__':
    main()