# Overview da criação dos targets (tgt)

Este documento descreve a estrutura e o processo de geração de cada amostra de **target (`tgt`)** a partir de uma janela de entrada (**`src`**). O objetivo é detalhar como as 20 _features_ do `tgt` são derivadas, divididas em dois grupos principais: **parâmetros baseados em perfis de classe** e **parâmetros de textura**.

---

## Estrutura Geral

Cada amostra de `tgt` possui **shape [10, 20]**:

- **10 passos no tempo:** com **variação temporal frame-a-frame** — cada frame possui valores específicos calculados individualmente.
- **20 features normalizadas em [-1, 1]:** divididas em dois grupos:
  - **14 baseadas em perfis de classe e características espectrais**
  - **6 relacionadas à textura com evolução temporal**

---

## 1. Parâmetros Baseados em Classe (14 features)

Esses parâmetros descrevem aspectos tonais e dinâmicos do som. São **gerados a partir de perfis pré-definidos por classe técnica**, modulados por características espectrais extraídas de cada frame individual do `src`.

### 1.1 Pitches (Índices 0–6)

- Representam **7 alturas em MIDI cents** características da técnica instrumental.
- **Método de geração** (`generate_pitches_amps_from_class()`):
  1. Cada **classe técnica** possui um perfil base com `base_pitch` e `pitch_range` específicos.
  2. Para cada frame, calcula-se o **brightness factor** a partir do centróide espectral daquele frame.
  3. Os pitches são gerados como: `base_pitch + brightness_factor * pitch_range + noise`
  4. Aplicado ordenação crescente e clamping no range válido (6000-9600 cents).
- **Normalização**: Z-score com média 7800 cents e desvio padrão 1200 cents.
- **Resultado**: 7 valores normalizados que descrevem o perfil tonal característico da classe, ajustado ao brilho espectral de cada frame.

### 1.2 Amplitudes (Índices 7–13)

- Correspondem às **7 intensidades em escala MIDI velocity (0-127)**.
- **Método de geração**:
  1. Cada classe possui `base_amp` e `amp_range` específicos em seu perfil.
  2. Modulados pelo **duration factor** (duração total do áudio normalizada).
  3. Gerados como: `base_amp + duration_factor * amp_range + noise`
  4. Aplicado clamping no range válido (0-127).
- **Normalização**: Escala logarítmica mapeada para [-1, 1].
- **Resultado**: 7 valores normalizados que representam o envelope dinâmico característico da classe.

---

## 2. Parâmetros de Textura (6 features)

Esses parâmetros descrevem características mais globais do som — sua _textura_, densidade e comportamento temporal.  
São definidos a partir de uma combinação entre **ranges específicos por classe técnica** e **modulações temporais frame-a-frame**.

Para descrever as texturas foram utilizadas 4 dimensões principais:

#### Relações Verticais (Densidade Espectral)

- **Textura densa**: muitos componentes espectrais simultâneos
- **Textura rarefeita**: poucos componentes espectrais

#### Relações Horizontais (Evolução Temporal)

- **Textura dilatada**: evolução lenta, grãos longos
- **Textura contraída**: evolução rápida, grãos curtos

#### Características Adicionais

- **Impulso**: ataque rápido e decay abrupto
- **Ruidoso**: alta entropia espectral
- **Alure**: texturas com brilho sustentado
- **Evolutivo**: transformações graduais ao longo do tempo

### 2.1 Definição Heurística por Classe

A função `get_process_params_for_label()` define **ranges específicos** para os 6 parâmetros de textura baseados na **classe técnica** (ex: `flatterzunge`, `ordinario`, `jet_whistle`, etc.).

**Exemplos de mapeamento**:

- **Aeolian**: rarefeita + contraída + ruidosa → metros curtos (135-453ms), grain pequeno (50-123ms)
- **Crescendo**: rarefeita + dilatada + evolutiva → metros médios (443-1230ms), grain pequeno (50-75ms)
- **Jet Whistle**: densa + dilatada + ruidosa → metros médios (347-1575ms), grain grande (50-200ms)
- **Ordinario**: rarefeita + contraída + alure → metros longos (847-2657ms), grain médio (50-95ms)
- **Staccato**: densa + dilatada + impulso → metros curtos (200-646ms), grain grande (50-243ms)

### 2.2 Variação Temporal Frame-a-Frame

A função `get_temporal_texture_params()` adiciona **variação temporal natural** aos parâmetros base:

1. **Seed por frame**: cada frame usa `seed + frame_position` para garantir variação consistente mas única
2. **Noise scaling progressivo**: `noise_scale = 0.05 + 0.1 * (frame_position / 10.0)` — aumenta ao longo do tempo
3. **Aplicação seletiva**:
   - **Metrônomos**: variação proporcional ao valor base (±5-10%)
   - **Grain**: variação reduzida (±1%)
   - **Âmbito**: variação absoluta (±10 unidades)

### 2.3 Modulações Específicas do Áudio

Cada parâmetro é ajustado dinamicamente com base em características extraídas **de cada frame individual**:

#### • 4 Metrônomos (Índices 14–17)

- | Determinam **tempos de pulsação em milissegundos** para síntese g | Geração | Normalização                    |
  | :---------------------------------------------------------------- | :-----: | :------------------------------ | :---------------------------------------- | :----------------------- |
  | Pitches                                                           |   0–6   | 7 alturas em MIDI cents         | Perfil de classe + brightness (por frame) | Z-score (μ=7800, σ=1200) |
  | Amplitudes                                                        |  7–13   | 7 intensidades em MIDI velocity | Perfil de classe + duration factor        | Log scale → [-1, 1]      |
  | Metrônomos                                                        |  14–17  | Tempos de pulsação (ms)         | Range de classe + entropia (por frame)    | Log scale → [-1, 1]      |
  | Grain Size                                                        |   18    | Tamanho de grãos sonoros (ms)   | Range de classe + duration factor         | Log scale → [-1, 1]      |
  | Âmbito                                                            |   19    | Abertura espacial/espectral     | Range de classe + brightness (por frame)  | Log scale → [-1, 1]      |

### Ranges de Normalização (TCVAEdataset.py)

```python
NORMALIZATION_RANGES = {
    'pitch': {'min': 6000.0, 'max': 9600.0},    # 3 oitavas
    'amp': {'min': 0.0, 'max': 127.0},           # MIDI velocity
    'metro': {'min': 100.0, 'max': 4083.56},     # milissegundos
    'grain': {'min': 50.0, 'max': 243.0},        # milissegundos
    'ambito': {'min': 10.0, 'max': 96.18}        # unidades arbitrárias
}
```

---

## Pipeline de Geração (calculate_temporal_targets)

Para cada um dos **10 frames temporais**:

1. **Extração de características espectrais do frame**:

   - Potência espectral: `melspec_power[:, frame_idx]`
   - Entropia espectral: `calculate_spectral_entropy(frame_power)`
   - Centróide espectral: `calculate_spectral_centroid(frame_power, MEL_FREQS_HZ)`
   - Curtose espectral: `calculate_spec_kurtosis(frame_power, MEL_FREQS_HZ)`

2. **Cálculo do brightness factor**: `(centroid - fmin) / (fmax - fmin)` → [0, 1]

3. **Geração de pitches/amplitudes** (`generate_pitches_amps_from_class`):

   - Seed: `seed_global + frame_idx`
   - Entrada: classe técnica, brightness do frame, duration global
   - Saída: 7 pitches + 7 amplitudes específicos deste frame

4. **Geração de parâmetros de textura** (`get_temporal_texture_params`):

   - Base: `get_process_params_for_label()` com ranges da classe
   - Variação temporal: random walk com noise crescente ao longo dos frames
   - Entrada: seed por frame, entropia/brightness do frame, position temporal
   - Saída: 4 metros + 1 grain + 1 âmbito com evolução temporal

5. **Normalização**: Todos os 20 valores são normalizados para [-1, 1]

6. **Empilhamento**: 10 frames formam tensor `[10, 20]`

---

## Conclusão

## O processo implementado integra:

## Arquitetura do Modelo e Inferência

### Treinamento

1. **Conditional Encoder**: Recebe `src` (10 frames × 80 mel-bins) → gera contexto `C`
2. **Variational Encoder**: Recebe `C` + `tgt_in` → retorna espaço latente `(mu, logvar)`
3. **Reparametrização**: `Z = mu + eps * exp(0.5 * logvar)`
4. **Teacher Forcing**: Descarta-se o último passo de `tgt_in` para shift temporal
5. **Decoder**: Recebe `tgt_in[:-1]` + `Z` + `C` → gera predições autoregressivas
6. **Projeção**: Camada linear + ativação `tanh` → saída normalizada em [-1, 1]

### Problema na Inferência

**Na inferência em tempo real não temos `tgt_in` (features), somente `src`.**

### Soluções Possíveis

#### 1. **Zero Initialization** (Mais Simples)

```python
tgt_in = torch.zeros((batch, seq_len, 20), device=device)
```

- Inicia com vetor nulo e deixa o modelo gerar autoregressivamente
- Funciona se o modelo aprendeu a depender mais de `C` e `Z` que de `tgt_in`

#### 2. **Class-Based Initialization** (Implementado)

```python
# Usa centróides do espaço latente por classe
centroids = load_centroids_dict("latent_centroids.txt")
z_vector = centroids[class_name]  # Vetor latente da classe
model.latent(z_vector)
out = model.forwardz(mel_tensor)
```

- Força o vetor latente `Z` a representar a classe desejada
- O decoder usa `Z` + `C` para gerar características coerentes com a classe
- Permite síntese condicional: áudio de classe A com vetor latente de classe B

#### 3. **Heuristic Initialization** (Proposta)

```python
# Gera tgt_in aproximado usando as mesmas funções do dataset
brightness = calculate_spectral_centroid(src_power, MEL_FREQS_HZ)
entropy = calculate_spectral_entropy(src_power)
pitches, amps = generate_pitches_amps_from_class(
    seed=class_label,
    class_name=class_name,
    brightness_factor=brightness,
    duration_factor=0.5
)
texture = get_process_params_for_label(
    seed=class_label,
    folder_name=class_name,
    duration_factor=0.5,
    entropy_factor=entropy,
    brightness_factor=brightness
)
tgt_in = normalize_and_concat([pitches, amps, texture])
```

- Reconstrói `tgt_in` usando as mesmas heurísticas do dataset
- Garante coerência com o processo de treinamento
- Requer conhecimento da classe e acesso às funções do dataset

#### 4. **Autoregressive Generation** (Ideal)

```python
# Frame 0: inicia com zeros ou heurística
pred_0 = model.forward_step(tgt_in[0], Z, C)
# Frame 1: usa predição anterior
pred_1 = model.forward_step(pred_0, Z, C)
# ... continua autoregressivamente
```

- Geração passo-a-passo usando predições anteriores como entrada
- Requer que o decoder suporte inferência autoregressiva
- Mais lento mas potencialmente mais expressivo

### Recomendação

Para inferência em tempo real, a **abordagem híbrida** (2 + 3) é mais robusta:

1. Definir `Z` usando centróides de classe (controle de identidade técnica)
2. Inicializar `tgt_in` com heurísticas simples baseadas em `src` (coerência espectral)
3. Deixar o decoder refinar autoregressivamente se necessário

Isso garante que o modelo tenha informação suficiente para gerar saídas coerentes mesmo sem ground truth de `tgt_in`.

- **Síntese condicional por classe**: O modelo pode aprender mapeamentos específicos por técnica
- **Geração com variação temporal**: Cada frame pode ter características levemente diferentes
- **Modelagem de timbre evolutivo**: Suporta transformações graduais (crescendo, transitions, etc.)
- **Geração** (`_generate_grain_ambito()`):
  1. Base no ponto médio do range da classe
  2. Deslocamento determinístico pelo **duration factor** global: `±0.5 * range`
  3. Adição de ruído gaussiano proporcional à variabilidade
- **Normalização**: Escala logarítmica mapeada para [-1, 1].
- **Modulação**:
  - Arquivos mais longos → grãos maiores (textura dilatada)
  - Arquivos curtos → grãos menores (textura contraída)

#### • 1 Âmbito (Índice 19)

- Representa a **abertura espacial/espectral** do som.
- **Geração**:
  1. Base no ponto médio do range da classe
  2. Deslocamento determinístico pelo **brightness factor do frame**: `±0.5 * range`
  3. Adição de ruído gaussiano proporcional à variabilidade
- **Normalização**: Escala logarítmica mapeada para [-1, 1].
- **Modulação por frame**:
  - **Centróide alto** (frame brilhante) → âmbito positivo/expansivo
  - **Centróide baixo** (frame escuro) → âmbito negativo/contraído

---

## Resumo da Estrutura das Features

| Grupo      | Faixa de Índices | Descrição                        | Modulação Principal        |
| :--------- | :--------------: | :------------------------------- | :------------------------- |
| Pitches    |       0–6        | 7 picos espectrais em MIDI cents | Espectro de potência médio |
| Amplitudes |       7–13       | 7 amplitudes em MIDI velocity    | Normalização dB → MIDI     |
| Metrônomos |      14–17       | Relações harmônicas/inarmônicas  | Entropia espectral         |
| Grain Size |        18        | Tamanho de grãos sonoros         | Duração total do áudio     |
| Âmbito     |        19        | Abertura espectral               | Centróide espectral        |

---

## Conclusão

O processo integra **características espectrais**, **dinâmicas** e **de textura** em uma representação padronizada (`tgt`) capaz de refletir tanto a **identidade sonora da classe** quanto as **particularidades do som analisado**.  
Essa estrutura é essencial para tarefas de **síntese condicional**, **geração sonora guiada por aprendizado de máquina** ou **modelagem de timbre**.

No treinamento:
1 - O modelo recebe src (10 frames de 64 mel-spectrogramas) passo pelo conditional encoder para gerar o contexto C
2 - O variational encoder recebe o contexto C e tgt_in e retrona o espaço latente(mu, logvar)
3 - Mu, logvar são reparametrizados retornando Z
4 - Descarta-se o último passo de tgt_in (teacher forcing)
5 - O decoder recebe tgt_in (sem o último passo), Z e C,
6 - A saída do decoder é projetada por uma camada linear e em seguida passa pela ativação tanh

O problema é que na inferência em tempo real não temos tgt_in (features), somente src. Como lidar corretamente com essa situação na inferência com o wrapper?
