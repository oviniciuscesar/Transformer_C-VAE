# Overview do Processo de Geração de Targets (tgt)

Este documento descreve a estrutura e o processo de geração de cada amostra de **target (`tgt`)** a partir de uma janela de entrada (**`src`**). O objetivo é detalhar como as 20 _features_ do `tgt` são derivadas, divididas em dois grupos principais: **parâmetros derivados do sinal de origem** e **parâmetros de textura**.

---

## Estrutura Geral

Cada amostra de `tgt` possui **shape (10, 20)**:

- **10 passos no tempo:** todos idênticos — uma repetição de uma mesma linha estática.
- **20 features:** divididas em dois grupos:
  - **14 derivadas do `src`**
  - **6 relacionadas à textura**

---

## 1. Parâmetros Derivados do `src` (14 features)

Esses parâmetros descrevem aspectos espectrais e dinâmicos do áudio analisado. São calculados a partir da **janela de entrada `src`**.

### 1.1 Pitches (Índices 0–6)

- Representam os **7 picos espectrais** com maior energia média.
- Obtidos via:
  1. Cálculo do **espectro de potência médio** da janela.
  2. Seleção dos **7 maiores valores** usando `topk`.
  3. Conversão dos índices **de escala Mel para MIDI cents**.
- Resultado: 7 valores que descrevem as principais componentes tonais do som.

### 1.2 Amplitudes (Índices 7–13)

- Correspondem às **amplitudes** (ou intensidades) das mesmas 7 frequências mais fortes.
- Obtidas via:
  1. Seleção das amplitudes correspondentes aos picos espectrais.
  2. Conversão de dB para **escala MIDI velocity (0–127)**.
- Resultado: 7 valores que representam o perfil dinâmico do espectro.

---

## 2. Parâmetros de Textura (6 features)

Esses parâmetros descrevem características mais globais do som — sua _textura_, densidade e comportamento temporal.  
São definidos a partir de uma combinação entre **regras por classe** e **modulações específicas do áudio**.
Para descrever as texturas foram utilizadas 4 classes:

#### relações verticais

- **textura densa**
- **textura rarefeita**

#### relações horizontais

- **textura dilatada**
- **textura contraída**

### 2.1 Base por Classe

A função `get_process_params_for_label()` define **valores-base** e **ranges (mín./máx.)** para os 6 parâmetros de textura, conforme a **classe do arquivo original** (por exemplo: `flatterzunge`, `sul ponticello`, etc.).

- O **rótulo numérico da classe** é usado como _seed_ aleatória.
- Isso garante **consistência** na geração de valores para cada classe, mantendo variações dentro de faixas coerentes.

### 2.2 Modulações Específicas

Após definir os valores-base, cada parâmetro é ajustado dinamicamente com base em características extraídas da janela `src`:

#### • 4 Metrônomos (Índices 14–17)

- Determinam a **relação entre pulsações harmônicas e inarmônicas**.
- Modulação pela **Entropia Espectral**:
  - **Alta entropia** (sons ruidosos) → metrônomos mais inarmônicos/aleatórios.
  - **Baixa entropia** → relações mais regulares e harmônicas.

#### • 1 Grain Size (Índice 18)

- Representa o **tamanho médio dos grãos sonoros**.
- Modulado pela **duração total do arquivo de áudio**:
  - Arquivos mais longos → grãos maiores.
  - Arquivos curtos → grãos menores e mais densos.

#### • 1 Âmbito (Índice 19)

- Representa a **amplitude espacial ou espectral do som**.
- Modulado pelo **Centróide Espectral (Brilho)**:
  - Sons mais brilhantes → âmbitos mais positivos (expansivos).
  - Sons escuros → âmbitos mais negativos (contraídos).

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
