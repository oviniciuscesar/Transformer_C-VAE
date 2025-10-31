# Relatório de treinamento

## `1_model[ok].pt`

- `Epoch 200/200 | Loss=0.000179 | Recon=0.000179 | KL=0.000000 | Mu=-0.0193 | LogVar=-18.3855 | beta=0.0000`

#### dados usados no treinamento

- N_PITCH = 7 (`min': 6000.0, 'max: 9600.0`)
- N_AMPS = 7 (`min': 0.0, 'max': 127.0`)
- N_TEXTURE_PARAMS = 6 -> 4 metros (`'min': 10, 'max': 8000.0`) + 1 dur (`min': 10.0, 'max': 2500.0`) + 1 âmbito (`'min': -100, 'max': 100`)
- TARGET_FEATURES = N_PITCHES + N_AMPS + N_TEXTURE_PARAMS

#### arquitetura do modelo:

- num_layers_enc=2,
- num_layers_dec=2,
- d_model=64,
- num_heads=2,
- d_ff=128,
- input_features=64 mel-spectrograma
- target_features=20 parâmetros
- latent_dim=32
- max_pos=100
- dropout=0.1

#### hiperparâmetros do treinamento:

- EPOCHS = 200
- BATCH_SIZE = 64
- LR = 1e-3
- CONDITION_DROPOUT_RATE = 0.5
- SEQ_LEN = 10
- MAX_POS = 100 # número máximo de passos temporais
- BETA_START_EPOCH = 0 (beta = 0 constante)
- FREE_BITS_PER_DIM = 0

#### performance do modelo:

- o modelo parece se comportar como previsto, a cada entrada dos mesmo 10 frames de mel-spectrograma ele gera saídas diferentes
- se uso z aleatório a cada entrada, as saídas são variadas
- se uso o mesmo z para todas as entradas, as saídas são muito próximas e pouco variadas (quase idênticas) mesmo para classes diferentes
- alguns parâmetros saem do range que definimos, mas provavelmente tem a ver com a ausência da ativação sigmoid na saída do modelo
