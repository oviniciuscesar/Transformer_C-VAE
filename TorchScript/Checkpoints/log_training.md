# Relatório de treinamento

## `8_model[ok].pt`

- `Epoch 100/100 | Loss=0.003078 | Recon=0.000305 | KL=0.002773 | KL/Recon ratio=9.0776 | Mu=-0.0005 | LogVar=-0.0054 | beta=0.2000`

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

- EPOCHS = 100
- BATCH_SIZE = 64
- LR = 5e-4
- CONDITION_DROPOUT_RATE = 0.1
- SEQ_LEN = 10
- BETA_START_EPOCH = 20
- BETA_WARMUP_EPOCHS = 50
- BETA_MAX = 0.2
- FREE_BITS_PER_DIM = 0.02

#### performance do modelo:

- o modelo parece se comportar como previsto, a cada entrada dos mesmo 10 frames de mel-spectrograma ele gera saídas diferentes
- se uso z aleatório a cada entrada, as saídas são variadas
- se uso o mesmo z para todas as entradas as saídas são diferentes
