# Transformer Conditional Variational Autoencoder (T-CVAE)

### 1. Resumo Técnico

- **Positional encoding:** `_build_sinusoidal_position_encoding(max_len, d_model)`.

- **MultiHeadAttention:** Projeções lineares `(q/k/v/out)`, `split/combine heads`, produto escalar escalarizado + softmax + dropout.

- **FeedForward:** `lin1 -> ReLU -> dropout -> lin2`.

- **EncoderLayer:** `Self-attention + dropout + LayerNorm, seguido por FeedForward + dropout + LayerNorm`.

- **Encoder (conditional_encoder):** Projeção de entrada `(input_features → d_model)`, adição de positional encoding, pilha de EncoderLayer. Retorna contexto condicional C.

- **VAEEncoder:** Usa um Encoder para processar a sequência alvo `(target_features → d_model)`. Faz pooling (média) sobre a dimensão temporal. Projeta para `mu` e `logvar`.

- **DecoderLayer:** Masked `self-attention` (causal) + cross-attention (sobre `enc_out/C`) + `FeedForward` (com `LayerNorms` e `dropout`).

- **Decoder:** Concatena `z` (latente) expandido com `tgt`. Projeta (`target_features` + `latent_dim`) → `d_model`. Adiciona positional encoding. Pilha de `DecoderLayer`.å

### 2. Transformer C-VAE:

- **1. Encoder Condicional:** Processa `entrada → contexto C`.
  <img src= "img/encoderc.png" width="300" alt="Arquitetura do Decoder Condicional" />

- **2. Encoder Variacional:** Processa `tgt e C → mu e logvar`.
  <img src= "img/encoderv.png" width="500" alt="Arquitetura do Decoder Condicional" />

- **Reparametrização:** `(mu, logvar) → z`.

- **3. Decoder Generativo:** Recebe `tgt_in, z, C` e gera representações.
  <img src= "img/decoderg.png" width="400" alt="Arquitetura do Decoder Condicional" />
