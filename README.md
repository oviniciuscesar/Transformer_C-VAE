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

### Transformer C-VAE:

- **conditional_encoder:** Processa `src → contexto C`.

- **vae_encoder:** Processa `tgt → mu e logvar`.

- **reparameterize:** `(mu, logvar) → z`.

- **decoder:** Recebe `tgt_in, z, C` e gera representações.

- **final_projection:** `d_model → target_features`.

- **forward retorna:** `(predictions, mu, logvar)`.

### 2. Diagrama da Arquitetura (Fluxo de Dados)

O diagrama abaixo ilustra o fluxo de dados (inputs src e tgt) através do modelo para gerar as predições finais, mu e logvar.

```mermaid
graph TD
    src["Input: Sequência Fonte (src)"]
    tgt["Input: Sequência Alvo (tgt)"]
    tgt_in["Input: Alvo Shiftado (tgt_in)"]

    subgraph "1. Encoder Condicional (Gera Contexto C)"
        direction TB
        src --> enc_proj_cond["Projeção de Entrada src"]
        enc_proj_cond --> pe_enc_cond["Add Positional Encoding"]
        pe_enc_cond --> enc_stack_cond["Pilha de EncoderLayers\nSelf-Attention"]
        enc_stack_cond --> C["Contexto C\n(B, L_src, d_model)"]
    end

    subgraph "2. Encoder VAE (Gera mu/logvar)"
        direction TB
        tgt --> enc_proj_vae["Projeção de Entrada tgt"]
        enc_proj_vae --> pe_enc_vae["Add Positional Encoding"]
        pe_enc_vae --> enc_stack_vae["Pilha de EncoderLayers\nSelf-Attention"]
        enc_stack_vae --> pool["Average Pooling Temporal"]
        pool --> fc_mu["Linear -> mu"]
        pool --> fc_logvar["Linear -> logvar"]
    end

    subgraph "3. Amostragem Latente (Reparametrização)"
        direction LR
        fc_mu --> rep["z = reparameterize(mu, logvar)"]
        fc_logvar --> rep
        rep --> z_exp["Expandir z\n(B, L_tgt, latent_dim)"]
    end

    subgraph "4. Decoder (Gera Saída)"
        direction TB
        subgraph "Preparação da Entrada do Decoder"
            direction LR
            tgt_in --> concat["Concat"]
            z_exp --> concat
        end
        concat --> dec_proj["Projeção de Entrada\n(d_model)"]
        dec_proj --> pe_dec["Add Positional Encoding"]
        pe_dec --> dec_stack["Pilha de DecoderLayers\n1. Masked Self-Attention\n2. Cross-Attention"]

        C --> dec_stack

        dec_stack --> dec_out["Saída do Decoder\n(B, L_tgt, d_model)"]
    end

    subgraph "5. Projeção Final"
        direction TB
        dec_out --> final_proj["Linear -> target_features"]
        final_proj --> preds["Predições\n(B, L_tgt, target_features)"]
    end

    subgraph "Saídas do Modelo"
        preds
        mu
        logvar
    end

    classDef inputs fill:#D6EAF8,stroke:#5DADE2,stroke-width:2px;
    class src,tgt,tgt_in inputs;

    classDef outputs fill:#D5F5E3,stroke:#58D68D,stroke-width:2px;
    class preds,mu,logvar outputs;

    classDef context fill:#FCF3CF,stroke:#F7DC6F,stroke-width:2px;
    class C,z_exp context;
```
