import math
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Transformer conditional Variational Autoencoder (C-VAE): seq2seq model
Parâmetros:
- num_layers: número de camadas encoder/decoder
- latent_dim: dimensão do espaço latente
- d_model: dimensão interna do modelo
- num_heads: número de cabeças de atenção
- d_ff: dimensão da camada feed-forward
- input_features: número de features de entrada (ex: 64 para melspectrograma)
- target_features: número de features de saída (ex: 5 para cents, amp, grain_size, metro, duration)
- max_pos: comprimento máximo das sequências (para positional encoding)
- dropout: taxa de dropout
"""

def _build_sinusoidal_position_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """
    Build sinusoidal positional encoding matrix.
    - input:
        - max_len: maximum sequence length
        - d_model: embedding dimension
    - output:
        - pos_encoding: (1, max_len, d_model) positional encoding tensor
    """
    pos_encoding = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
    )
    pos_encoding[:, 0::2] = torch.sin(position * div_term)
    pos_encoding[:, 1::2] = torch.cos(position * div_term)
    return pos_encoding.unsqueeze(0)  # (1, max_len, d_model)


# --- MultiHeadAttention layer ---
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention class
    - input:
        - d_model: embedding dimension
        - num_heads: number of attention heads
        - dropout: dropout rate
    - forward:
        - query: (B, Lq, d_model)
        - key: (B, Lk, d_model)
        - value: (B, Lk, d_model)
        - mask: optional mask (B, 1, Lq, Lk) or (Lq, Lk) or (B, Lk)
    - output:
        - out: (B, Lq, d_model) attention output
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_lin = nn.Linear(d_model, d_model)
        self.k_lin = nn.Linear(d_model, d_model)
        self.v_lin = nn.Linear(d_model, d_model)
        self.out_lin = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, d_model) -> (B, num_heads, seq_len, head_dim)
        B, seq_len, _ = x.size()
        x = x.view(B, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, num_heads, seq_len, head_dim) -> (B, seq_len, d_model)
        x = x.permute(0, 2, 1, 3).contiguous()
        B, seq_len, _, _ = x.size()
        return x.view(B, seq_len, self.d_model)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # query,key,value: (batch, seq_len, d_model)
        q = self.q_lin(query) # query is a target (what I want to search in the input) and it is represented by a sequence 
        k = self.k_lin(key) # key is the label (description) of a query 
        v = self.v_lin(value) # value is the real content information of each query
        # in self-attention query=key=value

        # split into heads
        qh = self._split_heads(q)  # (batch, nh, Lq, hd)
        kh = self._split_heads(k)  # (batch, nh, Lk, hd)
        vh = self._split_heads(v)  # (batch, nh, Lk, hd)

        # scaled dot-product
        scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (batch, nh, Lq, Lk)

        """
            aplica máscara booleana ou aditiva: usada para garantir que o modelo não "veja" posições futuras:
            (ex: impedir que no tempo t+1 o modelo acesse informações do tempo t+2)
        """
        if mask is not None:
            if mask.dtype == torch.bool:
                if mask.dim() == 2:
                    mask_b = mask.unsqueeze(0).unsqueeze(1)        # (1,1,Lq,Lk)
                elif mask.dim() == 3:
                    mask_b = mask.unsqueeze(1)                     # (batch,1,Lq,Lk)
                elif mask.dim() == 4:
                    mask_b = mask                                  # already (batch,1,Lq,Lk)
                else:
                    raise RuntimeError(f"Unexpected boolean mask dim: {mask.dim()}")
                # expand batch dim if necessary
                if mask_b.size(0) != scores.size(0):
                    if mask_b.size(0) == 1:
                        mask_b = mask_b.expand(scores.size(0), -1, -1, -1)
                    else:
                        raise RuntimeError("Mask batch size incompatible with scores")
                scores = scores.masked_fill(~mask_b, float("-1e9"))
            else:
                # float additive mask: broadcast to (batch,1,Lq,Lk)
                if mask.dim() == 2:
                    mask_f = mask.unsqueeze(0).unsqueeze(1)       # (1,1,Lq,Lk)
                elif mask.dim() == 3:
                    mask_f = mask.unsqueeze(1)                    # (batch,1,Lq,Lk)
                elif mask.dim() == 4:
                    mask_f = mask                                  # already (batch,1,Lq,Lk)
                else:
                    raise RuntimeError(f"Unexpected float mask dim: {mask.dim()}")
                if mask_f.size(0) != scores.size(0):
                    if mask_f.size(0) == 1:
                        mask_f = mask_f.expand(scores.size(0), -1, -1, -1)
                    else:
                        raise RuntimeError("Mask batch size incompatible with scores")
                scores = scores + mask_f

        attn = F.softmax(scores, dim=-1) # aplica softmax sobre a dimensão Lk
        attn = self.dropout(attn) # aplica camada dropout no tensor de atenção
        context = torch.matmul(attn, vh) # (batch, nh, Lq, hd)
        concat = self._combine_heads(context)  # concatena o tensor de contexto (batch, Lq, d_model)
        out = self.out_lin(concat) # camada linear final aplicada ao tensor de contexto concatenado (batch, Lq, d_model)
        return out

# --- FeedForward layer ---
class FeedForward(nn.Module):
    """
    Feed-Forward Network (2 linear layers with ReLU and dropout)
    - input:
        - d_model: embedding dimension
        - d_ff: feed-forward hidden dimension
        - dropout: dropout rate
    - forward:
        - x: (B, L, d_model)
    - output:
        - out: (B, L, d_model)"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.lin1 = nn.Linear(d_model, d_ff)
        self.lin2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.dropout(self.activation(self.lin1(x))))

# --- EncoderLayer ---
class EncoderLayer(nn.Module):
    """
    Encoder Layer:
    - Multi-Head Self-Attention + Feed-Forward with residual connections and layer norm
    - input:
        - d_model: embedding dimension
        - num_heads: number of attention heads
        - d_ff: feed-forward hidden dimension
        - dropout: dropout rate
    - forward:
        - x: (B, L, d_model)
        - src_mask: optional source mask
    - output:
        - out: (B, L, d_model)
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # x: (B, L, d_model)
        sa = self.self_attn(x, x, x, mask=src_mask)
        x = x + self.dropout(sa)
        x = self.norm1(x)
        ff = self.ff(x)
        x = x + self.dropout(ff)
        x = self.norm2(x)
        return x

# --- DecoderLayer ---
class DecoderLayer(nn.Module):
    """
    Decoder Layer:
    - Masked Multi-Head Self-Attention + Multi-Head Cross-Attention + Feed-Forward with residual connections, layer norm and dropout
    - input:
        - d_model: embedding dimension
        - num_heads: number of attention heads
        - d_ff: feed-forward hidden dimension
        - dropout: dropout rate
    - forward:
        - x: (B, Lt, d_model)
        - enc_out: (B, Ls, d_model)
        - look_ahead_mask: optional look-ahead mask for self-attention
        - padding_mask: optional padding mask for cross-attention
    - output:
        - out: (B, Lt, d_model)
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm3 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_out: torch.Tensor,
        look_ahead_mask: Optional[torch.Tensor],
        padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # x: (B, Lt, d_model), enc_out: (B, Ls, d_model)
        sa = self.self_attn(x, x, x, mask=look_ahead_mask)
        x = x + self.dropout(sa)
        x = self.norm1(x)

        ca = self.cross_attn(x, enc_out, enc_out, mask=padding_mask)
        x = x + self.dropout(ca)
        x = self.norm2(x)

        ff = self.ff(x)
        x = x + self.dropout(ff)
        x = self.norm3(x)
        return x

# --- Encoder original ---
class Encoder(nn.Module):
    """
    Encoder:
    - input:
        - num_layers: número de camadas
        - d_model: dimensão interna
        - num_heads: número de cabeças de atenção
        - d_ff: dimensão feed-forward
        - input_features: número de features de entrada (ex: 4)
        - max_positions: comprimento máximo da sequência
        - dropout: taxa de dropout
    - forward:
        - src: (B, Ls, input_features) tensor float
        - src_mask: máscara opcional
    - output:
        - out: (B, Ls, d_model)
    """
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, input_features: int, max_positions: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        # linear layer (N features to d_model)
        self.input_projection = nn.Linear(input_features, d_model)
        # talvez adicionar mais uma camada linear?
        pos_encoding = _build_sinusoidal_position_encoding(max_positions, d_model)
        self.register_buffer("pos_encoding", pos_encoding)  # (1, max_pos, d_model)
        layers: List[nn.Module] = []
        for _ in range(num_layers):
            layers.append(EncoderLayer(d_model, num_heads, d_ff, dropout))
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # src: (B, Ls, input_features) - float tensor
        
        x = self.input_projection(src) * math.sqrt(self.d_model)
        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x  # (B, Ls, d_model)

class VAEEncoder(nn.Module):
    """
    Codificador VAE condicional:
    - Mapeia (tgt, C) para (mu, logvar)
    - C é o contexto derivado de src
    - tgt é a sequência alvo
    """
    def __init__(
        self, 
        num_layers: int, 
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        target_features: int, 
        latent_dim: int, 
        max_positions: int, 
        dropout: float = 0.1
    ) -> None:
        super().__init__()

        # Projeção inicial para alinhar dimensões após concatenação [tgt + C]
        self.input_proj = nn.Linear(target_features + d_model, d_model)

        # Encoder Transformer
        self.encoder_base = Encoder(
            num_layers, d_model, num_heads, d_ff,
            d_model, max_positions, dropout
        )

        # Projeções para mu e logvar
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_logvar = nn.Linear(d_model, latent_dim)

    def forward(self, tgt: torch.Tensor, C: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # C: (B, Ls, d_model) → média no tempo para obter contexto fixo
        C_pooled = C.mean(dim=1)  # (B, d_model)

        # Expande o contexto para cada passo de tempo da sequência alvo
        C_expanded = C_pooled.unsqueeze(1).repeat(1, tgt.size(1), 1)

        # Concatena tgt e contexto condicional
        x = torch.cat([tgt, C_expanded], dim=-1)  # (B, L, target_features + d_model)

        # Projeta para o espaço d_model
        x = self.input_proj(x)  # (B, L, d_model)

        # Passa pelo encoder
        x = self.encoder_base(x, tgt_mask)  # (B, L, d_model)

        # Pooling sobre a dimensão temporal
        x = x.mean(dim=1)  # (B, d_model)

        # Projeções finais
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


# --- Decoder modificado ---
class Decoder(nn.Module):
    """
    Decoder: (Modificado para aceitar 'z')
    - forward:
        - tgt: (batch, Lt, target_features) tensor float (entrada do decoder)
        - z: (batch, latent_dim) tensor float (vetor latente)
        - enc_out: (batch, Ls, d_model) saída do encoder condicional
    - output:
        - out: (batch, Lt, d_model)
    """
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, 
                 target_features: int, latent_dim: int, max_positions: int, 
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        
        # Projeta (features + z) para d_model
        self.target_projection = nn.Linear(target_features + latent_dim, d_model)
        
        pos_encoding = _build_sinusoidal_position_encoding(max_positions, d_model)
        self.register_buffer("pos_encoding", pos_encoding)
        layers: List[nn.Module] = []
        for _ in range(num_layers):
            layers.append(DecoderLayer(d_model, num_heads, d_ff, dropout))
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt: torch.Tensor, z: torch.Tensor, enc_out: torch.Tensor, look_ahead_mask: Optional[torch.Tensor], padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # tgt: (batch, Lt, target_features)
        # z: (batch, latent_dim)
        batch, Lt, _ = tgt.shape

        # Expande z e concatena com tgt (target features)
        z_expanded = z.unsqueeze(1).expand(batch, Lt, -1) # (batch, Lt, latent_dim)
        combined_input = torch.cat([tgt, z_expanded], dim=-1) # (batch, Lt, target_features + latent_dim)

        # Projeta a entrada combinada
        x = self.target_projection(combined_input) * math.sqrt(self.d_model)
        
        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, enc_out, look_ahead_mask, padding_mask)
        return x  # (batch, Lt, d_model)
    

# --- Prior ----
class Prior(nn.Module):
    def __init__(self, d_model, latent_dim, prior_d_ff=128, dropout=0.1):
        super().__init__()

        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, prior_d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(prior_d_ff, d_model),
        )

        self.mu = nn.Linear(d_model, latent_dim)
        self.logvar = nn.Linear(d_model, latent_dim)

        # inicialização dos pesos do prior d
        nn.init.xavier_uniform_(self.mu.weight, gain=0.1)
        nn.init.zeros_(self.mu.bias)
        nn.init.xavier_uniform_(self.logvar.weight, gain=0.1)
        nn.init.constant_(self.logvar.bias, -3.0)

    def forward(self, context): 
        """
        context: (B, T, d_model)
        """
        x = self.norm(context)
        x = x + self.ff(x)               # pequeno resblock
        x = x.mean(dim=1)                # pooling temporal
        return self.mu(x), self.logvar(x)

# class Prior(nn.Module):
#     def __init__(self, d_model, latent_dim, num_heads=2, prior_d_ff=128, dropout=0.05):
#         super().__init__()

#         self.attn = MultiHeadAttention(d_model, num_heads, dropout)
#         self.norm1 = nn.LayerNorm(d_model)

#         self.ff = FeedForward(d_model, prior_d_ff, dropout)
#         self.norm2 = nn.LayerNorm(d_model)

#         self.mu = nn.Linear(d_model, latent_dim)
#         self.logvar = nn.Linear(d_model, latent_dim)

#     def forward(self, context):
#         x = context

#         attn_out = self.attn(x, x, x)
#         x = self.norm1(x + attn_out)

#         ff_out = self.ff(x)
#         x = self.norm2(x + ff_out)

#         pooled = x.mean(dim=1)
#         return self.mu(pooled), self.logvar(pooled)
 

# --- Transformer C-VAE Model ---
class TransformerCVAE(nn.Module):
    """
    Transformer C-VAE (Conditional Variational Autoencoder)
    Combina os três componentes para o treinamento.
    - input:
        - num_layers_enc: número de camadas do encoder
        - num_layers_dec: número de camadas do decoder
        - d_model: dimensão interna
        - num_heads: número de cabeças de atenção
        - d_ff: dimensão feed-forward
        - input_features: número de features de entrada (ex: 4)
        - target_features: número de features de saída (ex: 5)
        - latent_dim: dimensão do espaço latente
        - max_pos: comprimento máximo da sequência
        - dropout: taxa de dropout
    - forward:
        - src: (batch, Ls, input_features) tensor float (sequência condicional)
        - tgt: (batch, Lt, target_features) tensor float (sequência alvo completa)
    - output:
        - predictions: (batch, Lt-1, target_features) tensor float (saída do decoder)
        - mu: (batch, latent_dim) média do espaço latente
        - logvar: (batch, latent_dim) log-variância do espaço latente
    """
    def __init__(
        self,
        num_layers_enc: int, # N layers para encoders
        num_layers_vae: int, # N layers para encoder VAE
        num_layers_dec: int, # N layers para decoder
        d_model: int, # dimensão interna do modelo
        num_heads: int, # número de cabeças de atenção no encoder
        decoder_num_heads: int, # número de cabeças de atenção do decoder
        encoder_d_ff: int, # dimensão da camada feed-forward
        vaencoder_d_ff: int, # dimensão da camada feed-forward do encoder VAE
        decoder_d_ff: int, # dimensão da camada feed-forward do decoder
        input_features: int,  # features da flauta
        target_features: int, # features de saída (parâmetros da eletrônica)
        latent_dim: int,      # Dimensão do espaço latente z
        max_pos: int,       # comprimento máximo das sequências geradas
        encoder_dropout: float = 0.1, # taxa de dropout
        vae_dropout: float = 0.1, # taxa de dropout
        decoder_dropout: float = 0.1, # taxa de dropout
        final_proj_dropout: float = 0.1, # taxa de dropout na camada final
    ) -> None:
        super().__init__()
        
        self.latent_dim = latent_dim
        self.target_features = target_features

        # 1. Encoder Condicional (features de entrada -> Contexto (representação da entrada))
        self.conditional_encoder = Encoder(
            num_layers_enc, d_model, num_heads, encoder_d_ff, 
            input_features, max_pos, encoder_dropout
        )

        #2. Encoder VAE (features de saída -> Espaço Latente)
        self.vae_encoder = VAEEncoder(
            num_layers_vae, d_model, num_heads, vaencoder_d_ff, 
            target_features, latent_dim, max_pos, vae_dropout
        )
        
        # prior treinável p(z|C) ~ N(prior_mu(C), prior_logvar(C)) para uso na inferência
        # self.prior_mu = nn.Linear(d_model, latent_dim)
        # self.prior_logvar = nn.Linear(d_model, latent_dim)
        # prior mais equilibrado
        self.prior = Prior(d_model, latent_dim, prior_d_ff=vaencoder_d_ff, dropout=encoder_dropout)

        # self.fc_mu = nn.Linear(d_model, latent_dim)
        # self.fc_logvar = nn.Linear(d_model, latent_dim)

        # 3. Decoder ((tgt_in, z, C) -> Predição) tgt_in: target features de entrada, z: vetor latente, C: contexto das features de entrada
        self.decoder = Decoder(
            num_layers_dec, d_model, decoder_num_heads, decoder_d_ff, 
            target_features, latent_dim, max_pos, decoder_dropout
        )

        # 4. Camada final (camada linear para projetar d_model -> target_features)
        # self.final_projection = nn.Linear(d_model, target_features)
        self.final_projection = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, target_features),
            # nn.GELU(),
            # nn.Dropout(final_proj_dropout),
            # nn.Linear(d_model//2, target_features)
        )

        # Inicialização suave para não saturar o Tanh
        for m in self.final_projection:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Reduz ligeiramente o ganho da última camada
        if isinstance(self.final_projection[-1], nn.Linear):
            nn.init.xavier_uniform_(self.final_projection[-1].weight, gain=0.5)

        # 5. ativação tanh na saída para garantir que os valores estejam entre -1 e 1
        self.output_activation = nn.Tanh()

    # Cria máscara causal (look-ahead)
    def _create_look_ahead_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """Cria uma máscara causal (look-ahead) booleana."""
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        return mask == 0 # (True onde pode olhar, False onde está mascarado)

    # Truque de Reparametrização: Amostragem do espaço latente
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Truque de Reparametrização VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # forward do modelo completo para treinamento (multihead attention + encoder VAE + decoder)
    def forward(
        self, 
        src: torch.Tensor, # (batch, Ls, input_features)
        tgt: torch.Tensor, # (batch, Lt, target_features)
        src_mask: Optional[torch.Tensor] = None, # máscara opcional para src
        tgt_padding_mask: Optional[torch.Tensor] = None, # máscara opcional para tgt
        memory_mask: Optional[torch.Tensor] = None # máscara opcional para memória
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: # (posterior_mu, posterior_logvar, prior_mu, prior_logvar, z, predictions)
        """
        Fluxo de treinamento do C-VAE.
        entradas:
        - 1 src: (batch, Ls, input_features) - features da flauta
        - 2 tgt: (batch, Lt, target_features) - features alvo (Sequência *completa*)
        Retorna:
        - predictions: (batch, Lt-1, target_features) - Saída do decoder
        - mu: (batch, latent_dim) - Média do espaço latente
        - logvar: (batch, latent_dim) - Log-variância do espaço latente
        """

        # 1. features de entrada -> contexto (representação da entrada)
        # (batch, Ls, input_features) -> (batch, Ls, d_model)
        C = self.conditional_encoder(src, src_mask) 

    
        #2. features alvo -> Espaço Latente (SOMENTE durante o treinamento)
        #(batch, Lt, target_features) -> (batch, latent_dim), (batch, latent_dim)
        posterior_mu, posterior_logvar = self.vae_encoder(tgt, C, tgt_padding_mask)

        #2.1 prior p(z | C)  <- treinável a partir do contexto C
        # C_pooled = C.mean(dim=1)  # (B, d_model)
        # prior_mu = self.prior_mu(C_pooled)          # (B, latent_dim)
        # prior_logvar = self.prior_logvar(C_pooled)  # (B, latent_dim)
        prior_mu, prior_logvar = self.prior(C)
        prior_logvar = prior_logvar.clamp(-12., 5.)

        # # clamp para evitar valores extremos
        # posterior_logvar = posterior_logvar.clamp(-12., 5.)
        # prior_logvar = prior_logvar.clamp(-12., 5.)

        # 3. Amostragem do espaço latente
        z = self.reparameterize(posterior_mu, posterior_logvar) # (batch, latent_dim)

        # Cria o token SOS (zeros) com mesmo tipo e device do tgt (para compatibilidade com o wrapper)
        sos_token = torch.zeros(tgt.size(0), 1, tgt.size(2), device=tgt.device)

        # 4. Preparação da entrada do Decoder (Teacher Forcing): remove o último passo de tempo de tgt para o modelo prever o próximo passo
        # (batch, Lt, target_features) -> (batch, Lt-1, target_features)
        # tgt_in = tgt[:, :-1, :]
        
        #  Concatena SOS no início e remove o último frame do original
        # Entrada: [SOS, F0, F1, ..., F8] (Total 10 frames)
        tgt_in = torch.cat([sos_token, tgt[:, :-1, :]], dim=1)
        
        # Cria máscara causal para o decoder
        look_ahead_mask = self._create_look_ahead_mask(tgt_in.size(1), tgt.device)

        # 5. Geração pelo Decoder
        # (features alvo, z, contexto) -> (batch, Lt-1, d_model)
        dec_out = self.decoder(tgt_in, z, C, look_ahead_mask, memory_mask)
        
        # 6. Projeção final (camada linear)
        # (batch, Lt-1, d_model) -> (batch, Lt-1, target_features)
        predictions = self.final_projection(dec_out)
        
        # 7. Aplica ativação final (tanh)
        predictions = self.output_activation(predictions)

        # retorna as previsões, média e log-variância do espaço latente
        return posterior_mu, posterior_logvar, prior_mu, prior_logvar, z, predictions


if __name__ == "__main__":
    # Teste rápido para a arquitetura transformer C-VAE
    num_layers = 2
    num_layers_enc = 2
    num_layers_vae = 2
    num_layers_dec = 2
    d_model = 64
    num_heads = 4
    decoder_num_heads = 2
    encoder_d_ff = 128
    decoder_d_ff = 128
    vaencoder_d_ff = 128

    # Parâmetros
    input_features = 64  # features de entrada (melspectrograma, centroid, inarmonicidade, etc.)
    target_features = 20   # features alvo (cents, amp, grain_size, metro, duration, etc)
    latent_dim = 32      # Dimensão do espaço latente
    max_pos = 100        # número máximo de passos previstos (para embeddings posicional)
    seq_len = 10         # número de passos de tempo (frames/events)

    # cria o modelo transformer C-VAE
    model = TransformerCVAE(
        num_layers_enc=num_layers,
        num_layers_vae=num_layers,
        num_layers_dec=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        decoder_num_heads=decoder_num_heads,
        encoder_d_ff=encoder_d_ff,
        decoder_d_ff=decoder_d_ff,
        vaencoder_d_ff=encoder_d_ff,
        input_features=input_features,
        target_features=target_features,
        latent_dim=latent_dim,
        max_pos=max_pos,
    )
    model.eval()

    # inputs dummy (float) para teste rápido
    # Ls=10 (10 frames de entrada)
    # Lt=10 (10 eventos de saída)
    src = torch.randn(1, seq_len, input_features)   # (batch, Ls, 64)
    tgt = torch.randn(1, seq_len, target_features)  # (batch, Lt, 5)

    # O modelo agora retorna 3 tensores: (predictions, mu, logvar)
    posterior_mu, posterior_logvar, prior_mu, prior_logvar, z, predictions = model(src, tgt)
    
    print("--- Teste de Arquitetura Transformer C-VAE ---")
    print(f"Shape da entrada (src): {src.shape}")
    print(f"Shape do alvo (tgt): {tgt.shape}")
    
    print("\n--- Shapes da Saída ---")
    print(f"Shape das Predições: {predictions.shape}")
    print(f"Shape de Mu (latente): {posterior_mu.shape}")
    print(f"Shape de LogVar (latente): {posterior_logvar.shape}")
    
    # A saída de predição deve ser (batch, Lt-1, target_features)
    assert predictions.shape == (1, seq_len - 1, target_features)
    assert posterior_mu.shape == (1, latent_dim) # shape de mu deve ser (batch, latent_dim)
    assert posterior_logvar.shape == (1, latent_dim) # shape de logvar deve ser (batch, latent_dim)
    
    print("\nTeste de shape passou!")
    
    # Teste TorchScript (pode falhar devido a lógicas mais complexas)
    try:
        scripted = torch.jit.script(model)
        out_scripted, _, _ = scripted(src, tgt)
        assert out_scripted.shape == predictions.shape
        print("Teste de TorchScript passou!")
    except Exception as e:
        print(f"\nTeste de TorchScript falhou (esperado para VAEs complexos): {e}")