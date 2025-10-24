import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Transformer Model implementation in PyTorch
Based on "Attention is All You Need" (Vaswani et al., 2017)
### ALTERADO ###
Versão para Regressão Seq2Seq (vetor-para-vetor).

Parâmetros:
- num_layers: número de camadas encoder/decoder
- d_model: dimensão interna do modelo
- num_heads: número de cabeças de atenção
- d_ff: dimensão da camada feed-forward
- input_features: número de features de entrada (ex: 4 para inarmonicidade, etc.)
- target_features: número de features de saída (ex: 5 para cents, amp, etc.)
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

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # query,key,value: (B, seq_len, d_model)
        q = self.q_lin(query)
        k = self.k_lin(key)
        v = self.v_lin(value)

        qh = self._split_heads(q)  # (B, nh, Lq, hd)
        kh = self._split_heads(k)  # (B, nh, Lk, hd)
        vh = self._split_heads(v)  # (B, nh, Lk, hd)

        # scaled dot-product
        scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, nh, Lq, Lk)

        if mask is not None:
            # mask expected to be either (B, 1, Lq, Lk) or (Lq, Lk) or (B, Lk) boolean/float
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(~mask.unsqueeze(1), float("-1e9"))
            else:
                # assume float additive mask where masked positions are -inf or large negative
                if mask.dim() == 2:
                    scores = scores + mask.unsqueeze(0).unsqueeze(1)  # (1,1,Lq,Lk)
                elif mask.dim() == 3:
                    scores = scores + mask.unsqueeze(1)  # (B,1,Lq,Lk)
                else:
                    scores = scores + mask

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, vh)  # (B, nh, Lq, hd)
        concat = self._combine_heads(context)  # (B, Lq, d_model)
        out = self.out_lin(concat)
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


class Decoder(nn.Module):
    """
    Decoder:
    - input:
        - num_layers: número de camadas
        - d_model: dimensão interna
        - num_heads: número de cabeças de atenção
        - d_ff: dimensão feed-forward
        - target_features: número de features de saída (ex: 5)
        - max_positions: comprimento máximo da sequência
        - dropout: taxa de dropout
    - forward:
        - tgt: (B, Lt, target_features) tensor float
        - enc_out: (B, Ls, d_model) saída do encoder
        - look_ahead_mask: máscara causal
        - padding_mask: máscara de padding (para cross-attention)
    - output:
        - out: (B, Lt, d_model)
    """
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, target_features: int, max_positions: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        # linear layer (N features to d_model)
        self.target_projection = nn.Linear(target_features, d_model)
        pos_encoding = _build_sinusoidal_position_encoding(max_positions, d_model)
        self.register_buffer("pos_encoding", pos_encoding)
        layers: List[nn.Module] = []
        for _ in range(num_layers):
            layers.append(DecoderLayer(d_model, num_heads, d_ff, dropout))
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt: torch.Tensor, enc_out: torch.Tensor, look_ahead_mask: Optional[torch.Tensor], padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # tgt: (B, Lt, target_features) - float tensor
        x = self.target_projection(tgt) * math.sqrt(self.d_model)
        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, enc_out, look_ahead_mask, padding_mask)
        return x  # (B, Lt, d_model)


class TransformerModel(nn.Module):
    """
    Transformer Model: Encoder-Decoder architecture 
    - input:
        - num_layers: número de camadas
        - d_model: dimensão interna
        - num_heads: número de cabeças de atenção
        - d_ff: dimensão feed-forward
        - input_features: número de features de entrada (ex: 4)
        - target_features: número de features de saída (ex: 5)
        - max_pos: comprimento máximo da sequência
        - dropout: taxa de dropout
    - forward:
        - src: (B, Ls, input_features) tensor float
        - tgt: (B, Lt, target_features) tensor float
        - ... máscaras
    - output:
        - predictions: (B, Lt, target_features)
    """
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        input_features: int,
        target_features: int,
        max_pos: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # encoder and decoder
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, input_features, max_pos, dropout)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, target_features, max_pos, dropout)
        # linear layer to project decoder output to target features
        self.final_projection = nn.Linear(d_model, target_features)

    def forward(
        self, src: torch.Tensor, tgt: torch.Tensor, src_mask: Optional[torch.Tensor] = None, tgt_mask: Optional[torch.Tensor] = None, memory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        src: (B, Ls, input_features) float
        tgt: (B, Lt, target_features) float
        masks: optional 
        Returns predictions: (B, Lt, target_features) float
        """
        enc_out = self.encoder(src, src_mask)  # (B, Ls, d_model)
        dec_out = self.decoder(tgt, enc_out, tgt_mask, memory_mask)  # (B, Lt, d_model)
        predictions = self.final_projection(dec_out)
        return predictions # shape (batch, Lt, target_features)


if __name__ == "__main__":
    num_layers = 2
    d_model = 64
    num_heads = 4
    d_ff = 128

    input_features = 4  # inarmonicity, spectral flux, zero crossing rate, energy entropy
    target_features = 8  # pitch in midi cents, amp in midi, grain_size in ms, 4 metro tempo ms, event duration ms
    max_pos = 100       # max sequence length

    # create model
    model = TransformerModel(
        num_layers,
        d_model,
        num_heads,
        d_ff,
        input_features,
        target_features,
        max_pos,
    )
    model.eval()

    # inputs dummy (float)
    # Ls=10 (10 frames de entrada)
    # Lt=10 (10 passos de decoder para prever 10 saídas)
    src = torch.randn(1, 10, input_features)   # (batch, Ls, 4)
    tgt = torch.randn(1, 10, target_features)   # (batch, Lt, 8)

    out = model(src, tgt)
    
    print("Shape da entrada (src):", src.shape)
    print("Shape do decoder input (tgt):", tgt.shape)
    print("Shape da saída (predictions):", out.shape)
    
    # A saída deve ser (batch, Lt, target_features)
    assert out.shape == (1, 10, target_features)
    print("\nTeste de shape passou!")
    
    # Teste TorchScript
    try:
        scripted = torch.jit.script(model)
        out_scripted = scripted(src, tgt)
        assert out_scripted.shape == out.shape
        print("Teste de TorchScript passou!")
    except Exception as e:
        print(f"Teste de TorchScript falhou: {e}")