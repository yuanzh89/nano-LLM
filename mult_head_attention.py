import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttentionBlock(nn.Module):
    """
    Implement Multi Head Attention block with tunable num_query_heads and num_key_value_heads.
    So that can be setup as Multi Query Attention block or Group Query Attention block.
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, mask: torch.Tensor = None):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = torch.tensor(d_model // num_heads)

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.o_proj = nn.Linear(d_model, d_model, bias=True)

        self.dropout = None if math.isclose(dropout, 0.0) else nn.Dropout(dropout)

        self.mask = mask

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = input.size()

        # Pre-norm
        normed_input = self.layer_norm(input)

        # (batch_size, seq_len, d_model) -> (batch_size, num_heads, seq_len, d_k)
        Q = self.q_proj(normed_input).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(normed_input).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(normed_input).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # (batch_size, num_heads, seq_len, seq_len)
        attn_scores = torch.matmul(Q, K.transpose(-1, -2))
        attn_scores = attn_scores / torch.sqrt(self.d_k)

        if self.mask is not None:
            attn_scores = attn_scores.masked_fill(self.mask == 0, -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1)

        if self.dropout is not None:
            attn_weights = self.dropout(attn_weights)

        # (batch_size, num_heads, seq_len, d_k)
        attn_output = torch.matmul(attn_weights, V)
        # (batch_size, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, d_model)

        # (batch_size, seq_len, d_model)
        output = self.o_proj(attn_output)

        output = input + output

        return output, attn_weights