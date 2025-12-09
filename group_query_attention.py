import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupQueryAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int, dropout: float = 0.1, mask: torch.Tensor = None):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.num_kv_groups = num_heads // num_kv_heads

        # Projections
        # Q projects to standard multi-head dimension
        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        # K and V project to the compressed (grouped) dimension
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)

        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, d_model = input.shape

        # 1. Linear projections
        q = self.q_proj(input)  # (batch_size, seq_len, num_heads * head_dim)
        k = self.k_proj(input)  # (batch_size, seq_len, num_kv_heads * head_dim)
        v = self.v_proj(input)  # (batch_size, seq_len, num_kv_heads * head_dim)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)



