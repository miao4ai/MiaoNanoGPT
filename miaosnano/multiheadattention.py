import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_model, n_heads):
        super().__init__()

        self.d_k = d_k
        self.n_heads = n_heads

        self.key = nn.Linear(d_model, d_k * n_heads)
        self.query = nn.Linear(d_model, d_k * n_heads)
        self.value = nn.Linear(d_model, d_k * n_heads)
        self.fc = nn.Linear(d_k * n_heads, d_model)

    def forward(self, x):
        batch_size = x.size(0)

        # Linear projections: (B, T, D) -> (B, T, n_heads * d_k)
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # Reshape to (B, n_heads, T, d_k)
        k = k.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        q = q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention: (B, n_heads, T, T)
        attn_scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Attention output: (B, n_heads, T, d_k) -> (B, T, n_heads * d_k)
        attn_output = attn_weights @ v
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)

        # Final linear projection: (B, T, D)
        return self.fc(attn_output)
