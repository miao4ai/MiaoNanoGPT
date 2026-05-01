import torch
import torch.nn as nn
import torch.nn.functional as F

from miaosnano.embedding import embedding
from miaosnano.positional_encoding import PositionalEncoding
from miaosnano.transformer_block import TransformerBlock


class MiaoGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048,
        max_len=1024,
        dropout=0.1,
    ):
        super().__init__()

        self.max_len = max_len

        self.token_embedding = embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_len)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Causal mask: True 表示需要遮蔽（上三角，不含对角线）
        causal_mask = torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1)
        self.register_buffer('causal_mask', causal_mask)

    def forward(self, idx, targets=None):
        # idx: (B, T) 整数 token id
        B, T = idx.shape
        assert T <= self.max_len, f"序列长度 {T} 超过 max_len={self.max_len}"

        # Token + positional embedding
        x = self.token_embedding(idx)              # (B, T, d_model)
        x = self.positional_encoding(x)            # (B, T, d_model)
        x = self.dropout(x)

        # 为当前序列长度切出 causal mask: (1, 1, T, T) 方便广播到 (B, n_heads, T, T)
        mask = self.causal_mask[:T, :T].view(1, 1, T, T)

        # 堆叠 N 层 Transformer block
        for block in self.blocks:
            x = block(x, mask=mask)

        x = self.norm(x)
        logits = self.lm_head(x)                   # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        # idx: (B, T) 起始 prompt
        for _ in range(max_new_tokens):
            # 截断到 max_len
            idx_cond = idx[:, -self.max_len:]
            logits, _ = self(idx_cond)

            # 取最后一个 token 的 logits
            logits = logits[:, -1, :] / temperature

            # 可选 top-k 采样
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)

        return idx
