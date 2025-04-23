# miaosnano/model.py

import torch
import torch.nn as nn
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config["n_heads"]
        self.head_dim = config["n_embd"] // config["n_heads"]
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(config["n_embd"], config["n_embd"] * 3)
        self.proj = nn.Linear(config["n_embd"], config["n_embd"])
        self.dropout = nn.Dropout(config.get("attn_dropout", 0.1))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x).reshape(B, T, self.n_heads, 3 * self.head_dim).permute(3, 0, 2, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, n_heads, T, head_dim]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.masked_fill(torch.triu(torch.ones(T, T), 1).to(x.device) == 1, float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config["n_embd"], 4 * config["n_embd"]),
            nn.GELU(),
            nn.Linear(4 * config["n_embd"], config["n_embd"]),
            nn.Dropout(config.get("resid_dropout", 0.1)),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config["n_embd"])
        self.ln2 = nn.LayerNorm(config["n_embd"])
        self.attn = MultiHeadSelfAttention(config)
        self.ff = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config["vocab_size"]
        self.block_size = config["block_size"]

        self.token_emb = nn.Embedding(self.vocab_size, config["n_embd"])
        self.pos_emb = nn.Parameter(torch.zeros(1, config["block_size"], config["n_embd"]))
        self.drop = nn.Dropout(config.get("embd_dropout", 0.1))

        self.blocks = nn.Sequential(*[Block(config) for _ in range(config["n_layer"])])
        self.ln_f = nn.LayerNorm(config["n_embd"])
        self.head = nn.Linear(config["n_embd"], self.vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.block_size, "Block size exceeded!"

        tok_emb = self.token_emb(idx)
        x = tok_emb + self.pos_emb[:, :T, :]
        x = self.drop(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits
