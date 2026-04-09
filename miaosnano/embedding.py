import torch
import torch.nn as nn


class embedding(nn.Module):
    def __init__(self, vocab_size: int, output_dim: int):
        super().__init__()
        self.dim = vocab_size
        self.output_dim = output_dim
        self.embedding_layer = nn.Embedding(self.dim, self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding_layer(x)
