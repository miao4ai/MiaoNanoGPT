import torch
import torch.nn as nn


class embedding(nn.Module):
    __init__(vocab_size: int, output_dim: int):
        self.dim = vocab_size
        self.output_dim = output_dim
        self.embedding_layer = nn.Embedding(self.dim, self.output_dim)
 

