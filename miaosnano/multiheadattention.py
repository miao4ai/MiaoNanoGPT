import math
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_model, n_heads):
        super().__init__()

        self.d_k = d_k
        self.n_heads = n_heads

        self.key = nn.Linear(d_model, d_k*n_heads)
        self.query = nn.Linear(d_model, d_k* n_heads)
    
    def forward(self,x):