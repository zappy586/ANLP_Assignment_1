import torch
from torch import nn

from positional import SinusoidalPositionalEmbeddings
from attention import MHA, CrossAttention
from norm import LayerNorm

class ByteEmbeddings(nn.Module):
    def __init__(self, vocab_size, model_dim):
        self.vocab_size = vocab_size
        self.embedding_table = nn.Embedding(vocab_size, model_dim)

    def forward(self, x):
        x = self.embedding_table(x)
        return x

class BLTEncoder(nn.Module):
    def __init__(self, vocab_size, num_blocks, num_heads, model_dim):
        super().__init__()
        self.byte_embedding_layer = ByteEmbeddings(vocab_size, model_dim)
        self.positional_embeddings = SinusoidalPositionalEmbeddings(model_dim=model_dim)
        self.mha_blocks = nn.ModuleList([mha_block for mha_block in ])

    def forward(self, x):


