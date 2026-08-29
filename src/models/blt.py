import torch
import torch.nn.functional as F
from torch import nn

from positional import SinusoidalPositionalEmbeddings
from attention import MHA, CrossAttention
from norm import LayerNorm

class ByteEmbeddings(nn.Module):
    def __init__(self, vocab_size, model_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_table = nn.Embedding(vocab_size, model_dim)

    def forward(self, x):
        x = self.embedding_table(x)
        return x

class BLTTransformerEncoderBlock(nn.Module):
    def __init__(self, num_heads, model_dim):
        super().__init__()
        self.attn = CrossAttention(num_heads, model_dim)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, 4 * model_dim),
            nn.GELU(),
            nn.Linear(4 * model_dim, model_dim)
        )

    def forward(self, KV, Q):
        x = Q + (self.attn(self.norm1(KV), self.norm1(Q)))
        x = x + self.mlp(self.norm2(x))
        return x

class BLTEncoder(nn.Module):
    def __init__(self, vocab_size, num_layers, num_heads, model_dim, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.model_dim = model_dim
        self.byte_embeddings = ByteEmbeddings(vocab_size, model_dim)
        self.positional_embeddings = SinusoidalPositionalEmbeddings(model_dim)
        self.transformer_layers = nn.ModuleList([BLTTransformerEncoderBlock(num_heads, model_dim) for _ in range(num_layers)])

    def forward(self, x):
        b, n = x.shape
        padding_len = n % self.patch_size

        if padding_len != 0:   # if the seq length is not divisible by patch size, we pad it.
            pad_len = self.patch_size - padding_len
            x = F.pad(x, (0, pad_len), value=0)
            n = x.shape[1]

        x = self.byte_embeddings(x)
        x = self.positional_embeddings(x)

        num_patches = n // self.patch_size
        kv = x.view(b * num_patches, self.patch_size, self.model_dim)
        q = kv.mean(dim=1, keepdim=True)
        for layer in self.transformer_layers:
            q = layer(KV=kv, Q=q)

        x = q.view(b, num_patches, self.model_dim)
        return x
        
class BLTTransformerDecoderBlock(nn.Module):
    def __init__(self, num_heads, model_dim):
        super().__init__()
        self.self_attn = MHA(num_heads, model_dim)
        self.cross_attn = CrossAttention(num_heads, model_dim)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm_kv = nn.LayerNorm(model_dim)
        self.norm_q = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, 4 * model_dim),
            nn.GELU(),
            nn.Linear(4 * model_dim, model_dim)
        )

    def forward(self, x, encoder_out):
        x = x + self.self_attn(self.norm1(x))
        x = x + self.cross_attn(self.norm_kv(encoder_out), self.norm_q(x))
        x = x + self.mlp(self.norm2(x))
        return x

class BLTDecoder(nn.Module):
    def __init__(self, vocab_size, num_layers, num_heads, model_dim):
        super().__init__()
        self.model_dim = model_dim
        self.byte_embeddings = ByteEmbeddings(vocab_size, model_dim)
        self.positional_embeddings = SinusoidalPositionalEmbeddings(model_dim)
        self.transformer_layers = nn.ModuleList([BLTTransformerDecoderBlock(num_heads, model_dim) for _ in range(num_layers)])
        self.fc_out = nn.Linear(model_dim, vocab_size)

    def forward(self, x, encoder_out, tgt_mask=None):
        x = self.byte_embeddings(x)
        x = self.positional_embeddings(x)

        for layer in self.transformer_layers:
            x = layer(x, encoder_out)

        x = self.fc_out(x)
        return x


