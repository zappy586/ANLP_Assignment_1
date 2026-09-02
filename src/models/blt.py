import torch
import torch.nn.functional as F
from torch import nn

from .positional import SinusoidalPositionalEmbeddings
from .attention import MHA, CrossAttention
from .norm import LayerNorm


class ByteEmbeddings(nn.Module):
    def __init__(self, vocab_size, model_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_table = nn.Embedding(vocab_size, model_dim)

    def forward(self, x):
        return self.embedding_table(x)


class BLTTransformerEncoderBlock(nn.Module):
    def __init__(self, num_heads, model_dim):
        super().__init__()
        self.attn = CrossAttention(num_heads, model_dim)
        self.norm_kv = LayerNorm(model_dim)
        self.norm_q = LayerNorm(model_dim)
        self.norm2 = LayerNorm(model_dim)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, 4 * model_dim),
            nn.GELU(),
            nn.Linear(4 * model_dim, model_dim)
        )

    def forward(self, KV, Q):
        x = Q + self.attn(self.norm_kv(KV), self.norm_q(Q))
        x = x + self.mlp(self.norm2(x))
        return x


class BLTEncoder(nn.Module):
    def __init__(self, vocab_size, num_layers, num_heads, model_dim, patch_size=4, pad_token_id=0):
        super().__init__()
        self.patch_size = patch_size
        self.model_dim = model_dim
        self.pad_token_id = pad_token_id
        self.byte_embeddings = ByteEmbeddings(vocab_size, model_dim)
        self.positional_embeddings = SinusoidalPositionalEmbeddings(model_dim)

        self.patch_proj = nn.Linear(patch_size * model_dim, model_dim)

        self.byte_attn = nn.ModuleList([MHA(num_heads, model_dim) for _ in range(num_layers)])
        self.byte_norm = nn.ModuleList([LayerNorm(model_dim) for _ in range(num_layers)])

        self.transformer_layers = nn.ModuleList(
            [BLTTransformerEncoderBlock(num_heads, model_dim) for _ in range(num_layers)]
        )

    def forward(self, x):
        b, n = x.shape
        remainder = n % self.patch_size

        if remainder != 0:
            pad_len = self.patch_size - remainder
            x = F.pad(x, (0, pad_len), value=self.pad_token_id)
            n = x.shape[1]

        emb = self.byte_embeddings(x)
        emb = self.positional_embeddings(emb)

        num_patches = n // self.patch_size
        kv = emb.view(b * num_patches, self.patch_size, self.model_dim)

        # initial patch vector preserves bit order (concat + project, not mean)
        q = self.patch_proj(kv.reshape(b * num_patches, -1)).unsqueeze(1)

        for attn, norm, layer in zip(self.byte_attn, self.byte_norm, self.transformer_layers):
            kv = kv + attn(norm(kv))      # bytes talk to each other
            q = layer(KV=kv, Q=q)         # then pool into the patch vector

        return q.view(b, num_patches, self.model_dim)


class BLTTransformerDecoderBlock(nn.Module):
    def __init__(self, num_heads, model_dim):
        super().__init__()
        self.self_attn = MHA(num_heads, model_dim)
        self.cross_attn = CrossAttention(num_heads, model_dim)
        self.norm1 = LayerNorm(model_dim)
        self.norm_kv = LayerNorm(model_dim)
        self.norm_q = LayerNorm(model_dim)
        self.norm2 = LayerNorm(model_dim)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, 4 * model_dim),
            nn.GELU(),
            nn.Linear(4 * model_dim, model_dim)
        )

    def forward(self, x, encoder_out, self_mask=None, cross_mask=None):
        x = x + self.self_attn(self.norm1(x), self_mask)
        x = x + self.cross_attn(self.norm_kv(encoder_out), self.norm_q(x), cross_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class BLTDecoder(nn.Module):
    def __init__(self, vocab_size, num_layers, num_heads, model_dim):
        super().__init__()
        self.model_dim = model_dim
        self.byte_embeddings = ByteEmbeddings(vocab_size, model_dim)
        self.positional_embeddings = SinusoidalPositionalEmbeddings(model_dim)
        self.transformer_layers = nn.ModuleList([BLTTransformerDecoderBlock(num_heads, model_dim) for _ in range(num_layers)])
        self.final_norm = LayerNorm(model_dim)
        self.fc_out = nn.Linear(model_dim, vocab_size)

    def forward(self, x, encoder_out, self_mask=None, cross_mask=None):
        x = self.byte_embeddings(x)
        x = self.positional_embeddings(x)

        for layer in self.transformer_layers:
            x = layer(x, encoder_out, self_mask, cross_mask)

        return self.fc_out(self.final_norm(x))