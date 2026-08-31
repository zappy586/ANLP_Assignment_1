import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MHA, GQA, CrossAttention, GQACrossAttention
from .positional import SinusoidalPositionalEmbeddings, MHA_RoPE
from .norm import LayerNorm, RMSNorm
from .blt import BLTEncoder, BLTDecoder


def pad_mask(seq, pad_id):
    return (seq != pad_id).unsqueeze(1).unsqueeze(2)


def causal_mask(size, device):
    return torch.tril(torch.ones(size, size, device=device, dtype=torch.bool))


class Block(nn.Module):
    def __init__(self, model_dim, num_heads, self_attn, cross_attn, norm):
        super().__init__()
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.norms = nn.ModuleList([norm(model_dim) for _ in range(4)])
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, 4 * model_dim),
            nn.GELU(),
            nn.Linear(4 * model_dim, model_dim),
        )

    def forward(self, x, enc=None, self_m=None, cross_m=None):
        x = x + self.self_attn(self.norms[0](x), self_m)
        if self.cross_attn is not None:
            x = x + self.cross_attn(self.norms[1](enc), self.norms[2](x), cross_m)
        return x + self.ffn(self.norms[3](x))


class Transformer(nn.Module):
    def __init__(self, src_size, tgt_size, src_pad, tgt_pad, model_dim=256, num_heads=8,
                 num_layers=4, attn="mha", norm="ln", rope=False):
        super().__init__()
        self.src_pad, self.tgt_pad, self.rope = src_pad, tgt_pad, rope
        norm = RMSNorm if norm == "rms" else LayerNorm

        def sa():
            if attn == "rope":
                return MHA_RoPE(num_heads, model_dim)
            return GQA(num_heads, 2, model_dim) if attn == "gqa" else MHA(num_heads, model_dim)

        def ca():
            return GQACrossAttention(num_heads, 2, model_dim) if attn == "gqa" else CrossAttention(num_heads, model_dim)

        self.src_emb = nn.Embedding(src_size, model_dim)
        self.tgt_emb = nn.Embedding(tgt_size, model_dim)
        self.pos = SinusoidalPositionalEmbeddings(model_dim)
        self.enc = nn.ModuleList([Block(model_dim, num_heads, sa(), None, norm) for _ in range(num_layers)])
        self.dec = nn.ModuleList([Block(model_dim, num_heads, sa(), ca(), norm) for _ in range(num_layers)])
        self.out = nn.Linear(model_dim, tgt_size)

    def embed(self, tokens, table):
        x = table(tokens)
        return x if self.rope else self.pos(x)

    def encode(self, src):
        src_m = pad_mask(src, self.src_pad)
        x = self.embed(src, self.src_emb)
        for layer in self.enc:
            x = layer(x, self_m=src_m)
        return x, src_m

    def decode(self, tgt, enc, src_m):
        tgt_m = pad_mask(tgt, self.tgt_pad) & causal_mask(tgt.shape[1], tgt.device)
        y = self.embed(tgt, self.tgt_emb)
        for layer in self.dec:
            y = layer(y, enc=enc, self_m=tgt_m, cross_m=src_m)
        return self.out(y)

    def forward(self, src, tgt):
        enc, src_m = self.encode(src)
        return self.decode(tgt, enc, src_m)


class BLT(nn.Module):
    def __init__(self, vocab_size, src_pad, tgt_pad, model_dim=256, num_heads=8,
                 num_layers=4, patch_size=4, local_enc_layers=2, local_dec_layers=2, eos_id=None):
        super().__init__()
        self.src_pad, self.tgt_pad, self.patch_size = src_pad, tgt_pad, patch_size
        self.local_enc = BLTEncoder(vocab_size, local_enc_layers, num_heads, model_dim, patch_size, src_pad)
        self.glob = nn.ModuleList([
            Block(model_dim, num_heads, MHA(num_heads, model_dim), None, LayerNorm)
            for _ in range(num_layers)
        ])
        self.patch_pos = SinusoidalPositionalEmbeddings(model_dim)
        self.local_dec = BLTDecoder(vocab_size, local_dec_layers, num_heads, model_dim)
        self.eos_id = eos_id

    def encode(self, src):
        bits = src[:, 1:]                     
        b, n = bits.shape
        rem = n % self.patch_size
        if rem:
            bits = F.pad(bits, (0, self.patch_size - rem), value=self.src_pad)

        patches = self.local_enc(bits)

        valid = (bits != self.src_pad) & (bits != self.eos_id)
        cross_m = valid.view(b, -1, self.patch_size).any(-1).unsqueeze(1).unsqueeze(2)

        x = self.patch_pos(patches)
        for layer in self.glob:
            x = layer(x, self_m=cross_m)
        return x, cross_m

    def decode(self, tgt, enc, cross_m):
        tgt_m = pad_mask(tgt, self.tgt_pad) & causal_mask(tgt.shape[1], tgt.device)
        return self.local_dec(tgt, enc, tgt_m, cross_m)

    def forward(self, src, tgt):
        enc, cross_m = self.encode(src)
        return self.decode(tgt, enc, cross_m)


def build_model(config, meta, model_dim=256, num_heads=8, num_layers=4, patch_size=4, local_enc_layers=2, local_dec_layers=2):
    src_pad = meta["source_specials"]["pad_token_id"]
    tgt_pad = meta["target_specials"]["pad_token_id"]
    src_size = meta["source_specials"]["embedding_size"]
    tgt_size = meta["target_specials"]["embedding_size"]

    if config == "C5":
        return BLT(src_size, src_pad, tgt_pad, model_dim, num_heads, num_layers, patch_size,local_enc_layers, local_dec_layers, eos_id=meta["source_specials"]["eos_token_id"])

    kwargs = {
        "C1": {},
        "C2": {"attn": "rope", "rope": True},
        "C3": {"attn": "gqa"},
        "C4": {"norm": "rms"},
    }[config]

    return Transformer(src_size, tgt_size, src_pad, tgt_pad, model_dim, num_heads, num_layers, **kwargs)


