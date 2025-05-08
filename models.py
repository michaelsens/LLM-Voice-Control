
import argparse
import time
import random
import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


import torch
import torch.nn as nn



class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (B, T, D)
        scale = (x.pow(2).mean(-1, keepdim=True) + self.eps).rsqrt()
        return self.weight * x * scale


def causal_mask(t: int, device):
    """(t, t) mask with -inf above diagonal."""
    return torch.triu(torch.full((t, t), float("-inf"), device=device), 1)



class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = RMSNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, key_padding_mask=None):
        x = x + self.attn(
            self.norm1(x), self.norm1(x), self.norm1(x),
            key_padding_mask=key_padding_mask
        )[0]
        x = x + self.mlp(self.norm2(x))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = RMSNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm3 = RMSNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self, x, enc_out,
        *, tgt_mask, tgt_key_padding_mask=None, src_key_padding_mask=None
    ):
        x = x + self.self_attn(
            self.norm1(x), self.norm1(x), self.norm1(x),
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )[0]
        x = x + self.cross_attn(
            self.norm2(x), enc_out, enc_out,
            key_padding_mask=src_key_padding_mask
        )[0]
        x = x + self.mlp(self.norm3(x))
        return x



class MiniRPCTransformer(nn.Module):


    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_enc_layers: int = 4,
        n_dec_layers: int = 4,
        max_len: int = 128,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.max_len = max_len

        # embeddings
        self.src_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.tgt_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))

        # encoder & decoder stacks
        self.encoder = nn.ModuleList(
            [EncoderBlock(d_model, n_heads, dropout) for _ in range(n_enc_layers)]
        )
        self.decoder = nn.ModuleList(
            [DecoderBlock(d_model, n_heads, dropout) for _ in range(n_dec_layers)]
        )

        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tgt_emb.weight  

    def make_pad_mask(self, seq):
        return (seq == self.pad_id)  


    def forward(self, src, tgt_in):

        B, S = src.shape
        _, T = tgt_in.shape
        if S > self.max_len or T > self.max_len:
            raise ValueError("Sequence length exceeds max_len")

        src_mask = self.make_pad_mask(src)
        tgt_mask = causal_mask(T, tgt_in.device)
        tgt_pad  = self.make_pad_mask(tgt_in)


        x = self.src_emb(src) + self.pos_emb[:, :S]
        for blk in self.encoder:
            x = blk(x, key_padding_mask=src_mask)
        enc_out = x                                   

        y = self.tgt_emb(tgt_in) + self.pos_emb[:, :T]
        for blk in self.decoder:
            y = blk(y, enc_out,
                    tgt_mask=tgt_mask,
                    tgt_key_padding_mask=tgt_pad,
                    src_key_padding_mask=src_mask)

        logits = self.lm_head(self.final_norm(y))     
        return logits