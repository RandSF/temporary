import math
from typing import Union, Optional, Tuple

from einops.layers.torch import Rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import Block

from functools import partial
from typing import Tuple

from timm.models.vision_transformer import Mlp, PatchEmbed , Attention
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class CrossAttention(nn.Module):
    def __init__(self, dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        context_dim = dim if context_dim is None else context_dim   # default(context_dim, dim)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, context=None):
        context = x if context is None else context   # default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q = self.to_q(x)
        q, k, v = map(lambda t: Rearrange(t, "b n (h d) -> b h n d", h=self.heads), [q, k, v])

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = Rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)
    
class DropTokenDropout(nn.Module):
    def __init__(self, p: float = 0.1):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(
                "dropout probability has to be between 0 and 1, " "but got {}".format(p)
            )
        self.p = p

    def forward(self, x: torch.Tensor):
        # x: (batch_size, seq_len, dim)
        if self.training and self.p > 0:
            zero_mask = torch.full_like(x[0, :, 0], self.p).bernoulli().bool()
            # TODO: permutation idx for each batch using torch.argsort
            if zero_mask.any():
                x = x[:, ~zero_mask, :]
        return x


class ZeroTokenDropout(nn.Module):
    def __init__(self, p: float = 0.1):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(
                "dropout probability has to be between 0 and 1, " "but got {}".format(p)
            )
        self.p = p

    def forward(self, x: torch.Tensor):
        # x: (batch_size, seq_len, dim)
        if self.training and self.p > 0:
            zero_mask = torch.full_like(x[:, :, 0], self.p).bernoulli().bool()
            # Zero-out the masked tokens
            x[zero_mask, :] = 0
        return x

class CABlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        context_dim: Optional[int] = None,
    ):
        super().__init__()

        self.sa = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.ca = CrossAttention(
            dim, context_dim=context_dim, heads=heads, dim_head=dim_head, dropout=dropout
        )
        self.ff = Mlp(dim, mlp_dim, drop=dropout)

        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ln3 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, context=None):
        x = self.sa(self.ln1(x)) + x
        x = self.ca(self.ln2(x), context) + x
        x = self.ff(self.ln3(x)) + x
        return x

class SABlock(nn.Module):
    def __init__(self, 
                 dim: int,
                 heads: int,
                 dim_head: int,
                 mlp_dim: int,
                 dropout: float = 0.0,) -> None:
        super().__init__()
        self.sa = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.ff = Mlp(dim, mlp_dim, drop=dropout)

        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor):
        x = self.sa(self.ln1(x)) + x
        x = self.ff(self.ln2(x)) + x
        return x
    
class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        token_dim: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        emb_dropout_type: str = "drop",
        emb_dropout_loc: str = "token",
        norm: str = "layer",
        norm_cond_dim: int = -1,
        token_pe_numfreq: int = -1,
    ):
        super().__init__()

        if token_pe_numfreq > 0:
            token_dim_new = token_dim * (2 * token_pe_numfreq + 1)
            self.to_token_embedding = nn.Sequential(
                Rearrange("b n d -> (b n) d", n=num_tokens, d=token_dim),
                FrequencyEmbedder(token_pe_numfreq, token_pe_numfreq - 1),
                Rearrange("(b n) d -> b n d", n=num_tokens, d=token_dim_new),
                nn.Linear(token_dim_new, dim),
            )
        else:
            self.to_token_embedding = nn.Linear(token_dim, dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, dim))

        if emb_dropout_type == "drop":
            self.dropout = DropTokenDropout(emb_dropout)
        elif emb_dropout_type == "zero":
            self.dropout = ZeroTokenDropout(emb_dropout)
        else:
            raise ValueError(f"Unknown emb_dropout_type: {emb_dropout_type}")
        self.emb_dropout_loc = emb_dropout_loc

        self.blocks = nn.ModuleList([
            SABlock(dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)
        ])

    def forward(self, inp: torch.Tensor):
        x = inp

        if self.emb_dropout_loc == "input":
            x = self.dropout(x)
        x = self.to_token_embedding(x)

        if self.emb_dropout_loc == "token":
            x = self.dropout(x)
        b, n, _ = x.shape
        x += self.pos_embedding[:, :n]

        if self.emb_dropout_loc == "token_afterpos":
            x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        return x

class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        token_dim: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        emb_dropout_type: str = 'drop',
        context_dim: Optional[int] = None,
        skip_token_embedding: bool = False,
    ):
        super().__init__()
        if not skip_token_embedding:
            self.to_token_embedding = nn.Linear(token_dim, dim)
        else:
            self.to_token_embedding = nn.Identity()
            if token_dim != dim:
                raise ValueError(
                    f"token_dim ({token_dim}) != dim ({dim}) when skip_token_embedding is True"
                )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, dim))
        if emb_dropout_type == "drop":
            self.dropout = DropTokenDropout(emb_dropout)
        elif emb_dropout_type == "zero":
            self.dropout = ZeroTokenDropout(emb_dropout)
        elif emb_dropout_type == "normal":
            self.dropout = nn.Dropout(emb_dropout)

        self.blocks = nn.ModuleList([
            CABlock(dim, heads, dim_head, mlp_dim, dropout, context_dim=context_dim)
        for _ in range(depth)])

    def forward(self, inp: torch.Tensor, context=None):
        x = self.to_token_embedding(inp)
        b, n, _ = x.shape

        x = self.dropout(x)
        x += self.pos_embedding[:, :n]

        for block in self.blocks:
            x = block(x, context=context)
        return x