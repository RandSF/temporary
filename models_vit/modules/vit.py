import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from timm.layers import PatchEmbed, Mlp, DropPath, trunc_normal_

def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    Args
        timesteps (torch.Tensor):
            a 1-D Tensor of N indices, one per batch element. These may be fractional.
        embedding_dim (int):
            the dimension of the output.
        flip_sin_to_cos (bool):
            Whether the embedding order should be `cos, sin` (if True) or `sin, cos` (if False)
        downscale_freq_shift (float):
            Controls the delta between frequencies between dimensions
        scale (float):
            Scaling factor applied to the embeddings.
        max_period (int):
            Controls the maximum frequency of the embeddings
    Returns
        torch.Tensor: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

def get_position_embedding(patch_size = 32, half_embed_dim = 128, temperature=10000):
    not_mask = torch.ones((1, patch_size, patch_size), dtype=torch.float32)

    y_embed = not_mask.cumsum(1, dtype=torch.float32)
    x_embed = not_mask.cumsum(2, dtype=torch.float32)

    eps = 1e-6
    y_embed = y_embed / (y_embed[:, -1:, :] + eps) * 2*math.pi
    x_embed = x_embed / (x_embed[:, :, -1:] + eps) * 2*math.pi

    dim_t = torch.arange(half_embed_dim, dtype=torch.float32)
    dim_t = temperature ** (2 * (dim_t // 2) / half_embed_dim)

    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t
    pos_x = torch.stack(
        (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
    ).flatten(3)
    pos_y = torch.stack(
        (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
    ).flatten(3)
    pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
    return pos

class LayerScale(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = True,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class SelfAttention(nn.Module):

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor, 
                mask = None) -> torch.Tensor:
        B, L, C = query.shape
        B, S, C = key.shape
        assert key.shape == value.shape
        q = self.q(query).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(key).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(value).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if mask is not None:
            attn.masked_fill_(mask, float('-inf'))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, L, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class EncoderLayer(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: float = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.sa = SelfAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.sa(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class DecoderLayer(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: float = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.sa = SelfAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.ca = CrossAttention(
            dim, 
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm3 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls3 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, tgt: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        x = tgt
        x = x + self.drop_path1(self.ls1(self.sa(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.ca(self.norm2(x), mem, mem)))
        x = x + self.drop_path3(self.ls3(self.mlp(self.norm3(x))))
        return x
    

class TVAE(nn.Module):
    def __init__(self,
                 num_k = 16,
                 num_timestep = 3,
                 num_joint = 21,
                 patch_size = 8, 
                 input_dim = 256, 
                 embed_dim = 256, 
                 num_block = 4,
                 num_heads = 4, 
                 drop_rate = 0.1,
                 learnable_te = False,
                 **kwargs):
        super().__init__()
        self.num_k = num_k
        self.num_t = num_timestep
        self.num_j = num_joint
        self.embed_dim = embed_dim

        self.pe_joint = nn.Parameter(torch.zeros(1, 1, num_joint, embed_dim))  # [1, 1, J, E]
        trunc_normal_(self.pe_joint, std=0.2)
        pe_img = get_position_embedding(patch_size, input_dim//2)   # [1, E, H, W]
        self.register_buffer('pe_img', pe_img.flatten(-2, -1).transpose(1, 2))  # [1, HW, E]
        if learnable_te:
            pe_time = get_timestep_embedding(torch.arange(3))
            self.register_buffer('pe_time', pe_time[None, :, None, :])  # [1, T, 1, E]
        else:
            self.pe_time = nn.Parameter(torch.zeros(1, num_timestep, 1, embed_dim))  # [1, T, 1, E]
            trunc_normal_(self.pe_time, std=0.2)

        self.patch_layer = PatchEmbed(img_size=patch_size, patch_size=1, in_chans=2048, embed_dim=embed_dim)

        self.candidate_gen = Normal(torch.zeros([embed_dim,]).cuda(), torch.ones([embed_dim,]).cuda())
        self.cand_layer = Mlp(embed_dim, 4*embed_dim, norm_layer=nn.BatchNorm1d, drop=drop_rate)

        self.encoder = nn.ModuleList([
            EncoderLayer(dim = embed_dim, num_heads=num_heads, attn_drop=drop_rate, proj_drop=drop_rate, drop_path=drop_rate)
            for _ in range(num_block)
        ])

        self.decoder = nn.ModuleList([
            DecoderLayer(dim = embed_dim, num_heads=num_heads, attn_drop=drop_rate, proj_drop=drop_rate, drop_path=drop_rate)
            for _ in range(num_block)
        ])

    def forward(self, feat_base, feat_joint):
        # feat_base: [B, C, H, W]
        # feat_joint: [B, T, J, E]
        B, C, H, W = feat_base.shape
        K, T, J, E = self.num_k, self.num_t, self.num_j, self.embed_dim

        candidates = self.cand_layer(self.candidate_gen.sample([B, K, ]).flatten(0, 1)).reshape([B, K, E])

        x = self.patch_layer(feat_base)   # [B, HW, E]

        x = torch.cat([candidates, x], dim=1)   # [B, K+HW, E]

        pos_embed = torch.cat([torch.zeros(1, K, E).to(feat_base.device), self.pe_img], dim=1)
        for block in self.encoder:
            x = block(x + pos_embed)

        feat_cand = x[:, :K]  # [B, K, E]

        mem = feat_cand.flatten(0, 1).unsqueeze(1)   # [BK, 1, E]

        # tgt = torch.zeros(B*K, J*T, E).to(feat_base.device)  # [BK, JT, E]
        tgt = feat_joint.unsqueeze(1).expand(B, K, T, J, E).flatten(2, 3).flatten(0, 1)    # [BK, TJ, E]
        pos = (self.pe_time + self.pe_joint).flatten(1, 2)
        for block in self.decoder:
            tgt = block(tgt + pos, mem)

        return tgt.reshape(B, K, T, J, E), candidates
    

if __name__ == '__main__':
    B, C, H, W = 24, 256, 8, 8
    T, J = 3, 21
    dummy = torch.randn(B, 2048, H, W).cuda()
    dummy2 = torch.randn(B, T, J, C).cuda()
    model = TVAE().cuda()

    res, candidates = model(dummy, dummy2)

    import numpy as np

    np.savetxt('./temp_cand.txt', candidates.detach().flatten(0, 1).cpu().numpy())
    print(res.shape)









