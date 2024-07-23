from typing import Optional
import math
from einops import rearrange
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, LayerScale, DropPath
from timm.layers import Mlp, PatchEmbed
from timm.layers import trunc_normal_

def get_timestep_embedding(
    timesteps: int,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """
    Returns
        torch.Tensor: an [N x dim] Tensor of positional embeddings.
    """
    # assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32#, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = torch.arange(timesteps).unsqueeze(-1).float() * emb[None, :]

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

def get_position_embedding(patch_size = 8, embed_dim = 256, temperature=10000):
    half_embed_dim = embed_dim // 2
    not_mask = torch.ones((1, patch_size, patch_size), dtype=torch.float32)

    y_embed = not_mask.cumsum(1, dtype=torch.float32)
    x_embed = not_mask.cumsum(2, dtype=torch.float32)

    eps = 1e-6
    y_embed = y_embed / (y_embed[:, -1:, :] + eps) * 2*torch.pi
    x_embed = x_embed / (x_embed[:, :, -1:] + eps) * 2*torch.pi

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

class Attention(nn.Module):

    def __init__(
            self,
            dim: int = 256,
            num_heads: int = 8,
            qkv_bias: bool = False,
            drop: float = 0.,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        q, k, v = self.q(query), self.k(key), self.v(value)
        q, k, v = [rearrange(x, 'b n (h e) -> b h n e', h=self.num_heads, e=self.head_dim) for x in [q,k,v]]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = rearrange(x, 'b h n e -> b n (h e)')
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class SABlock(nn.Module):
    def __init__(self, dim: int = 256, mlp_ratio: float = 4., init_values: Optional[float] = None,
                 num_heads: int = 8, qkv_bias: bool = False, drop: float = 0., 
                 **kwargs) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, drop=drop)
        self.ls1 = LayerScale(dim) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop) if drop > 0. else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop) if drop > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class CABlock(nn.Module):
    def __init__(self, dim: int = 256, mlp_ratio: float = 4., init_values: Optional[float] = None,
                 num_heads: int = 8, qkv_bias: bool = False, drop: float = 0., 
                 **kwargs) -> None:
        super().__init__()
        self.norm1_t = nn.LayerNorm(dim)
        self.norm1_m = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, drop=drop)
        self.ls1 = LayerScale(dim) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop) if drop > 0. else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop) if drop > 0. else nn.Identity()

    def forward(self, tgt: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        x = tgt + self.drop_path1(self.ls1(self.attn(self.norm1_t(tgt), self.norm1_m(mem), self.norm1_m(mem))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class Transformer(nn.Module):
    def __init__(self, num_k = 16, num_timestep = 3, in_chans = 2048, embed_dim = 256, patch_size = 8, out_dim = 256, 
                 num_blocks = 4, num_heads = 4, drop_rate = 0.1, 
                 **kwargs) -> None:
        super().__init__()
        self.num_k = num_k
        self.num_t = num_timestep
        self.dim = out_dim

        self.patch_emb = nn.Linear(in_chans, embed_dim)

        self.time_unfolder = nn.ModuleList([
            nn.ModuleList([SABlock(embed_dim, num_heads=num_heads, drop=drop_rate) for _ in range(num_blocks)]) 
            for t in range(num_timestep)
        ])

        self.extractor = nn.ModuleList([
            CABlock(embed_dim, num_heads=num_heads, drop=drop_rate) for _ in range(num_blocks)
        ])

        self.mixer = nn.ModuleList([
            SABlock(embed_dim, num_heads=num_heads, drop=drop_rate) for _ in range(num_blocks)
        ])

        self.out_proj = nn.Linear(embed_dim, out_dim)

        pe_patch = get_position_embedding(patch_size, embed_dim)   
        self.register_buffer('pe_patch', rearrange(pe_patch, 'b e h w -> b (h w) e'))    # [1, HW, E]

        self.candidate_token = nn.Parameter(torch.zeros(1, num_k, embed_dim))
        trunc_normal_(self.candidate_token)

        self.pe_time = nn.Parameter(torch.zeros(1, 1, num_k, embed_dim))
        self.pe_cand = nn.Parameter(torch.zeros(1, num_timestep, 1, embed_dim))
        trunc_normal_(self.pe_time)
        trunc_normal_(self.pe_cand)

    def forward(self, feat_blur):
        # feat_blur: [B, C, H, W]
        B = feat_blur.shape[0]
        K, T, E = self.num_k, self.num_t, self.dim
        feat = rearrange(feat_blur, 'b c h w -> b (h w) c')
        feat = self.patch_emb(feat)

        ## get spatial features from each timestep
        res = []
        for unfolder in self.time_unfolder:
            x = feat
            for block in unfolder:
                x = block(x + self.pe_patch)
            res.append(x)

        ## extract candidate feature via a shared extractor
        mem =torch.cat(res, dim=0)    # [BT, N, E]
        tgt = self.candidate_token.expand(B*T, -1, -1)
        for block in self.extractor:
            tgt = block(tgt, mem + self.pe_patch)
        
        ## enhance all feats
        x = rearrange(tgt, '(b t) k e -> b (t k) e', t=T)
        pe = (self.pe_time + self.pe_cand).flatten(1, 2)
        for block in self.mixer:
            x = block(x + pe)

        x = self.out_proj(x)

        return rearrange(x, 'b (t k) e -> b k t e', t=T)

if __name__ == '__main__':
    B, T, K, E = 4, 3, 16, 256

    x = torch.randn([B, 2048, 8, 8]).cuda()
    model = Transformer().cuda()

    y = model(x)

    print(y.shape)
    
