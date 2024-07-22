import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block

from functools import partial
from typing import Tuple
import math
from timm.models.vision_transformer import Mlp, PatchEmbed
from timm.models.layers import trunc_normal_
def get_timestep_embedding(
    timesteps: torch.Tensor,
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

def get_position_embedding_sine_2d(patch_size=8, num_feats=768//2, temperature=10000):
    scale = 2*torch.tensor(torch.pi)
    not_mask = torch.ones((1, patch_size, patch_size), dtype=torch.float32)

    y_embed = not_mask.cumsum(1, dtype=torch.float32)
    x_embed = not_mask.cumsum(2, dtype=torch.float32)

    eps = 1e-6
    y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
    x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

    dim_t = torch.arange(num_feats, dtype=torch.float32)
    dim_t = temperature ** (2 * (dim_t // 2) / num_feats)

    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t
    pos_x = torch.stack(
        (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
    ).flatten(3)
    pos_y = torch.stack(
        (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
    ).flatten(3)
    pos = torch.cat((pos_y, pos_x), dim=3)  # [1, H, W, E]
    return pos.flatten(1, 2)

class Block(nn.Module):
    def __init__(self, embed_dim=512, mlp_ratio = 4, num_heads = 4, dropout_rate = 0.1) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.sa = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout_rate, batch_first=True)
        self.drop1 = nn.Dropout(p=dropout_rate)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ca = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout_rate, batch_first=True)
        self.drop2 = nn.Dropout(p=dropout_rate)
        self.norm3 = nn.LayerNorm(embed_dim)
        # self.ffn1 = nn.Linear(embed_dim, int(mlp_ratio*embed_dim))
        # self.drop = nn.Dropout(p=dropout_rate)
        # self.ffn2 = nn.Linear(int(mlp_ratio*embed_dim), embed_dim)
        # self.drop3 = nn.Dropout(p=dropout_rate)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, int(mlp_ratio*embed_dim)), 
            nn.LeakyReLU(), 
            nn.Dropout(p=dropout_rate), 
            nn.Linear(int(mlp_ratio*embed_dim), embed_dim)
        )
        self.drop3 = nn.Dropout(p=dropout_rate)

    def forward(self, tgt: torch.Tensor, mem: torch.Tensor, 
                pe_tgt = None, pe_mem = None):
        # we do not use mask in the task
        pe_tgt = 0 if pe_tgt is None else pe_tgt
        pe_mem = 0 if pe_mem is None else pe_mem

        x = self.norm1(tgt)
        x2 = self.sa(query = x + pe_tgt, key = x + pe_tgt, value = x)[0]
        x = x + self.drop1(x2)

        x2 = self.norm2(x)
        x2 = self.ca(query = x2 + pe_tgt, key = mem + pe_mem, value = mem)[0]
        x = x + self.drop2(x2)

        x2 = self.norm3(x)
        x2 = self.ffn(x2)
        x = x + self.drop3(x2)

        return x


class RewardModel(nn.Module):
    def __init__(self,
                 num_k = 16, 
                 num_timestep = 3, 
                 embed_dim = 512,
                 patch_size = 8,
                 input_embed = 512,
                 in_chans = 2048,
                 num_blocks = 8,
                 num_heads = 4, 
                 drop_rate = 0.2, 
                 mlp_ratio = 4.0,
                 **kwargs) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        self.base_embed = PatchEmbed(img_size=patch_size, patch_size=1, in_chans=in_chans, embed_dim=embed_dim)
        self.pose_embed = Mlp(in_features=48, hidden_features=embed_dim, out_features=embed_dim)

        pe_time = get_timestep_embedding(torch.arange(num_timestep), embed_dim)
        pe_base = get_position_embedding_sine_2d(patch_size, embed_dim//2)
        self.register_buffer('pe_time', pe_time.unsqueeze(0).unsqueeze(0))  # [1, 1, T, E]
        self.register_buffer('pe_base', pe_base)   # [1, HW, E]
        self.pe_score = nn.Parameter(torch.zeros(1, num_k, 1, embed_dim))
        trunc_normal_(self.pe_score, std=0.02)

        self.blocks =  nn.ModuleList([
            Block(embed_dim = embed_dim, num_heads = num_heads, mlp_ratio = mlp_ratio, dropout_rate = drop_rate)
            for i in range(num_blocks)])

        self.score_head = nn.Sequential(
            nn.Linear(embed_dim, int(mlp_ratio*embed_dim)),
            nn.ReLU(),
            nn.Linear(int(mlp_ratio*embed_dim), 2), # for JRC
        )

        

    def forward(self, img, feat_blur, pose):
        '''
        feat_blur: [B, C, H, W]
        pose: [T, B, K, theta]
        '''
        T, B, K, _ = pose.shape
        E = self.embed_dim

        x_blur = self.base_embed(feat_blur) # [B, HW, E]

        pose = pose.permute(1, 2, 0, 3)  # [B, K, T, theta]
        x_pose = self.pose_embed(pose).flatten(1, 2)      # [B, KT, E]


        pe_tgt = (self.pe_score + self.pe_time).flatten(1, 2)   # [1, KT, E]
        pe_mem = self.pe_base   # [1, HW, E]
        x = x_pose
        for block in self.blocks:
            x = block(tgt = x, mem=x_blur, 
                        pe_tgt = pe_tgt , pe_mem = pe_mem)

        score_feature = x   # [B, KT, E]
        score_feature = score_feature.reshape(B, K, T, E).flatten(0, 1)

        score = self.score_head(score_feature).reshape(B, K, T, 2)  # [B, K, T, 2]

        return score.sum(2)    # [B, K, 2]


if __name__ == '__main__':
    B, T, K = 8, 3, 16
    rm = RewardModel().cuda()
    featb = torch.randn([B, 2048, 8, 8]).cuda()
    pose = torch.randn([T, B, K, 48]).cuda()

    res = rm(None, featb, pose)
    print(res.shape)