import math
import torch
import torch.nn as nn

from timm.models.vision_transformer import Block
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

class Transformer_temporal(nn.Module):
    def __init__(self, num_k=16, num_timestep=3, num_joints = 21, 
                 patch_size = 8, in_chans = 2048, embed_dim=512, 
                 num_blocks=2, num_heads=4, 
                 mlp_ratio=4., drop_rate = 0.1, 
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_k = num_k
        self.num_t = num_timestep
        self.num_j = num_joints
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(img_size=patch_size, patch_size=1, in_chans=in_chans, embed_dim=embed_dim)

        patch_pe = get_position_embedding(patch_size=patch_size, embed_dim=embed_dim)  # [1, E, H, W]
        self.register_buffer('patch_pe', patch_pe.flatten(-2).transpose(1, 2))  # [1, HW, E]

        time_pe = get_timestep_embedding(num_timestep, embedding_dim=embed_dim)
        self.register_buffer('time_pe', time_pe.unsqueeze(1).unsqueeze(1))   # [T, 1, 1, E]

        self.temporal_blocks = nn.ModuleList([
            nn.ModuleList([
                Block(embed_dim, num_heads, mlp_ratio, qkv_bias=False, norm_layer=norm_layer, 
                  proj_drop=drop_rate, attn_drop=drop_rate, drop_path=drop_rate)
                for _ in range(num_blocks)
            ])
            for t in range(num_timestep)
        ])

        self.cls_tokens = nn.Parameter(torch.zeros(num_timestep, 1, 1, embed_dim))
        trunc_normal_(self.cls_tokens)

    def forward(self, feat_blur):
        # feat_blur: [B, C, H, W]

        B = feat_blur.shape[0]
        T, K, E = self.num_t, self.num_k, self.embed_dim
        feat_base = self.patch_embed(feat_blur) # [B, HW, E]

        # forwarding transformer block
        outputs = torch.zeros([T, B, E]).to(feat_base.device)
        for t, blocks in enumerate(self.temporal_blocks):
            x = torch.cat([self.cls_tokens[t].expand(B, -1, -1), feat_base], dim=1) # [B, 1+HW, E]
            pe = torch.cat([self.time_pe[t], self.patch_pe], dim=1) # [1, 1+HW, E]
            for block in blocks:
                x = block(x+pe)
            outputs[t] = x[:,0]
        
        return outputs.transpose(0, 1)  # [B, T, E]
    
class Transformer_candidate(nn.Module):
    def __init__(self, num_k=16, num_timestep=3, num_joints = 21, 
                 patch_size = 8, in_chans = 2048, embed_dim=512, 
                 num_blocks=2, num_heads=4, 
                 mlp_ratio=4., drop_rate = 0.1, 
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_k = num_k
        self.num_t = num_timestep
        self.num_j = num_joints
        self.embed_dim = embed_dim


        time_pe = get_timestep_embedding(num_timestep, embedding_dim=embed_dim)
        self.register_buffer('time_pe', time_pe.unsqueeze(0))   # [1, T, E]

        self.cand_blocks = nn.ModuleList([
            nn.ModuleList([
                Block(embed_dim, num_heads, mlp_ratio, qkv_bias=False, norm_layer=norm_layer, 
                  proj_drop=drop_rate, attn_drop=drop_rate, drop_path=drop_rate)
                for _ in range(num_blocks)
            ])
            for k in range(num_k)
        ])

    def forward(self, feat_temp):
        # feat_temp: [B, T, E]
        B, T, E = feat_temp.shape
        K = self.num_k
        outputs = torch.zeros([K, B, T, E]).to(feat_temp.device)
        for k, blocks in enumerate(self.cand_blocks):
            x = feat_temp
            for block in blocks:
                x = block(x+self.time_pe)
            outputs[k] = x
        
        return outputs.transpose(0, 1)  # [B, K, T, E]



if __name__ == '__main__':
    B, T, K, E = 4, 3, 16, 256

    x = torch.randn([B, 2048, 8, 8]).cuda()
    model_t = Transformer_temporal().cuda()
    model_k = Transformer_candidate().cuda()

    y = model_t(x)
    print(y.shape)
    y = model_k(y)
    print(y.shape)
    
