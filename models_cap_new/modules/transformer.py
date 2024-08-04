import torch
import torch.nn as nn
from einops import rearrange

from timm.models.vision_transformer import Block

class Transformer(nn.Module):
    def __init__(self, num_k = 16, num_timestep = 3, num_joints = 21, in_dims = 2048, embed_dim = 256, patch_size = 8, out_dim = 256, 
                 num_blocks = 4, num_heads = 4, drop_rate = 0.1, norm_layer=nn.LayerNorm, 
                 **kwargs):
        super().__init__()
        self.num_joints = num_joints
        self.num_k = num_k

        self.patch_embed = nn.Linear(embed_dim, embed_dim)   # maybe freeze as MOCO?
        self.pe_time = nn.Parameter(torch.zeros(1, 1, 3, 1, embed_dim))  # time direction
        self.pe_joints = nn.Parameter(torch.zeros(1, 1, 1, self.num_joints, embed_dim))  # joint direction
        self.token_cand = nn.Parameter(torch.zeros(1, num_k, 1, 1, embed_dim))
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio=4.0, qkv_bias=False, norm_layer=norm_layer)
            for _ in range(num_blocks)])

    def forward(self, feat_joint):
        # feat_joint: [B, T, J, E]
        B, T, J, E = feat_joint.shape
        K = self.num_k
        # forwarding transformer block
        x = self.patch_embed(feat_joint).unsqueeze(1)   # [B, 1, T, J, E]
        x = x + (self.pe_time + self.pe_joints + self.token_cand)   # [B, K, T, J, E]
        x = rearrange(x, 'b k t j e -> (bk) (tj) e')
        for blk in self.blocks:
            x = blk(x)
            
        feat_joint_out = rearrange(x, '(bk) (tj) e -> b k t j e', b=B, k=K, t=T, j=J)
        
        return feat_joint_out
