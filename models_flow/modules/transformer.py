import torch
import torch.nn as nn
from einops import rearrange

from timm.models.vision_transformer import Block
from timm.layers import trunc_normal_

class Transformer(nn.Module):
    def __init__(self, num_k = 16, num_timestep = 3, num_joints = 21, in_dims = 2048, embed_dim = 256, patch_size = 8, out_dim = 256, 
                 num_blocks = 4, num_heads = 4, drop_rate = 0.1, norm_layer=nn.LayerNorm, 
                 **kwargs):
        super().__init__()
        self.num_joints = num_joints
        self.num_k = num_k

        self.patch_embed = nn.Linear(embed_dim, embed_dim)   # maybe freeze as MOCO?
        self.pe_time = nn.Parameter(torch.zeros(1, num_timestep, 1, embed_dim))  # time direction
        self.pe_joints = nn.Parameter(torch.zeros(1, 1, self.num_joints, embed_dim))  # joint direction
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio=4.0, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(num_blocks)])
        
        trunc_normal_(self.pe_time)
        trunc_normal_(self.pe_joints)

    def forward(self, feat_joint):
        # feat_joint: [B, T, J, E]
        B, T, J, E = feat_joint.shape
        # forwarding transformer block
        x = self.patch_embed(feat_joint)   # [B, T, J, E]
        x = x + (self.pe_time + self.pe_joints)   # [B, T, J, E]
        x = rearrange(x, 'b t j e -> b (t j) e')
        for blk in self.blocks:
            x = blk(x)
            
        feat_joint_out = rearrange(x, 'b (t j) e -> b t j e', b=B, t=T, j=J)
        
        return feat_joint_out
