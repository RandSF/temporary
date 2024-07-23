import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCE(nn.Module):
    def __init__(self, 
                 num_k_select, 
                 embed_dim = 512,
                 in_chans = 2048,
                 in_dim = 512, 
                 patch_size = 8,
                 **kwargs) -> None:
        super().__init__()
        self.num_ks = num_k_select

        self.ctx_layer = nn.Conv2d(in_chans, embed_dim, 8, 1, 0)
        self.enc_layer = nn.Linear(in_dim, embed_dim)

    def forward(self, feat_blur, candidates):
        # feat_blur [B, C, H, W]
        # candidates [T, K, B, ...]

        cand_flatten = candidates.flatten(3)
        T, K, B = cand_flatten.shape[:-1]
        cand_input = self.enc_layer(cand_flatten.permute(2, 0, 1, 3).flatten(0, 1)) # [BT, K, E]
        ctx = self.ctx_layer(feat_blur).reshape(B, -1)  # [B, E]
        ctx_input = ctx.unsqueeze(1).repeat(T, K, 1)  # [BT, K, E]
        

        similarity = (F.normalize(cand_input, p=2, dim=-1)*F.normalize(ctx_input, p=2, dim=-1)).sum(dim=-1) # [BT, K]

        loss_infonce = - torch.log(torch.softmax(similarity, dim=-1)[:, :self.num_ks]).sum(dim=-1)

        return loss_infonce
        