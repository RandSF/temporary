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

    def forward(self, feat_blur, candidates, idx):
        # feat_blur [B, C, H, W]
        # candidates [T, K, B, ...]
        # idx: [K, B]

        cand_flatten = candidates.flatten(3)
        T, K, B = cand_flatten.shape[:-1]
        cand_input = self.enc_layer(cand_flatten.permute(2, 0, 1, 3).flatten(0, 1)) # [BT, K, E]
        ctx = self.ctx_layer(feat_blur).reshape(B, -1)  # [B, E]
        ctx_input = ctx.unsqueeze(1).repeat(T, K, 1)  # [BT, K, E]
        

        similarity = (F.normalize(cand_input, p=2, dim=-1)*F.normalize(ctx_input, p=2, dim=-1)).sum(dim=-1) # [BT, K]
        probs = torch.softmax(similarity, dim=-1)
        index = idx.transpose(0, 1).repeat(T, 1)[:, :self.num_ks]
        prob_pick = torch.gather(probs, dim=-1, index=index)    # [BT, Ks]
        
        loss_infonce = - torch.log(prob_pick).sum(dim=-1)

        # compute rank of the 1st candidates
        with torch.no_grad():
            rank_p = torch.argsort(
                            torch.argsort(probs, descending=True, dim=-1), 
                            descending=False, dim=-1
                        )
            rank = torch.gather(rank_p, dim=-1, index=index)    # [BT, Ks]

        return loss_infonce, rank.float().mean(dim=-1)
        