import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple

class FCHead(nn.Module):

    def __init__(self, num_timestep, 
                       context_features, 
                       hidden_features, 
                       param_path):
        """
        Fully connected head for camera and betas regression.
        Args:
            cfg (CfgNode): Model config as yacs CfgNode.
        """
        super(FCHead, self).__init__()
        self.layers = nn.Sequential(nn.Linear(context_features, hidden_features),
                                              nn.ReLU(inplace=False),
                                              nn.Linear(hidden_features, 13))
        nn.init.xavier_uniform_(self.layers[2].weight, gain=0.02)

        mean_params = np.load(param_path)
        init_cam = torch.from_numpy(mean_params['cam'].astype(np.float32))[None, None]  # [1, 1, 3]
        init_betas = torch.from_numpy(mean_params['shape'].astype(np.float32))[None, None]  # [1, 1, 10]

        self.register_buffer('init_cam', init_cam)
        self.register_buffer('init_betas', init_betas)

    def forward(self, feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        feats: [BT, JE]
        """

        batch_size = feats.shape[0]

        offset = self.layers(feats)
        betas_offset = offset[:, :10]
        cam_offset = offset[:, 10:]
        pred_cam = cam_offset + self.init_cam
        pred_betas = betas_offset + self.init_betas

        return pred_betas, pred_cam
