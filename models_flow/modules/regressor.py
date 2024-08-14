from typing import Optional, Dict, Tuple
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchgeometry as tgm
from einops import rearrange

from .heads.glow import ConditionalGlow
from .heads.fc_head import FCHead
from utils.transforms import rot6d_to_axis_angle



NUM_MANO_JOINT = 16
class Regressor(nn.Module):
    def __init__(self, 
                 num_k = 4,
                 num_joint = 21,
                 num_timestep = 3,
                 input_dim = 256, 
                 hiddens = 1024, 
                 num_layer = 4,
                 num_block = 2,
                 mano_param_path = '/heads/mano_mean_params.npz', 
                 **kwargs):
        super().__init__()
        self.flow = ConditionalGlow(
            features = NUM_MANO_JOINT*6, 
            hidden_features = hiddens, 
            num_blocks_per_layer = num_block,
            num_layers = num_layer,
            context_features = input_dim*num_joint, 
        )
        self.global_head = FCHead(num_timestep = num_timestep, 
                                  context_features = input_dim*num_joint, 
                                  hidden_features = hiddens,
                                  param_path = os.path.dirname(__file__) + mano_param_path)

        self.num_k = num_k  # number of samples
        self.num_t = num_timestep
        self.num_j = num_joint

    def forward(self, feat_mano: torch.Tensor, only_mode = False) -> Dict:
        '''
        feat_mano: [B, T, J, E]
        '''
        B, T, J, E = feat_mano.shape
        K = self.num_k
        ctx = feat_mano.reshape(B*T, J*E)
        # sample the mode of the distribution
        mu = torch.zeros([B*T, 1, NUM_MANO_JOINT*6], device=feat_mano.device)
        samples_mode, log_prob_mode, mu = self.flow.sample_and_log_prob(num_samples=1, context=ctx, noise=mu) #! check the return values

        # special case that K=1
        shape_mode, pose_mode, camera_mode = self._get_predictio_and_reshape(samples_mode.reshape(-1, 6), ctx, 
                                                                                            B=B, T=T, K=1)
        pred_mode = {'shape': shape_mode, 'pose': pose_mode, 'camera': camera_mode, 'logp': log_prob_mode, 'z': mu}
        if only_mode:   
            return {'mode': pred_mode}
        
        samples, log_prob, z = self.flow.sample_and_log_prob(num_samples=K, context=ctx)
        shape, pose, camera = self._get_predictio_and_reshape(samples.reshape(-1, 6), ctx,
                                                              B=B, T=T, K=K)
        pred_sample = {'shape': shape, 'pose': pose, 'camera': camera, 'logp': log_prob, 'z': z}
        return {
            'sample': pred_sample,
            'mode': pred_mode
        }
        
    def log_prob(self, feat_mano: torch.Tensor, pose: torch.Tensor) -> Tuple:
        """
        Compute the log-probability of a set of smpl_params given a batch of images.
        pose: [B, T, K, \Theta]
        feat_mano: [B, T, J, E]
        """
        B, T, K, _ = pose.shape
        feat_ctx = feat_mano.reshape(B*T, 1, -1).expand(-1, K, -1).flatten(0, 1)   # [BTK, JE]
        samples = pose.flatten(0, 2)    # [BTK, \Theta]
        log_prob, z = self.flow.log_prob(samples, feat_ctx)
        log_prob = log_prob.reshape(B, T, K)
        z = z.reshape(B, T, K, -1)
        return log_prob, z

    def _get_predictio_and_reshape(self, samples: torch.Tensor, context: torch.Tensor, B, T, K):
        '''
        samples: [BTK*16, 6]
        context: [BT, JE]
        '''
        axis_angle = rot6d_to_axis_angle(samples)
        shape, camera = self.global_head(context)
        shape = shape.unsqueeze(1).expand(-1, K, -1).reshape(B, T, 1, 10)
        pose = axis_angle.reshape(B, T, K, 48)
        camera = camera.unsqueeze(1).expand(-1, K, -1).reshape(B, T, 1, 3)

        return shape, pose, camera



