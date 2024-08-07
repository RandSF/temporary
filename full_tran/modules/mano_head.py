import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_
import numpy as np
import einops

from .pose_transformer import TransformerDecoder
import torchgeometry as tgm

DEFAULT_DICT = {
    'depth': 6,
    'heads': 8,
    'mlp_dim': 1024,
    'dim_head': 64,
    'dropout': 0.0,
    'context_dim': 1280 
}
DEFAULT_PATH = os.path.dirname(__file__) + "/mano_mean_params.npz"

def build_mano_head(cfg):
    mano_head_type = cfg['mano_head'].get('type', '_NO_TYPE_ASSIGNED')
    if  mano_head_type == 'transformer_decoder':
        return MANOTransformerDecoderHead(**cfg['mano_head'])
    else:
        raise ValueError('Unknown MANO head type: {}'.format(mano_head_type))

class MANOTransformerDecoderHead(nn.Module):
    """ Cross-attention based MANO Transformer decoder
    """

    def __init__(self, num_mano_joints = 16, num_timestep = 3, transformer_input = 'zero', 
                 transformer_decoder = DEFAULT_DICT, init_decoder_xavier = False, 
                 mean_params = DEFAULT_PATH, 
                 **kwargs):
        super().__init__()
        self.num_mano_joints = num_mano_joints
        self.num_timestep = num_timestep
        self.joint_rep_type = '6d'
        self.joint_rep_dim = {'6d': 6, 'aa': 3}[self.joint_rep_type]
        npose = self.joint_rep_dim * num_mano_joints
        self.npose = npose
        self.input_is_mean_shape = transformer_input == 'mean_shape'
        transformer_args = dict(
            num_tokens=num_timestep,
            token_dim=(npose + 10 + 3) if self.input_is_mean_shape else 1,
            dim=1024,
        )
        transformer_args.update(transformer_decoder)
        self.transformer = TransformerDecoder(
            **transformer_args
        )
        dim=transformer_args['dim']
        self.decpose = nn.Linear(dim, npose)
        self.decshape = nn.Linear(dim, 10)
        self.deccam = nn.Linear(dim, 3)

        if init_decoder_xavier:
            # True by default in MLP. False by default in Transformer
            nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
            nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
            nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        mean_params = np.load(os.path.dirname(__file__) + mean_params)
        init_hand_pose = torch.from_numpy(mean_params['pose'].astype(np.float32)).unsqueeze(0).unsqueeze(0)
        init_betas = torch.from_numpy(mean_params['shape'].astype('float32')).unsqueeze(0).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam'].astype(np.float32)).unsqueeze(0).unsqueeze(0)
        self.register_buffer('init_hand_pose', init_hand_pose)
        self.register_buffer('init_betas', init_betas)
        self.register_buffer('init_cam', init_cam)

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

    def forward(self, x, **kwargs):

        batch_size = x.shape[0]
        # vit pretrained backbone is channel-first. Change to token-first
        x = einops.rearrange(x, 'b c h w -> b (h w) c')

        init_hand_pose = self.init_hand_pose.expand(batch_size, -1, -1)
        init_betas = self.init_betas.expand(batch_size, -1, -1)
        init_cam = self.init_cam.expand(batch_size, -1, -1)

        # TODO: Convert init_hand_pose to aa rep if needed
        if self.joint_rep_type == 'aa':
            raise NotImplementedError

        pred_hand_pose = init_hand_pose
        pred_betas = init_betas
        pred_cam = init_cam
        # pred_hand_pose_list = []
        # pred_betas_list = []
        # pred_cam_list = []
        # for i in range(self.cfg.MODEL.MANO_HEAD.get('IEF_ITERS', 1)):
        for i in range(1):
            # Input token to transformer is zero token
            if self.input_is_mean_shape:
                token = torch.cat([pred_hand_pose, pred_betas, pred_cam], dim=1)[:,None,:]
            else:
                token = torch.zeros(batch_size, self.num_timestep, 1).to(x.device)

            # Pass through transformer
            token_out = self.transformer(token, context=x)  # [B, T, C]

            # Readout from token_out
            pred_hand_pose = self.decpose(token_out) + pred_hand_pose   # [B, T, J*6]
            pred_betas = self.decshape(token_out) + pred_betas          # [B, T, 10]
            pred_cam = self.deccam(token_out) + pred_cam                # [B, T, 3]

            # pred_hand_pose_list.append(pred_hand_pose)
            # pred_betas_list.append(pred_betas)
            # pred_cam_list.append(pred_cam)

        # Convert self.joint_rep_type -> aa
        joint_conversion_fn = {
            '6d': rot6d_to_axis_angle_batched,
            'aa': lambda x: x
        }[self.joint_rep_type]

        pred_hand_pose = joint_conversion_fn(
            pred_hand_pose.reshape(*pred_hand_pose.shape[:-1], self.num_mano_joints, 6)
        ).flatten(-2).unsqueeze(1)   # [B, 1, T, J*3]
        pred_betas = (torch.mean(pred_betas, dim=-2, keepdim=True)
                      .expand(-1, self.num_timestep, -1).unsqueeze(1)) # [B, 1, T, 10]
        pred_cam = pred_cam.unsqueeze(1) # [B, 1, T, 3]
        return pred_hand_pose, pred_betas, pred_cam


def rot6d_to_axis_angle_batched(x: torch.Tensor):
    # x: [B,( K,) T, J, 6]
    tensor_shape = x.shape[:-1]
    batch_size = tensor_shape.numel()

    x = x.view(-1,3,2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    rot_mat = torch.stack((b1, b2, b3), dim=-1) # 3x3 rotation matrix
    
    rot_mat = torch.cat([rot_mat,torch.zeros((batch_size,3,1)).to(rot_mat.device).float()],2) # 3x4 rotation matrix
    axis_angle = tgm.rotation_matrix_to_angle_axis(rot_mat).reshape(-1,3) # axis-angle
    axis_angle[torch.isnan(axis_angle)] = 0.0
    axis_angle = axis_angle.reshape(*tensor_shape, 3)
    return axis_angle
