import torch
import torch.nn as nn

from models_pose.modules.layer_utils import make_linear_layers
from utils.MANO import mano
from utils.transforms import rot6d_to_axis_angle

# for applying KTD
ANCESTOR_INDEX = [
    [],  # Wrist
    [0],  # Index_1
    [0,1],  # Index_2
    [0,1,2],  # index_3
    [0],  # Middle_1
    [0,4],  # Middle_2
    [0,4,5],  # Middle_3
    [0],  # Pinky_1
    [0,7],  # Pinky_2
    [0,7,8],  #Pinky_3
    [0],  # Ring_1
    [0,10],  # Ring_2
    [0,10,11],  # Ring_3
    [0],  # Thumb_1
    [0,13],  # Thumb_2
    [0,13,14]  # Thumb_3
]


class Regressor(nn.Module):
    def __init__(self, opt_params, in_chans=2048, in_chans_pose=512):
        super().__init__()
        # mano shape regression, multiply the output channel by 3 to account e1, m, and e2
        self.shape_out = make_linear_layers([in_chans, mano.shape_param_dim * 3], relu_final=False)
        # camera parameter regression for projection loss, multiply 3 to account e1, m, and e2
        self.cam_out = make_linear_layers([in_chans, 3 * 3], relu_final=False)
        
        # mano pose regression, apply KTD using ancestral pose parameters
        self.joint_regs = nn.ModuleList()
        for ancestor_idx in ANCESTOR_INDEX:
            regressor = nn.Linear(in_chans_pose, 6)
            self.joint_regs.append(regressor)
            
        

    def forward(self, feat_blur, feat_joint):
        # feat_joint [B, K, T, J, E]
        # mano shape parameter regression
        B, K, T, J, _ = feat_joint.shape
        feat_pool = feat_blur.mean((2,3))
        shape_param = self.shape_out(feat_pool) # [B, T\beta]
        mano_shape = shape_param.reshape(B, T, 10).unsqueeze(1).expand(B, K, T, 10)
        # camera parameter regression
        camera = self.cam_out(feat_pool) # [B, T*3]
        cam_param = camera.reshape(B, T, 3).unsqueeze(1).expand(B, K, T, 3)
        
        # mano pose parameter regression
        batch_size = feat_blur.shape[0]
        feat_one = feat_joint.flatten(0, 2).transpose(0, 1) # [J, BKT, E]

        pose_6d_list = []

        # regression using KTD
        for feat, reg in zip(feat_one, self.joint_regs):
            pose_6d_list.append(reg(feat))
            
        pose_6d = torch.stack(pose_6d_list, dim=1).flatten(0, 1)   # [BKTJ', 6]
        
        # change 6d pose -> axis angles
        mano_pose = rot6d_to_axis_angle(pose_6d).reshape(B, K, T, mano.orig_joint_num * 3)
        

        return mano_shape, mano_pose, cam_param
