import torch
import torch.nn as nn

from models_pose.modules.layer_utils import make_linear_layers, make_conv_layers
from utils.MANO import mano
from utils.transforms import rot6d_to_axis_angle


class Regressor(nn.Module):
    def __init__(self, in_chans=2048, embed_dim = 256, **kwargs):
        super().__init__()

        self.blur_layer = make_conv_layers([in_chans, embed_dim], kernel=8, stride=1, padding=0, bnrelu_final=False)
        # mano shape regression, multiply the output channel by 3 to account e1, m, and e2
        self.shape_out = make_linear_layers([embed_dim, mano.shape_param_dim * 3], relu_final=False)
        # camera parameter regression for projection loss, multiply 3 to account e1, m, and e2
        self.cam_out = make_linear_layers([embed_dim, 3 * 3], relu_final=False)
        
        # mano pose regression, predict 6d pose
        self.joint_regs = make_linear_layers([embed_dim, embed_dim, 6*16], relu_final=False)
            
        

    def forward(self, feat_blur, feat_joint):
        # feat_joint [B, K, T, E]
        # mano shape parameter regression
        B, K, T, E = feat_joint.shape
        feat_pool = self.blur_layer(feat_blur).reshape(B, E)
        shape_param = self.shape_out(feat_pool) # [B, T\beta]
        mano_shape = shape_param.reshape(B, T, 10).unsqueeze(1).expand(B, K, T, 10)
        # camera parameter regression
        camera = self.cam_out(feat_pool) # [B, T*3]
        cam_param = camera.reshape(B, T, 3).unsqueeze(1).expand(B, K, T, 3)
        
        # mano pose parameter regression
        feat_one = feat_joint.flatten(0, 2) # [BKT, E]

        pose_6d = self.joint_regs(feat_one) # [BKT, 6*16]
        pose_6d = pose_6d.reshape(-1, 16, 6).flatten(0, 1)   # [BKTJ', 6]
        
        # change 6d pose -> axis angles
        mano_pose = rot6d_to_axis_angle(pose_6d).reshape(B, K, T, mano.orig_joint_num * 3)
        

        return mano_shape.flatten(0, 2), mano_pose.flatten(0, 2), cam_param.flatten(0, 2)
