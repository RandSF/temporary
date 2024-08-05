import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.layer_utils import make_linear_layers
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
    def __init__(self, num_joints = 21, in_dim=256, **kwargs):
        super().__init__()
        self.num_joints = num_joints
        self.orig_joint_num = mano.orig_joint_num
        self.joints_reg_p = nn.Linear(in_dim, self.orig_joint_num*6)
        # self.joints_reg_c = nn.Linear(in_dim, self.orig_joint_num*6)
        # self.joints_reg_f = nn.Linear(in_dim, self.orig_joint_num*6)
        # self.fw_reg = nn.Linear(in_dim + num_joints*3, self.orig_joint_num*6)
        # self.bw_reg = nn.Linear(in_dim + num_joints*3, self.orig_joint_num*6)
            
        #TODO use temporal cross attention?
        # # camera parameter regression for projection loss, the same param for account e1, m, and e2
        # self.cam_out = make_linear_layers([in_dim*3, 3*3], relu_final=False)
        self.cam_out = make_linear_layers([2048, 3*3], relu_final=False)
        self.shape_out = make_linear_layers([in_dim*3, mano.shape_param_dim], relu_final=False)

    def forward(self, feat_joint, feat_blur):
        # feat_joint: [B, K, T, C]
        # mano pose parameter regression
        B, K, T, C = feat_joint.shape
        J = self.num_joints

        feat_one = feat_joint.flatten(0, 1)     # [BK, T, C]
        pose_6d_flatten = torch.stack([
            self.joints_reg_p(feat_one[:, 0]), 
            self.joints_reg_p(feat_one[:, 1]), 
            self.joints_reg_p(feat_one[:, 2])
        ], dim=1)    # [BK, T, 16*6]
        pose_6d = pose_6d_flatten.reshape(B*K*T*self.orig_joint_num, 6)

        # dyna_fw_flatten = self.fw_reg(feat_one)
        # dyna_bw_flatten = self.fw_reg(feat_one)
        # pose_6d_fusion = self.predict_with_dynamics(pose_6d_flatten, dyna_fw_flatten, dyna_bw_flatten, B, T)
        # pose_6d = pose_6d_fusion.reshape(B*T*self.orig_joint_num, 6)

        # change 6d pose -> axis angles
        mano_pose = rot6d_to_axis_angle(pose_6d).reshape(B*K*T, self.orig_joint_num * 3)     # [B*K*T, \theta]

        feat_pool = feat_joint.reshape(B*K, T*C)  # [B*K, T*C]
        mano_shape = self.shape_out(feat_pool).unsqueeze(1).expand(B*K, T, 10)  # [B*K, T, 10]
        # camera = self.cam_out(feat_pool)    # [B*K, T*3]
        # camera parameter regression
        camera = self.cam_out(feat_blur.mean((2,3)))    # [B, T*3]
        camera = camera.unsqueeze(1).reshape(B, 1, T, 3)
        # cam_param_e1, cam_param_md, cam_param_e2 = torch.split(camera, 3, dim=1)

        return mano_shape.reshape(B, K, T, 10), mano_pose.reshape(B, K, T, 48), camera.expand(B, K, T, 3)
    
    def predict_with_dynamics(self, pred_pose, pred_fw, pred_bw, B, T, confidence = None):
        """
        shape: 
            pred_pose, pred_fw, pred_bw: [B*T, 16*6]
            #?confidence: [B*T, 3]
        
        """
        pred_pose_t = torch.reshape(pred_pose, [B, T, -1])
        pred_pose_expand = pred_pose_t.unsqueeze(2).repeat(1, 1, T, 1)    # [B, T, T, 16*6]

        pred_fw_t = torch.reshape(pred_fw, [B, T, -1])
        pred_bw_t = torch.reshape(pred_bw, [B, T, -1])
        mat_fw_orig = torch.zeros([B, T, T, self.orig_joint_num*6]).cuda()
        mat_bw_orig = torch.zeros([B, T, T, self.orig_joint_num*6]).cuda()
        for t in range(T):
            if t < T-1:
                # forward matrix
                col_fw = t+1
                row_fw = slice(0, t+1)
                mat_fw_orig[:, row_fw, col_fw] = pred_fw_t[:, t].unsqueeze(1)
            if t > 0:
                # backward matrix
                col_bw = t-1
                row_bw = slice(t, T)
                mat_bw_orig[:, row_bw, col_bw] = pred_bw_t[:, t].unsqueeze(1)
        mat_fw_cum = mat_fw_orig.cumsum(2)
        mat_bw_cum = mat_bw_orig.flip(2).cumsum(2).flip(2)
        mat_fw = mat_fw_cum
        mat_bw = mat_bw_cum

        mask_fw = (torch.ones_like(mat_fw)  # [B, T, T, 16*6]
                   .permute(0, 3, 1, 2)     # [B, 16*6, T, T]
                   .flatten(0 ,1)           # [B*16*6, T, T]
                   .triu_(1)                # [B*16*6, T, T]
                   .reshape(B, -1, T, T)    # [B, 16*6, T, T]
                   .permute(0, 2, 3, 1))    # [B, T, T, 16*6]
        mask_bw = (torch.ones_like(mat_bw)  # [B, T, T, 16*6]
                   .permute(0, 3, 1, 2)     # [B, 16*6, T, T]
                   .flatten(0 ,1)           # [B*16*6, T, T]
                   .tril_(-1)               # [B*16*6, T, T]
                   .reshape(B, -1, T, T)    # [B, 16*6, T, T]
                   .permute(0, 2, 3, 1))    # [B, T, T, 16*6]

        pred_pose_with_fw = ((pred_pose_expand + mat_fw) * mask_fw)
        pred_pose_with_bw = (pred_pose_expand + mat_bw) * mask_bw
        pred_pose_current = pred_pose_expand * (1-mask_fw)*(1-mask_bw)  # equivalent to diagonal of shape [B, T, T, 16*6] on dim (1, 2)

        if confidence is None:
            output_pose = torch.mean(
                pred_pose_with_fw + pred_pose_current + pred_pose_with_bw, 
                dim=1)  # [B, T, -1]
        else:
            raise NotImplementedError
            weight = F.softmax

        return output_pose.flatten(0, 1)    # [B*T, -1], the same as input