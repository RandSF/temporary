from full_tran.modules.hamer import HAMER
from models_deformer.modules.layer_utils import init_weights
from losses import CoordLoss, ParamLoss, CoordLossOrderInvariant, DiversityLoss
from utils.MANO import mano

import math
import copy
import torch
from torch import nn

class BlurHandNet(nn.Module):
    def __init__(self, opt, weight_init=True):
        super().__init__()
        # define trainable module
        opt_net = opt['network']
        self.hamer = HAMER(opt_net)

        # weight initialization
        if weight_init:
            self.hamer.apply(init_weights)

        # for producing 3d hand meshs
        self.mano_layer_right = copy.deepcopy(mano.layer['right']).cuda()
        self.mano_layer_left = copy.deepcopy(mano.layer['left']).cuda()

        # losses
        self.coord_loss = CoordLoss()
        self.coord_loss_order_invariant = CoordLossOrderInvariant()
        self.param_loss = ParamLoss()
        # self.d_loss = DiversityLoss()
        
        # parameters
        self.opt_params = opt['task_parameters']
        if opt.get('train', False):
            self.opt_loss = opt['train']['loss']

        self.num_k = max(1, opt['task_parameters']['num_k'])
        self.num_k_use = max(1, opt['task_parameters']['num_k_select'])
        self.n_joints = opt['task_parameters']['num_joints']
        self.n_time = opt['task_parameters']['num_timestep']

    def forward(self, inputs, targets, meta_info, mode):
        """
        shape:
            B: batch size
            T: timestep
            C: channel
            H: height
            W: weight
            J: number of joint (21, not MANO joint)
        """
        # extract feature from backbone
        img = inputs['img'] # [B, 3, 256, 256]
        mano_pose, mano_shape, cam_param = self.hamer(img) # [B, K, T, ...]
        B = img.shape[0]
        K, T, J = self.num_k, self.n_time, self.n_joints

        cam_tran = self.get_camera_trans(cam_param)   # [B, K, T, 3]

        # reshape 
        joint_proj, joint_cam, mesh_cam = \
            self.get_coord(mano_pose[...,:3], mano_pose[...,3:], mano_shape, cam_tran) # [B, K, T, ...]

        if mode == 'train':
            loss = {}

            joint_cam = joint_cam.reshape(B, K, T, J, 3).transpose(0, 2)    # [T, K, B, ...]
            joint_proj = joint_proj.reshape(B, K, T, J, 2).transpose(0, 2)
            mano_pose = mano_pose.transpose(0, 2)
            mano_shape = mano_shape.transpose(0, 2)

            # # select the good candidates
            # K_use = max(1, self.num_k_use)
            # if K > K_use:
            #     with torch.no_grad():
            #         loss_joint_cam_md = self.coord_loss(joint_cam[1], targets['joint_cam'][None], meta_info['joint_valid'][None] * meta_info['is_3D'][None,:,None,None])
            #         loss_joint_cam = loss_joint_cam_md.mean(dim=[-1, -2])# + loss_joint_cam_pf   # [K, B]
            #         idx = torch.argsort(loss_joint_cam, dim=0)  # [K, B]
            #         idx = idx.unsqueeze(0).expand(T, K, B)  # [T, K, B]

            #     # joint_img = torch.gather(joint_img, index=idx[...,None,None].expand(T, K, B, J, 3), dim=1)[:,:K_use]
            #     joint_cam = torch.gather(joint_cam, index=idx[...,None,None].expand(T, K, B, J, 3), dim=1)[:,:K_use]
            #     joint_proj = torch.gather(joint_proj, index=idx[...,None,None].expand(T, K, B, J, 2), dim=1)[:,:K_use]
            #     mano_pose = torch.gather(mano_pose, index=idx[...,None].expand(T, K, B, mano.orig_joint_num*3), dim=1)[:,:K_use]
            #     mano_shape = torch.gather(mano_shape, index=idx[...,None].expand(T, K, B, mano.shape_param_dim), dim=1)[:,:K_use]


            # losses on middle hand; we do not have to consider "order"
            IDX_MD = 1
            # loss['joint_img'] = self.opt_loss['lambda_joint_img'] * self.coord_loss(joint_img[IDX_MD], targets['joint_img'][None], meta_info['joint_trunc'][None], meta_info['is_3D'][None,:,None,None])
            loss['Loss/joint_proj'] = self.opt_loss['lambda_joint_proj'] * self.coord_loss(joint_proj[IDX_MD], targets['joint_img'][:,:,:2][None], meta_info['joint_trunc'][None])
            loss['Loss/joint_cam'] = self.opt_loss['lambda_joint_cam'] * self.coord_loss(joint_cam[IDX_MD], targets['joint_cam'][None], meta_info['joint_valid'][None] * meta_info['is_3D'][None,:,None,None])
            loss['Loss/mano_joint_cam'] = self.opt_loss['lambda_joint_cam'] * self.coord_loss(joint_cam[IDX_MD], targets['mano_joint_cam'][None], meta_info['mano_joint_valid'][None])
            loss['Loss/mano_pose'] = self.param_loss(mano_pose[IDX_MD], targets['mano_pose'][None], meta_info['mano_pose_valid'][None])
            loss['Loss/mano_shape'] = self.param_loss(mano_shape[IDX_MD], targets['mano_shape'][None], meta_info['mano_shape_valid'][None,:,None]) #/ T
            
            # losses on hands in both ends
            IDX_P, IDX_F = 0, -1
            loss['Loss/joint_proj_past'] = self.opt_loss['lambda_joint_proj'] * self.coord_loss(joint_proj[IDX_P], targets['joint_img_past'][:,:,:2][None], meta_info['joint_trunc_past'][None])
            loss['Loss/joint_proj_future'] = self.opt_loss['lambda_joint_proj'] * self.coord_loss(joint_proj[IDX_F], targets['joint_img_future'][:,:,:2][None], meta_info['joint_trunc_future'][None])
            
            loss['Loss/joint_cam_past'] = self.opt_loss['lambda_joint_cam'] * self.coord_loss(joint_cam[IDX_P], targets['joint_cam_past'][None], meta_info['joint_valid_past'][None])
            loss['Loss/joint_cam_future'] = self.opt_loss['lambda_joint_cam'] * self.coord_loss(joint_cam[IDX_F], targets['joint_cam_future'][None], meta_info['joint_valid_future'][None])
            loss['Loss/mano_joint_cam_past'] = self.opt_loss['lambda_joint_cam'] * self.coord_loss(joint_cam[IDX_P], targets['mano_joint_cam_past'][None], meta_info['mano_joint_valid_past'][None])
            loss['Loss/mano_joint_cam_future'] = self.opt_loss['lambda_joint_cam'] * self.coord_loss(joint_cam[IDX_F], targets['mano_joint_cam_future'][None], meta_info['mano_joint_valid_future'][None])
            
            loss['Loss/mano_pose_past'] = self.param_loss(mano_pose[IDX_P], targets['mano_pose_past'][None], meta_info['mano_pose_valid_past'][None])
            loss['Loss/mano_pose_future'] = self.param_loss(mano_pose[IDX_F], targets['mano_pose_future'][None], meta_info['mano_pose_valid_future'][None])
            
            loss['Loss/mano_shape_past'] = self.param_loss(mano_shape[IDX_P], targets['mano_shape_past'][None], meta_info['mano_shape_valid_past'][None, :,None]) #/ T
            loss['Loss/mano_shape_future'] = self.param_loss(mano_shape[IDX_F], targets['mano_shape_future'][None], meta_info['mano_shape_valid_future'][None, :,None]) #/ T
        
            # diversity promoting loss
            # loss['diversity'] = self.opt_loss['lambda_diversity'] * self.d_loss(feat_mano)

            return loss
        else:
            out = {}
            out['img'] = inputs['img']
            
            # our model predictions
            mesh_cam = mesh_cam.reshape(B, K, T, 778, 3).permute(2, 0, 1, 3, 4)
            # when evaluating hands in both ends, MPJPE will be calculated in order of minimizing the value
            out['mano_mesh_cam'] = mesh_cam[1]
            out['mano_mesh_cam_past'] = mesh_cam[0]
            out['mano_mesh_cam_future'] = mesh_cam[2]

            # ground-truth mano coordinate
            with torch.no_grad():
                batch_size = inputs['img'].shape[0]
                mesh_coord_cam_gt = torch.zeros((batch_size, mano.vertex_num, 3)).cuda()

                pose_param_right = targets['mano_pose'][meta_info['hand_type']==1]
                shape_param_right = targets['mano_shape'][meta_info['hand_type']==1]
                
                if pose_param_right.shape[0] != 0:
                    mano_output_right_gt = self.mano_layer_right(global_orient=pose_param_right[:,:3], hand_pose=pose_param_right[:,3:], betas=shape_param_right)
                    mesh_coord_cam_right_gt = mano_output_right_gt.vertices
                    mesh_coord_cam_right_gt -= mano_output_right_gt.joints[:,0,:][:,None,:]
                    mesh_coord_cam_gt[meta_info['hand_type']==1] = mesh_coord_cam_right_gt

                pose_param_left = targets['mano_pose'][meta_info['hand_type']==0]
                shape_param_left = targets['mano_shape'][meta_info['hand_type']==0]
                
                if pose_param_left.shape[0] != 0:
                    mano_output_left_gt = self.mano_layer_left(global_orient=pose_param_left[:,:3], hand_pose=pose_param_left[:,3:], betas=shape_param_left)
                    mesh_coord_cam_left_gt = mano_output_left_gt.vertices
                    mesh_coord_cam_left_gt -= mano_output_left_gt.joints[:,0,:][:,None,:]
                    mesh_coord_cam_gt[meta_info['hand_type']==0] = mesh_coord_cam_left_gt
                    
                out['mesh_coord_cam_gt'] = mesh_coord_cam_gt

            if 'bb2img_trans' in meta_info:
                out['bb2img_trans'] = meta_info['bb2img_trans']
            if 'mano_mesh_cam' in targets:
                out['mano_mesh_cam_target'] = targets['mano_mesh_cam']
                
            return out

    def get_camera_trans(self, cam_param):
        # camera translation
        t_xy = cam_param[...,:2]
        gamma = torch.sigmoid(cam_param[...,2]) # apply sigmoid to make it positive
        k_value = torch.FloatTensor([math.sqrt(self.opt_params['focal'][0] * self.opt_params['focal'][1] * self.opt_params['camera_3d_size'] * \
            self.opt_params['camera_3d_size'] / (self.opt_params['input_img_shape'][0] * self.opt_params['input_img_shape'][1]))]).cuda().view(-1)
        t_z = k_value * gamma
        cam_trans = torch.cat((t_xy, t_z[...,None]),-1)
        
        return cam_trans

    def get_coord(self, root_pose, hand_pose, shape, cam_trans):
        tensor_shape = shape.shape[:-1]
        batch_size = tensor_shape.numel()
        root_pose = root_pose.flatten(0, -2)
        hand_pose = hand_pose.flatten(0, -2)
        shape = shape.flatten(0, -2)
        cam_trans = cam_trans.flatten(0, -2)

        output = self.mano_layer_right(global_orient=root_pose, hand_pose=hand_pose, betas=shape)
        
        # camera-centered 3D coordinate
        mesh_cam = output.vertices
        joint_cam = torch.bmm(torch.from_numpy(mano.joint_regressor).to(mesh_cam.device)[None,:,:].repeat(batch_size,1,1), mesh_cam)
        
        # project 3D coordinates to 2D space
        x = (joint_cam[:,:,0] + cam_trans[:,None,0]) / (joint_cam[:,:,2] + cam_trans[:,None,2] + 1e-4) * \
            self.opt_params['focal'][0] + self.opt_params['princpt'][0]
        y = (joint_cam[:,:,1] + cam_trans[:,None,1]) / (joint_cam[:,:,2] + cam_trans[:,None,2] + 1e-4) * \
            self.opt_params['focal'][1] + self.opt_params['princpt'][1]
        x = x / self.opt_params['input_img_shape'][1] * self.opt_params['output_hm_shape'][2]
        y = y / self.opt_params['input_img_shape'][0] * self.opt_params['output_hm_shape'][1]
        joint_proj = torch.stack((x,y),2)
        # root-relative 3D coordinates
        root_cam = joint_cam[:,mano.root_joint_idx,None,:]
        joint_cam = joint_cam - root_cam

        # add camera translation for the rendering
        mesh_cam = mesh_cam + cam_trans[:,None,:]

        J, V = 21, 778
        joint_proj = joint_proj.reshape(*tensor_shape, J, 2)
        joint_cam = joint_cam.reshape(*tensor_shape, J, 3)
        mesh_cam = mesh_cam.reshape(*tensor_shape, V, 3)
        return joint_proj, joint_cam, mesh_cam

