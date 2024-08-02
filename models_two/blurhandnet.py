import copy
import math
import torch
import torch.nn as nn

from losses import CoordLoss, ParamLoss, CoordLossOrderInvariant, DiversityPromotingLoss
from models_two.modules.transformer import Transformer_temporal, Transformer_candidate
from models_two.modules.regressor import Regressor
from models_two.modules.resnetbackbone import ResNetBackbone
from models_two.modules.layer_utils import init_weights
from utils.MANO import mano


class BlurHandNet(nn.Module):
    def __init__(self, opt, weight_init=True):
        super().__init__()
        # define trainable module
        opt_net = opt['network']
        self.backbone = ResNetBackbone(**opt_net['backbone'])  # backbone
        # self.ktformer = KTFormer(opt['task_parameters'], **opt_net['ktformer'])  # KTFormer
        self.transformer_t = Transformer_temporal(**opt['task_parameters'], **opt_net['transformer'])
        self.transformer_k = Transformer_candidate(**opt['task_parameters'], **opt_net['transformer'])
        self.regressor = Regressor(opt['task_parameters'], **opt_net['regressor'])  # Regressor
        
        # weight initialization
        if weight_init:
            self.backbone.init_weights()
            self.transformer_t.apply(init_weights)
            self.transformer_k.apply(init_weights)
            self.regressor.apply(init_weights)
        
        # for producing 3d hand meshs
        self.mano_layer_right = copy.deepcopy(mano.layer['right'])#.cuda()
        self.mano_layer_left = copy.deepcopy(mano.layer['left'])#.cuda()
        
        # losses
        self.coord_loss = CoordLoss()
        self.coord_loss_order_invariant = CoordLossOrderInvariant()
        self.param_loss = ParamLoss()
        self.diversity_loss = DiversityPromotingLoss()
        
        # parameters
        self.opt_params = opt['task_parameters']
        if opt.get('train', False):
            self.opt_loss = opt['train']['loss']
        
    def forward(self, inputs, targets, meta_info, mode):
        # extract feature from backbone
        feat_blur, feat_pyramid = self.backbone(inputs['img'])  # [B, C, H, W]

        B = feat_blur.shape[0]
        K, KS = self.opt_params['num_k'], self.opt_params['num_k_select']
        T, J = self.opt_params['num_timestep'], self.opt_params['num_joints']
        
        feat_temp = self.transformer_t(feat_blur) # [B, T, E]

        feat_joint = self.transformer_k(feat_temp)  # [B, K, T, E]
        
        # regress mano shape, pose and camera parameter
        mano_shape, mano_pose, cam_param = self.regressor(feat_blur, feat_joint)    # [B, K, T, ...]
        
        # obtain camera translation to project 3D coordinates into 2D space
        cam_trans = self.get_camera_trans(cam_param)    # [B, K, T, 3]
        
        # obtain 1) projected 3D coordinates 2) camera-centered 3D joint coordinates 3) camera-centered 3D meshes
        joint_proj_flatten, joint_cam_flatten, mesh_cam_flatten = \
            self.get_coord(mano_pose[..., :3], mano_pose[...,3:], mano_shape, cam_trans)    # [B, K, T, ...]
    
        if mode == 'train':
            loss = {}

            joint_proj_all = joint_proj_flatten.transpose(0, 2) # [T, K, B]
            joint_cam_all = joint_cam_flatten.transpose(0, 2)
            mano_pose_all = mano_pose.transpose(0, 2)
            mano_shape_all = mano_shape.transpose(0, 2)

            IDX_P, IDX_M, IDX_F = 0, 1, 2
            ## select the better ones
            with torch.no_grad():
                loss_cam = self.coord_loss(joint_cam_all[IDX_P], targets['joint_cam'][None], meta_info['joint_valid'][None]) + \
                           self.coord_loss(joint_cam_all[IDX_M], targets['joint_cam_past'][None], meta_info['joint_valid_past'][None]) + \
                           self.coord_loss(joint_cam_all[IDX_F], targets['joint_cam_future'][None], meta_info['joint_valid_future'][None])  # [K, B, J, 3]
                idx = torch.argsort(loss_cam.mean([-1, -2]), dim=0, descending=False)   # [K, B]

            # joint_img = torch.gather(joint_img_all, dim=1, index=idx[None, ..., J, 3].expand_as(joint_img_all))[:, :KS]
            joint_proj_sort = torch.gather(joint_proj_all, dim=1, index=idx[None, ..., None, None].expand_as(joint_proj_all))
            joint_cam_sort = torch.gather(joint_cam_all, dim=1, index=idx[None, ..., None, None].expand_as(joint_cam_all))
            mano_pose_sort = torch.gather(mano_pose_all, dim=1, index=idx[None, ..., None].expand_as(mano_pose_all))
            mano_shape_sort = torch.gather(mano_shape_all, dim=1, index=idx[None, ..., None].expand_as(mano_shape_all))[:, :KS]

            joint_proj = joint_proj_sort[:, :KS]
            joint_cam = joint_cam_sort[:, :KS]
            mano_pose = mano_pose_sort[:, :KS]
            mano_shape = mano_shape_sort[:, :KS]
            
            # losses on middle hand; 
            # loss['joint_img'] = self.opt_loss['lambda_joint_img'] * self.coord_loss(joint_img[IDX_M], targets['joint_img'], meta_info['joint_trunc'], meta_info['is_3D'])
            loss['Loss/joint_proj'] = self.opt_loss['lambda_joint_proj'] * self.coord_loss(joint_proj[IDX_M], targets['joint_img'][None,...,:2], meta_info['joint_trunc'][None])
            loss['Loss/joint_cam'] = self.opt_loss['lambda_joint_cam'] * self.coord_loss(joint_cam[IDX_M], targets['joint_cam'][None], meta_info['joint_valid'][None])
            loss['Loss/mano_joint_cam'] = self.opt_loss['lambda_joint_cam'] * self.coord_loss(joint_cam[IDX_M], targets['mano_joint_cam'][None], meta_info['mano_joint_valid'][None])
            loss['Loss/mano_pose'] = self.param_loss(mano_pose[IDX_M], targets['mano_pose'][None], meta_info['mano_pose_valid'][None])
            loss['Loss/mano_shape'] = self.param_loss(mano_shape[IDX_M], targets['mano_shape'][None], meta_info['mano_shape_valid'][None,...,None])
            
            # losses on hands in both ends
            # loss['joint_img_past'] = self.opt_loss['lambda_joint_img'] * self.coord_loss(joint_img[IDX_P], targets['joint_img_past'], meta_info['joint_trunc_past'], meta_info['is_3D'])
            # loss['joint_img_future'] = self.opt_loss['lambda_joint_img'] * self.coord_loss(joint_img[IDX_F], targets['joint_img_future'], meta_info['joint_trunc_future'], meta_info['is_3D'])

            loss['Loss/joint_proj_past'] = self.opt_loss['lambda_joint_proj'] * self.coord_loss(joint_proj[IDX_P], targets['joint_img_past'][None,...,:2], meta_info['joint_trunc_past'][None])
            loss['Loss/joint_proj_future'] = self.opt_loss['lambda_joint_proj'] * self.coord_loss(joint_proj[IDX_F], targets['joint_img_future'][None,...,:2], meta_info['joint_trunc_future'][None])
            
            loss['Loss/joint_cam_past'] = self.opt_loss['lambda_joint_cam'] * self.coord_loss(joint_cam[IDX_P], targets['joint_cam_past'][None], meta_info['joint_valid_past'][None])
            loss['Loss/joint_cam_future'] = self.opt_loss['lambda_joint_cam'] * self.coord_loss(joint_cam[IDX_F], targets['joint_cam_future'][None], meta_info['joint_valid_future'][None])
            loss['Loss/mano_joint_cam_past'] = self.opt_loss['lambda_joint_cam'] * self.coord_loss(joint_cam[IDX_P], targets['mano_joint_cam_past'][None], meta_info['mano_joint_valid_past'][None])
            loss['Loss/mano_joint_cam_future'] = self.opt_loss['lambda_joint_cam'] * self.coord_loss(joint_cam[IDX_F], targets['mano_joint_cam_future'][None], meta_info['mano_joint_valid_future'][None])
            
            loss['Loss/mano_pose_past'] = self.param_loss(mano_pose[IDX_P], targets['mano_pose_past'][None], meta_info['mano_pose_valid_past'][None])
            loss['Loss/mano_pose_future'] = self.param_loss(mano_pose[IDX_F], targets['mano_pose_future'][None], meta_info['mano_pose_valid_future'][None])
            
            loss['Loss/mano_shape_past'] = self.param_loss(mano_shape[IDX_P], targets['mano_shape_past'][None], meta_info['mano_shape_valid_past'][None,...,None])
            loss['Loss/mano_shape_future'] = self.param_loss(mano_shape[IDX_F], targets['mano_shape_future'][None], meta_info['mano_shape_valid_future'][None,...,None])


            # DPL
            loss['Loss/diversity'] = self.opt_loss['lambda_diverisity'] * self.diversity_loss(mano_pose)
        
            return loss
            
        else:
            out = {}
            out['img'] = inputs['img']
            
            # our model predictions
            # when evaluating hands in both ends, MPJPE will be calculated in order of minimizing the value
            mesh_cam_all = mesh_cam_flatten.reshape(B, K, T, -1, 3).transpose(0, 2)
            assert K == 1
            out['mano_mesh_cam'] = mesh_cam_all[1, 0]
            out['mano_mesh_cam_past'] = mesh_cam_all[0, 0]
            out['mano_mesh_cam_future'] = mesh_cam_all[2, 0]

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
            self.opt_params['camera_3d_size'] / (self.opt_params['input_img_shape'][0] * self.opt_params['input_img_shape'][1]))]).to(gamma.device).view(-1)
        t_z = k_value * gamma
        cam_trans = torch.cat((t_xy, t_z[...,None]),-1)
        
        return cam_trans

    def get_coord(self, root_pose, hand_pose, shape, cam_trans):
        tensor_shape = shape.shape[:-1]
        batch_size = tensor_shape.numel()
        root_pose = root_pose.flatten(0, -1)
        hand_pose = hand_pose.flatten(0, -1)
        shape = shape.flatten(0, -1)
        cam_trans = cam_trans.flatten(0, -1)

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
        
        joint_proj = joint_proj.reshape(*tensor_shape, -1)
        joint_cam = joint_cam.reshape(*tensor_shape, -1)
        mesh_cam = mesh_cam.reshape(*tensor_shape, -1)
        return joint_proj, joint_cam, mesh_cam
   