import torch.distributions
from models_cap_new.modules.resnetbackbone import ResNetBackbone
from models_cap_new.modules.unfolder import Unfolder
from models_cap_new.modules.regressor import Regressor
from models_cap_new.modules.transformer import Transformer
from models_cap_new.modules.layer_utils import init_weights
from losses import CoordLoss, ParamLoss, CoordLossOrderInvariant #, DiversityLoss, RMLoss
from utils.MANO import mano

import math
import copy
import torch
from torch import nn
from torch.distributions import Normal

class BlurHandNet(nn.Module):
    def __init__(self, opt, weight_init=True):
        super().__init__()

        # define trainable module
        opt_net = opt['network']
        self.backbone = ResNetBackbone(**opt_net['backbone'])  # backbone
        self.transformer = Transformer(**opt['task_parameters'], **opt_net['transformer'])
        self.unfolder = Unfolder(**opt['task_parameters'], **opt_net['unfolder'])
        self.regressor = Regressor(**opt['task_parameters'], **opt_net['regressor'])  # Regressor
        self.K = opt['task_parameters']['num_k']
        self.k_use = opt['task_parameters']['num_k_select']
        

        # weight initialization
        if weight_init:
            self.backbone.init_weights()
            self.transformer.apply(init_weights), 
            self.unfolder.apply(init_weights), 
            self.regressor.apply(init_weights)

        # for producing 3d hand meshs
        self.mano_layer_right = copy.deepcopy(mano.layer['right']).cuda()
        self.mano_layer_left = copy.deepcopy(mano.layer['left']).cuda()

        # losses
        self.coord_loss = CoordLoss()
        self.coord_loss_order_invariant = CoordLossOrderInvariant()
        self.param_loss = ParamLoss()
        # self.d_loss = DiversityLoss()
        # self.reward_loss = RMLoss()

        
        # parameters
        self.opt_params = opt['task_parameters']
        if opt.get('train', False):
            self.opt_loss = opt['train']['loss']

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
        feat_blur, feat_pyramid = self.backbone(inputs['img'])  # [B, C', H, W]
        B, _, H, W = feat_blur.shape
        T, J = self.n_time, self.n_joints
        K = self.K

        feat_joint, feat_img = self.unfolder(feat_blur, feat_pyramid)   # [B, T, J, E], [B, T, J, 2]

        E = feat_joint.shape[-1]

        # mano params prediction
        mano_shape_org, mano_pose_org, camera_org = self.regressor(feat_blur, feat_joint)
        cam_tran_org = self.get_camera_trans(camera_org)    # [B, K, T, 3]

        # obtain 1) projected 3D coordinates 2) camera-centered 3D joint coordinates 3) camera-centered 3D meshes
        joint_proj_org, joint_cam_org, mesh_cam_org = \
            self.get_coord(mano_pose_org[..., :3], mano_pose_org[..., 3:], mano_shape_org, cam_tran_org)    # [B, K, T, J, 2]

        # # joint prediction (auxiliary task)
        # feat_joint = feat_spat.reshape(B*K*T, C)  # [B*K*T, C]
        # coord_k = self.joint_predictor(feat_joint[..., None, None])    # [B*T, J, 3]

        if mode == 'train':
            loss = {}

            # transpose for compute loss
            joint_proj_all = joint_proj_org.transpose(0, 2) # [T, K, B, J, 2]
            joint_cam_all = joint_cam_org.transpose(0, 2)   # [T, K, B, J, 3]
            mesh_cam_all = mesh_cam_org.transpose(0, 2)     # [T, K, B, V, 3]
            mano_pose_all = mano_pose_org.transpose(0, 2)   # [T, K, B, \theta]
            mano_shape_all = mano_shape_org.transpose(0, 2) # [T, K, B, \beta]

            # choose the 25% best samples
            with torch.no_grad():
                IDX_P, IDX_MD, IDX_F = 0, 1, 2
                loss_joint_cam = self.coord_loss(joint_cam_all[IDX_MD], targets['joint_cam'][None], meta_info['joint_valid'][None]) + \
                                self.coord_loss(joint_cam_all[IDX_P], targets['joint_cam_past'][None], meta_info['joint_valid_past'][None]) + \
                                self.coord_loss(joint_cam_all[IDX_F], targets['joint_cam_future'][None], meta_info['joint_valid_future'][None]) # [K, B, J, 3]
                loss_joint_cam = loss_joint_cam.mean([-1, -2])  # [K, B]
                idx_org = torch.argsort(loss_joint_cam, dim=0)  # [K, B]
                idx = idx_org.unsqueeze(0).repeat(T, 1, 1)  # [T, K, B]
            joint_proj = torch.gather(joint_proj_all, index=idx[...,None,None].repeat(1,1,1, J, 2), dim=1)[:,:self.k_use]
            joint_cam = torch.gather(joint_cam_all, index=idx[...,None,None].repeat(1,1,1, J, 3), dim=1)[:,:self.k_use]
            mesh_cam = torch.gather(mesh_cam_all, index=idx[...,None,None].repeat(1,1,1, 778, 3), dim=1)[:,:self.k_use]
            mano_pose = torch.gather(mano_pose_all, index=idx[...,None].repeat(1,1,1, mano.orig_joint_num*3), dim=1)[:,:self.k_use]
            mano_shape = torch.gather(mano_shape_all, index=idx[...,None].repeat(1,1,1, mano.shape_param_dim), dim=1)[:,:self.k_use]

            loss['joint_proj'] = self.opt_loss['lambda_joint_proj'] * self.coord_loss(joint_proj[IDX_MD],   # [K, B, J, 2]
                                                                                      targets['joint_img'][:,:,:2][None], # [1, B, J, 3]
                                                                                      meta_info['joint_trunc'][None]) # [K, B, J, 2]
            loss['joint_cam'] = self.opt_loss['lambda_joint_cam'] * self.coord_loss(joint_cam[IDX_MD], 
                                                                                    targets['joint_cam'][None], 
                                                                                    meta_info['joint_valid'][None] * meta_info['is_3D'][None,:,None,None]) # [K, B, J, 3]
            loss['mano_joint_cam'] = self.opt_loss['lambda_joint_cam'] * self.coord_loss(joint_cam[IDX_MD], 
                                                                                         targets['mano_joint_cam'][None], 
                                                                                         meta_info['mano_joint_valid'][None]) # [K, B, J, 3]
            loss['mano_pose'] = self.param_loss(mano_pose[IDX_MD], 
                                                targets['mano_pose'][None], 
                                                meta_info['mano_pose_valid'][None])   # [K, B, \theta]

            loss['mano_shape'] = self.param_loss(mano_shape[IDX_MD], 
                                                 targets['mano_shape'][None], 
                                                 meta_info['mano_shape_valid'][None,:,None]) / T # [K, B, \beta]
            # losses on hands in both ends
            # 1) temporal order invariant loss, we just use `pred_order` here
            # loss_joint_aux_pf, pred_order = self.coord_loss_order_invariant(joint_cam[0], joint_cam[2], targets['joint_img_past'][None], targets['joint_img_future'][None],
            #                                                                 meta_info['joint_trunc_past'][None], meta_info['joint_trunc_future'][None], 
            #                                                                 meta_info['is_3D'][None,:,None,None], return_order=True)  # [K, B, ]

            # 2) unfolder driven temporal order loss; use predicted order from Unfolder
            # IDX_P, IDX_F = (2-2*pred_order).to(torch.long), (2*pred_order).to(torch.long)
            loss['joint_proj_past'] = self.opt_loss['lambda_joint_proj'] * self.coord_loss(joint_proj[IDX_P], targets['joint_img_past'][:,:,:2][None], meta_info['joint_trunc_past'][None])
            loss['joint_proj_future'] = self.opt_loss['lambda_joint_proj'] * self.coord_loss(joint_proj[IDX_F], targets['joint_img_future'][:,:,:2][None], meta_info['joint_trunc_future'][None])
            loss['joint_cam_past'] = self.opt_loss['lambda_joint_cam'] * self.coord_loss(joint_cam[IDX_P], targets['joint_cam_past'][None], meta_info['joint_valid_past'][None])
            loss['joint_cam_future'] = self.opt_loss['lambda_joint_cam'] * self.coord_loss(joint_cam[IDX_F], targets['joint_cam_future'][None], meta_info['joint_valid_future'][None])
            loss['mano_joint_cam_past'] = self.opt_loss['lambda_joint_cam'] * self.coord_loss(joint_cam[IDX_P], targets['mano_joint_cam_past'][None], meta_info['mano_joint_valid_past'][None])
            loss['mano_joint_cam_future'] = self.opt_loss['lambda_joint_cam'] * self.coord_loss(joint_cam[IDX_F], targets['mano_joint_cam_future'][None], meta_info['mano_joint_valid_future'][None])
            loss['mano_pose_past'] = self.param_loss(mano_pose[IDX_P], targets['mano_pose_past'][None], meta_info['mano_pose_valid_past'][None])
            loss['mano_pose_future'] = self.param_loss(mano_pose[IDX_F], targets['mano_pose_future'][None], meta_info['mano_pose_valid_future'][None])
            # maybe dont need to do
            loss['mano_shape_past'] = self.param_loss(mano_shape[IDX_P], targets['mano_shape_past'][None], meta_info['mano_shape_valid_past'][None:,None]) / T
            loss['mano_shape_future'] = self.param_loss(mano_shape[IDX_F], targets['mano_shape_future'][None], meta_info['mano_shape_valid_future'][None:,None]) / T

            # 3) diversity promoting loss
            # loss['diversity'] = self.opt_loss['lambda_diversity'] * self.d_loss(feat_mano)
            loss_d, rank = self.info_nce(feat_blur.detach(), mano_pose_all, idx_org)
            loss['diversity'] = self.opt_loss['lambda_diversity'] * loss_d
            loss['rank_first'] = rank
            # choose the best sample according to current MPJPE

            rm_info = {}
            rm_info['img'] = None
            rm_info['feat_blur'] = feat_blur # [B, C, H, W]
            rm_info['mano_pose'] = mano_pose_all.transpose(1, 2)     # [T, B, K, theta]
            # rm_info['feat_mano'] = feat_mano    # [B, K, T, C]
            jpe_md = self.coord_loss(joint_cam_all[IDX_MD], targets['joint_cam'][None], meta_info['joint_valid'][None]).mean([-1, -2])   # [K, B, ]
            jpe_p = self.coord_loss(joint_cam_all[IDX_P], targets['joint_cam_past'][None], meta_info['joint_valid_past'][None]).mean([-1, -2])        # [K, B, ]
            jpe_f = self.coord_loss(joint_cam_all[IDX_F], targets['joint_cam_future'][None], meta_info['joint_valid_future'][None]).mean([-1, -2])    # [K, B, ]
            vpe_md = self.coord_loss(mesh_cam_all[IDX_MD], targets['mesh_cam'][None], meta_info['mesh_cam_valid'][None]).mean([-1, -2])
            vpe_p = self.coord_loss(mesh_cam_all[IDX_P], targets['mesh_cam_past'][None], meta_info['mesh_cam_valid_past'][None]).mean([-1, -2])
            vpe_f = self.coord_loss(mesh_cam_all[IDX_F], targets['mesh_cam_future'][None], meta_info['mesh_cam_valid_future'][None]).mean([-1, -2])
            rm_info['mpjpe_approx'] = torch.stack([jpe_p, jpe_md, jpe_f], dim=-1).permute(1, 2, 0)  # [B, T, K]
            rm_info['mpvpe_approx'] = torch.stack([vpe_p, vpe_md, vpe_f], dim=-1).permute(1, 2, 0)  # [B, T, K]
            return loss, rm_info
        
        elif mode == 'test' or mode == 'test-all':   # test or test-all
            mano_pose_all = mano_pose_org.reshape(B, K, T, -1).transpose(0, 2)                  # [T, K, B, \theta]
            # mano_pose_all = torch.gather(mano_pose_temp, index=idx[...,None].repeat(1,1,1, mano.orig_joint_num*3), dim=1)
            mesh_cam_all = mesh_cam_org.reshape(B, K, T, mano.vertex_num, -1).transpose(0, 1)   # [K, B, T, V, 3]
            mesh_cam = mesh_cam_all.transpose(0, 2) # [T, B, K, V, 3]

            out = {}
            out['img'] = inputs['img']
            
            # our model predictions
            # when evaluating hands in both ends, MPJPE will be calculated in order of minimizing the value
            idx_test = torch.randperm(K)[:self.k_use]  # randomly select K' samples
            out['mano_mesh_cam'] = mesh_cam[1,:,idx_test]  # [B, K, V, 3]
            out['mano_mesh_cam_past'] = mesh_cam[0,:,idx_test]
            out['mano_mesh_cam_future'] = mesh_cam[2,:,idx_test]

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
                
            rm_info = {}
            rm_info['img'] = None
            rm_info['feat_blur'] = feat_blur # [B, C, H, W]
            rm_info['mano_pose'] = mano_pose_all.transpose(1, 2)     # [T, B, K, theta]

            return out, rm_info
        else:
            raise NotImplementedError


    def get_camera_trans(self, cam_param):
        # camera translation
        t_xy = cam_param[...,:2]
        gamma = torch.sigmoid(cam_param[:,2]) # apply sigmoid to make it positive
        k_value = torch.FloatTensor([math.sqrt(self.opt_params['focal'][0] * self.opt_params['focal'][1] * self.opt_params['camera_3d_size'] * \
            self.opt_params['camera_3d_size'] / (self.opt_params['input_img_shape'][0] * self.opt_params['input_img_shape'][1]))]).cuda().view(-1)
        t_z = k_value * gamma
        cam_trans = torch.cat((t_xy, t_z[...,None]), -1)
        
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

        J, V = 21, 778
        joint_proj = joint_proj.reshape(*tensor_shape, J, 2)
        joint_cam = joint_cam.reshape(*tensor_shape, J, 3)
        mesh_cam = mesh_cam.reshape(*tensor_shape, V, 3)
        return joint_proj, joint_cam, mesh_cam

    def uniform_vis(self, inputs, targets, meta_info):
        feat_blur, feat_pyramid = self.backbone(inputs['img'])  # [B, C, H, W]
        B = feat_blur.shape[0]
        T, J = self.n_time, self.n_joints
        feat_base = self.unfolder(feat_blur, feat_pyramid)  # [B, C*T, H, W]
        _, _, H, W = feat_base.shape
        feat_base = feat_base.reshape(B, T, -1, H, W)
        C = feat_base.shape[2]
        K = self.K

        ## reshape for transformer_spatial
        src = (feat_base.reshape(B*T, -1, H, W) # [B*T, C, W, H]
                        .flatten(-2)            # [B*T, C, W*H]
                        .transpose(-1, -2)      # [B*T, H*W, C]
               )
        #* notoken part
        feat_spat = self.transformer_spatial(src)   # [B*T, H*W, C]

        feat_vae_k, mu, logstd = self.vae(feat_spat)    # [K, B*T, C]
        # dist = torch.distributions.Uniform(mu-2*logstd.exp(), mu+2*logstd.exp())    # [B*T, C]
        # feat_vae_k = dist.sample([K-1]) # [K-1, B*T, C]
        # feat_vae_k = feat_vae_k[1:]
        # feat_vae_k = torch.cat([mu[None], feat_vae_k], dim=0) # [K, B*T, C]
        feat_joint = feat_vae_k.reshape(K*B*T, C, 1, 1)
        feat_vae = feat_vae_k.reshape(K, B, T, C).flatten(0, 1)

        coord_k = self.joint_predictor(feat_joint)    # [(K*)B*T, J, 3]
        coord_k = coord_k.reshape(K*B, T, J, 3)          # [(K*)B, T, J, 3]

        coord = coord_k

        # reshape for transformer_temporal
        src = feat_vae  # [B, T, C]
        
        # apply transformer on temporal dim
        feat_mano = self.transformer_temporal(src)  # [(K*)B, T, C]

        # mano params prediction
        mano_shape, mano_pose_bt, camera = self.regressor(feat_mano, coord.detach(), feat_blur)    # [(K*)B, 10], [(K*)B*T, -1], [(K*)B*T, 3]

        cam_tran = self.get_camera_trans(camera)    # [(K*)B*T, 3]

        mano_shape_bt, cam_tran_bt = mano_shape[:, None, :].repeat(1, T, 1).reshape(K*B*T, -1), cam_tran    # [B*T, 10], [B*T, 3]
        # obtain 1) projected 3D coordinates 2) camera-centered 3D joint coordinates 3) camera-centered 3D meshes
        joint_proj_bt, joint_cam_bt, mesh_cam_bt = \
            self.get_coord(mano_pose_bt, mano_shape_bt, cam_tran_bt)    # [B*T, J, 2], [B*T, J, 3], [B*T, V, 3]
        mesh_cam = mesh_cam_bt.reshape(K, B, T, mano.vertex_num, -1).permute(2, 1, 0, 3, 4)    # [T, B, K, V, 3]

        out = {}
        out['img'] = inputs['img']
        
        # our model predictions
        # when evaluating hands in both ends, MPJPE will be calculated in order of minimizing the value
        out['mano_mesh_cam'] = mesh_cam[1]  # [B, K, V, 3]
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
            
        return out, feat_vae_k.reshape(K, B, T, C)