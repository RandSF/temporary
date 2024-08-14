import torch.distributions
from models_flow.modules.resnetbackbone import ResNetBackbone
from models_flow.modules.unfolder import Unfolder
from models_flow.modules.regressor import Regressor
from models_flow.modules.transformer import Transformer
from models_flow.modules.layer_utils import init_weights
from losses import CoordLoss, ParamLoss, CoordLossOrderInvariant, DiversityLoss
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
        

        # weight initialization
        if weight_init:
            self.backbone.init_weights()
            self.transformer.apply(init_weights)
            self.unfolder.apply(init_weights)
            self.regressor.apply(init_weights)

        # for producing 3d hand meshs
        self.mano_layer_right = copy.deepcopy(mano.layer['right']).cuda()
        self.mano_layer_left = copy.deepcopy(mano.layer['left']).cuda()

        # losses
        self.coord_loss = CoordLoss()
        self.coord_loss_order_invariant = CoordLossOrderInvariant()
        self.param_loss = ParamLoss()
        self.d_loss = DiversityLoss()
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

        feat_joint, joint_img = self.unfolder(feat_blur, feat_pyramid)   # [B, T, J, E], [B, T, J, 2]

        E = feat_joint.shape[-1]

        feat_mano = self.transformer(feat_joint)    # [B, T, J, E]

        # mano params prediction
        predictions = self.regressor(feat_mano)
        shape_mode, pose_mode, camera_mode = predictions['mode']['shape'], predictions['mode']['pose'], predictions['mode']['camera']   # [B, T, 1, ...]
        cam_tran_mode = self.get_camera_trans(camera_mode)
        # obtain 1) projected 3D coordinates 2) camera-centered 3D joint coordinates 3) camera-centered 3D meshes
        joint_proj_mode, joint_cam_mode, mesh_cam_mode = \
            self.get_coord(pose_mode[..., :3], pose_mode[..., 3:], shape_mode, cam_tran_mode)
        
        shape_mode = shape_mode.permute(1, 2, 0, 3)
        pose_mode = pose_mode.permute(1, 2, 0, 3)
        joint_proj_mode = joint_proj_mode.permute(1, 2, 0, 3, 4)
        joint_cam_mode = joint_cam_mode.permute(1, 2, 0, 3, 4)
        mesh_cam_mode = mesh_cam_mode.permute(1, 2, 0, 3, 4)
        
        if mode == 'train':
            loss = {}
            shape_sample, pose_sample, camera_sample= predictions['sample']['shape'], predictions['sample']['pose'], predictions['sample']['camera']   # [B, T, K, ...]
            cam_tran_sample = self.get_camera_trans(camera_sample)    # [B, T, K, 3]
            # obtain 1) projected 3D coordinates 2) camera-centered 3D joint coordinates 3) camera-centered 3D meshes
            joint_proj_sample, joint_cam_sample, mesh_cam_sample = \
                self.get_coord(pose_sample[..., :3], pose_sample[..., 3:], shape_sample, cam_tran_sample)

            # transpose for compute loss
            joint_img = joint_img.transpose(0, 1)   # [T, B, J, 2(3?)]
            shape_sample = shape_sample.permute(1, 2, 0, 3)
            pose_sample = pose_sample.permute(1, 2, 0, 3)
            joint_proj_sample = joint_proj_sample.permute(1, 2, 0, 3, 4)
            joint_cam_sample = joint_cam_sample.permute(1, 2, 0, 3, 4)
            mesh_cam_sample = mesh_cam_sample.permute(1, 2, 0, 3, 4)

            
            # 1) NLL loss
            pose_gt = torch.stack([targets['mano_pose_past'], targets['mano_pose'], targets['mano_pose_future']], dim=1).unsqueeze(2)   # [B, T, 1, \Theta]
            loss['Loss/nll'] = self.opt_loss['lambda_nll'] * -self.regressor.log_prob(feat_mano, pose_gt)[0]    # [B, T, 1]

            # 2) mode loss
            loss.update(self.compute_loss(shape_mode, pose_mode, joint_cam_mode, joint_proj_mode, targets, meta_info, 'mode'))

            # 3) sample loss
            loss.update(self.compute_loss(shape_sample, pose_sample, joint_cam_sample, joint_proj_sample, targets, meta_info, 'sample'))
            
            # 4) auxiliary task
            loss['Loss/joint_aux'] = self.opt_loss['joint_img'] * self.coord_loss(joint_img[1], targets['joint_img'][None], meta_info['joint_trunc'][None])
            loss['Loss/joint_aux_past'] = self.opt_loss['joint_img'] * self.coord_loss(joint_img[0], targets['joint_img_past'][None], meta_info['joint_trunc_past'][None])
            loss['Loss/joint_aux_future'] = self.opt_loss['joint_img'] * self.coord_loss(joint_img[2], targets['joint_img_future'][None], meta_info['joint_trunc_future'][None])

            # # 5) diversity promoting loss
            # loss['diversity'] = self.opt_loss['diversity'] * self.d_loss(pose_sample.transpose(0, 2))   # [T, K, B]

            rm_info = {}
            
            return loss, rm_info
        
        elif mode == 'train-rm':
            raise not ImportError
            rm_info = {}
            rm_info['pose'] = mano_pose_org # [B, K, T, \theta]
            rm_info['joint_cam'] = joint_cam_org    # [B, K, T, J, 3]
            rm_info['ctx'] = feat_blur  # [B, C, H, W]
            rm_info['feat_mano'] = feat_mano    # [B, K, T, J, E]

            jpe_md = self.coord_loss(joint_cam_all[IDX_MD], targets['joint_cam'][None], meta_info['joint_valid'][None]).mean([-1, -2])   # [K, B, ]
            jpe_p = self.coord_loss(joint_cam_all[IDX_P], targets['joint_cam_past'][None], meta_info['joint_valid_past'][None]).mean([-1, -2])        # [K, B, ]
            jpe_f = self.coord_loss(joint_cam_all[IDX_F], targets['joint_cam_future'][None], meta_info['joint_valid_future'][None]).mean([-1, -2])    # [K, B, ]
            # vpe_md = self.coord_loss(mesh_cam_all[IDX_MD], targets['mesh_cam'][None], meta_info['mesh_cam_valid'][None]).mean([-1, -2])
            # vpe_p = self.coord_loss(mesh_cam_all[IDX_P], targets['mesh_cam_past'][None], meta_info['mesh_cam_valid_past'][None]).mean([-1, -2])
            # vpe_f = self.coord_loss(mesh_cam_all[IDX_F], targets['mesh_cam_future'][None], meta_info['mesh_cam_valid_future'][None]).mean([-1, -2])
            mpjpe_approx = torch.stack([jpe_p, jpe_md, jpe_f], dim=-1).permute(1, 2, 0)  # [B, T, K]
            rm_info['error'] = mpjpe_approx.sum(dim=1)   # [B, K]
            # rm_info['mpvpe_approx'] = torch.stack([vpe_p, vpe_md, vpe_f], dim=-1).permute(1, 2, 0)  # [B, T, K]

            return rm_info

        elif mode == 'test' or mode == 'test-all':   # test or test-all
            raise NotImplementedError
            mano_pose_all = mano_pose_org.reshape(B, K, T, -1).transpose(0, 2)                  # [T, K, B, \theta]
            # mano_pose_all = torch.gather(mano_pose_temp, index=idx[...,None].repeat(1,1,1, mano.orig_joint_num*3), dim=1)
            mesh_cam_all = mesh_cam_org.reshape(B, K, T, mano.vertex_num, -1).transpose(0, 1)   # [K, B, T, V, 3]
            mesh_cam = mesh_cam_all.transpose(0, 2) # [T, B, K, V, 3]

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
                
            rm_info = {}
            rm_info['pose'] = mano_pose_org # [B, K, T, \theta]
            rm_info['joint_cam'] = joint_cam_org    # [B, K, T, J, 3]
            rm_info['ctx'] = feat_blur  # [B, C, H, W]
            rm_info['feat_mano'] = feat_mano    # [B, K, T, J, E]

            jpe_md = self.coord_loss(joint_cam_all[IDX_MD], targets['joint_cam'][None], meta_info['joint_valid'][None]).mean([-1, -2])   # [K, B, ]
            jpe_p = self.coord_loss(joint_cam_all[IDX_P], targets['joint_cam_past'][None], meta_info['joint_valid_past'][None]).mean([-1, -2])        # [K, B, ]
            jpe_f = self.coord_loss(joint_cam_all[IDX_F], targets['joint_cam_future'][None], meta_info['joint_valid_future'][None]).mean([-1, -2])    # [K, B, ]
            # vpe_md = self.coord_loss(mesh_cam_all[IDX_MD], targets['mesh_cam'][None], meta_info['mesh_cam_valid'][None]).mean([-1, -2])
            # vpe_p = self.coord_loss(mesh_cam_all[IDX_P], targets['mesh_cam_past'][None], meta_info['mesh_cam_valid_past'][None]).mean([-1, -2])
            # vpe_f = self.coord_loss(mesh_cam_all[IDX_F], targets['mesh_cam_future'][None], meta_info['mesh_cam_valid_future'][None]).mean([-1, -2])
            mpjpe_approx = torch.stack([jpe_p, jpe_md, jpe_f], dim=-1).permute(1, 2, 0)  # [B, T, K]
            rm_info['error'] = mpjpe_approx.sum(dim=1)   # [B, K]

            return out, rm_info
        else:
            raise NotImplementedError


    def get_camera_trans(self, cam_param):
        # camera translation
        t_xy = cam_param[...,:2]
        gamma = torch.sigmoid(cam_param[...,2]) # apply sigmoid to make it positive
        k_value = torch.FloatTensor([math.sqrt(self.opt_params['focal'][0] * self.opt_params['focal'][1] * self.opt_params['camera_3d_size'] * \
            self.opt_params['camera_3d_size'] / (self.opt_params['input_img_shape'][0] * self.opt_params['input_img_shape'][1]))]).cuda().view(-1)
        t_z = k_value * gamma
        cam_trans = torch.cat((t_xy, t_z[...,None]), -1)
        
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

    def compute_loss(self, shape, pose, joint_cam, joint_proj, targets, meta_info, kind):
        if kind == 'mode':
            weights = self.opt_loss['mode']
        elif kind == 'sample':
            weights = self.opt_loss['sample']
        else:
            raise NotImplementedError
        
        IDX_P, IDX_MD, IDX_F = 0, 1, 2
        loss = {}
        # 1) loss in middle frame
        loss['Loss/joint_proj_'+kind] = weights['joint_proj'] * \
            self.coord_loss(joint_proj[IDX_MD],   # [1, B, J, 2]
                            targets['joint_img'][:,:,:2][None], # [1, B, J, 3]
                            meta_info['joint_trunc'][None]) # [1, B, J, 2]
        
        loss['Loss/joint_cam_'+kind] = weights['joint_cam'] * \
            self.coord_loss(joint_cam[IDX_MD], targets['joint_cam'][None], meta_info['joint_valid'][None]) # [1, B, J, 3]
        
        loss['Loss/mano_joint_cam_'+kind] = weights['joint_cam'] * \
            self.coord_loss(joint_cam[IDX_MD], targets['mano_joint_cam'][None], meta_info['mano_joint_valid'][None]) # [1, B, J, 3]
        
        loss['Loss/mano_pose_'+kind] = weights['mano_pose'] * \
            self.param_loss(pose[IDX_MD], targets['mano_pose'][None], meta_info['mano_pose_valid'][None])   # [K, B, \theta]

        loss['Loss/mano_shape_'+kind] = weights['mano_shape'] * \
            self.param_loss(shape[IDX_MD], targets['mano_shape'][None], meta_info['mano_shape_valid'][None,:,None])    # [K, B, \beta]
        
        #OTTO unfolder driven temporal order loss
        # 2) loss in past
        loss['Loss/joint_proj_past_'+kind] = weights['joint_proj'] * \
            self.coord_loss(joint_proj[IDX_P],   # [1, B, J, 2]
                            targets['joint_img_past'][:,:,:2][None], # [1, B, J, 3]
                            meta_info['joint_trunc_past'][None]) # [1, B, J, 2]
        
        loss['Loss/joint_cam_past_'+kind] = weights['joint_cam'] * \
            self.coord_loss(joint_cam[IDX_P], targets['joint_cam_past'][None], meta_info['joint_valid_past'][None]) # [1, B, J, 3]
        
        loss['Loss/mano_joint_cam_past_'+kind] = weights['joint_cam'] * \
            self.coord_loss(joint_cam[IDX_P], targets['mano_joint_cam_past'][None], meta_info['mano_joint_valid_past'][None]) # [1, B, J, 3]
        
        loss['Loss/mano_pose_past_'+kind] = weights['mano_pose'] * \
            self.param_loss(pose[IDX_P], targets['mano_pose_past'][None], meta_info['mano_pose_valid_past'][None])   # [K, B, \theta]

        loss['Loss/mano_shape_past_'+kind] = weights['mano_shape'] * \
            self.param_loss(shape[IDX_P], targets['mano_shape_past'][None], meta_info['mano_shape_valid_past'][None,:,None])    # [K, B, \beta]
        
        # 3) loss in future 
        loss['Loss/joint_proj_future_'+kind] = weights['joint_proj'] * \
            self.coord_loss(joint_proj[IDX_F],   # [1, B, J, 2]
                            targets['joint_img_future'][:,:,:2][None], # [1, B, J, 3]
                            meta_info['joint_trunc_future'][None]) # [1, B, J, 2]
        
        loss['Loss/joint_cam_future_'+kind] = weights['joint_cam'] * \
            self.coord_loss(joint_cam[IDX_F], targets['joint_cam_future'][None], meta_info['joint_valid_future'][None]) # [1, B, J, 3]
        
        loss['Loss/mano_joint_cam_future_'+kind] = weights['joint_cam'] * \
            self.coord_loss(joint_cam[IDX_F], targets['mano_joint_cam_future'][None], meta_info['mano_joint_valid_future'][None]) # [1, B, J, 3]
        
        loss['Loss/mano_pose_future_'+kind] = weights['mano_pose'] * \
            self.param_loss(pose[IDX_F], targets['mano_pose_future'][None], meta_info['mano_pose_valid_future'][None])   # [K, B, \theta]

        loss['Loss/mano_shape_future_'+kind] = weights['mano_shape'] * \
            self.param_loss(shape[IDX_F], targets['mano_shape_future'][None], meta_info['mano_shape_valid_future'][None,:,None])    # [K, B, \beta]
        
        return loss

        

