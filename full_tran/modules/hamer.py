import os
import torch
import torch.nn as nn
from typing import Dict, Union

from .backbone import create_backbone
from .mano_head import build_mano_head


class HAMER(nn.Module):

    def __init__(self, cfg: Dict, init_renderer: bool = True):
        """
        Setup HAMER model
        Args:
            cfg: 'network' in cfg file
        """
        super().__init__()

        self.cfg = cfg
        # Create backbone feature extractor
        self.backbone = create_backbone(cfg)
        res = self.load_pretrain()
        assert res
        # Create MANO head
        self.mano_head = build_mano_head(cfg)

    def init_weights(self):
        self.mano_head.init_weights()

    def load_pretrain(self):
        if self.cfg['backbone'].get('pretrained_weights', None):
            # log.info(f'Loading backbone weights from {cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS}')
            ckpt = torch.load(os.path.dirname(__file__) + self.cfg['backbone']['pretrained_weights'], map_location='cpu')['state_dict']
            pop_key_list = []
            for k in ckpt.keys():   # we do not use keypoint layers
                if 'keypoint_head' in k:
                    pop_key_list.append(k)
            for k in pop_key_list:
                ckpt.pop(k, None)
            self.load_state_dict(ckpt)
            return True
        return False

    def forward_step(self, img: torch.Tensor, train=False) -> Union[torch.Tensor]:
        """
        Run a forward step of the network
        img: [B, 3, 256, 256]
        """

        # Use RGB image as input
        x = img
        # batch_size = x.shape[0]

        # Compute conditioning features using the backbone
        conditioning_feats = self.backbone(x)   # [B, L, E]

        pred_hand_pose, pred_betas, pred_cam = self.mano_head(conditioning_feats)  # [B, K, T, -1]

        return pred_hand_pose, pred_betas, pred_cam
        # Store useful regression outputs to the output dict
        output = {}
        output['pred_cam'] = pred_cam
        output['pred_mano_params'] = {k: v.clone() for k,v in pred_mano_params.items()}

        # Compute camera translation
        device = pred_mano_params['hand_pose'].device
        dtype = pred_mano_params['hand_pose'].dtype
        focal_length = self.cfg.EXTRA.FOCAL_LENGTH * torch.ones(batch_size, 2, device=device, dtype=dtype)
        pred_cam_t = torch.stack([pred_cam[:, 1],
                                  pred_cam[:, 2],
                                  2*focal_length[:, 0]/(self.cfg.MODEL.IMAGE_SIZE * pred_cam[:, 0] +1e-9)],dim=-1)
        output['pred_cam_t'] = pred_cam_t
        output['focal_length'] = focal_length

        # Compute model vertices, joints and the projected joints
        pred_mano_params['global_orient'] = pred_mano_params['global_orient'].reshape(batch_size, -1, 3, 3)
        pred_mano_params['hand_pose'] = pred_mano_params['hand_pose'].reshape(batch_size, -1, 3, 3)
        pred_mano_params['betas'] = pred_mano_params['betas'].reshape(batch_size, -1)
        mano_output = self.mano(**{k: v.float() for k,v in pred_mano_params.items()}, pose2rot=False)
        pred_keypoints_3d = mano_output.joints
        pred_vertices = mano_output.vertices
        output['pred_keypoints_3d'] = pred_keypoints_3d.reshape(batch_size, -1, 3)
        output['pred_vertices'] = pred_vertices.reshape(batch_size, -1, 3)
        pred_cam_t = pred_cam_t.reshape(-1, 3)
        focal_length = focal_length.reshape(-1, 2)
        pred_keypoints_2d = perspective_projection(pred_keypoints_3d,
                                                   translation=pred_cam_t,
                                                   focal_length=focal_length / self.cfg.MODEL.IMAGE_SIZE)

        output['pred_keypoints_2d'] = pred_keypoints_2d.reshape(batch_size, -1, 2)
        return output

    def forward(self, img: torch.Tensor) -> Dict:
        """
        Run a forward step of the network in val mode
        Args:
            batch (Dict): Dictionary containing batch data
        Returns:
            Dict: Dictionary containing the regression output
        """
        return self.forward_step(img, train=False)