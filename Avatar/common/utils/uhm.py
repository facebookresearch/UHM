# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import torch
import torch.nn as nn
import os.path as osp
from utils.UHM.uhm import UHM
from utils.transforms import rasterize_to_uv
import numpy as np
import cv2
from config import cfg

class UHM_Wrapper(object):
    def __init__(self):
        self.layer = UHM(cfg.uhm_root_path)
        self.vertex = self.layer.vertex
        self.face = self.layer.face
        self.vertex_num = self.layer.vertex_num
        self.vertex_uv = self.layer.vertex_uv
        self.face_uv = self.layer.face_uv
        self.joints_name = self.layer.joints_name
        self.joint_num = self.layer.joint_num
        self.root_joint_idx = self.layer.root_joint_idx
        self.joint = self.layer.joint
        self.pose_dof_mask = self.layer.pose_dof_mask
        self.id_code_dim = self.layer.id_code_dim
        self.pose_code_dim = self.layer.pose_code_dim
        self.uv_mask = self.get_uv_mask()

    def get_uv_mask(self):
        vertex_uv = torch.from_numpy(self.layer.vertex_uv).float().cuda()[None,:,:]
        face_uv = torch.from_numpy(self.layer.face_uv).long().cuda()[None,:,:]
        fragments = rasterize_to_uv(vertex_uv, face_uv, cfg.tex_shape)
        pix_to_face = fragments.pix_to_face # batch_size, cfg.tex_shape[0], cfg.tex_shape[1], faces_per_pixel. invalid: -1
        pix_to_face = pix_to_face[0,:,:,0]
        mask = (pix_to_face != -1).cpu().numpy().astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1) # dilate to prevent black line through the seam (due to the interpolation during the rendering)
        mask = torch.from_numpy(mask) > 0
        return mask

    def load_unwrapped_texture(self, root_path):
        unwrapped_tex = cv2.imread(osp.join(root_path, 'unwrapped_texture_wo_shadow_inpaint.png'))[:,:,::-1].copy()
        unwrapped_tex = torch.from_numpy(unwrapped_tex).float().permute(2,0,1)/255
        self.unwrapped_tex = unwrapped_tex

        unwrapped_tex_mask = cv2.imread(osp.join(root_path, 'unwrapped_texture_grad_mask.png'))[:,:,0].copy()
        unwrapped_tex_mask = torch.from_numpy(unwrapped_tex_mask).float()/255
        self.unwrapped_tex_mask = unwrapped_tex_mask

uhm = UHM_Wrapper()
