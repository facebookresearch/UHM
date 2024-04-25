# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from pytorch3d.transforms import matrix_to_axis_angle
from utils.uhm import uhm
from nets.vgg import Vgg16Features
from config import cfg

# joint angle loss
class PoseLoss(nn.Module):
    def __init__(self):
        super(PoseLoss, self).__init__()

    def forward(self, pose_out, pose_gt):
        batch_size = pose_out.shape[0]
        pose_out = matrix_to_axis_angle(pose_out)
        pose_gt = matrix_to_axis_angle(pose_gt)
        loss = torch.abs(pose_out - pose_gt)
        return loss

# loss for the root-relative 3D vertex coordinates 
class VertexLoss(nn.Module):
    def __init__(self):
        super(VertexLoss, self).__init__()

    def forward(self, mesh_out, mesh_gt, joint_out, joint_gt):
        mesh_out = mesh_out - joint_out[:,uhm.root_joint_idx,None,:]
        mesh_gt = mesh_gt - joint_gt[:,uhm.root_joint_idx,None,:]
        loss = torch.abs(mesh_out - mesh_gt)
        return loss

# loss for the root-relative 3D joint coordinates 
class Joint3DLoss(nn.Module):
    def __init__(self):
        super(Joint3DLoss, self).__init__()

    def forward(self, joint_3d_out, joint_3d_gt, joint_3d_valid):
        joint_3d_out = joint_3d_out - joint_3d_out[:,uhm.root_joint_idx,None,:]
        joint_3d_gt = joint_3d_gt - joint_3d_gt[:,uhm.root_joint_idx,None,:]
        loss = torch.abs(joint_3d_out - joint_3d_gt) * joint_3d_valid
        return loss

# image perceptual loss (VGG)
class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg16Features(layers_weights = [1, 1/16, 1/8, 1/4, 1]).cuda()

    def forward(self, img_out, img_target, mask):
        img_out = img_out * mask
        img_target = img_target * mask

        feat_out = self.vgg(img_out)
        feat_target = self.vgg(img_target)

        loss = torch.abs(feat_out - feat_target)
        return loss

# depthmap loss
class DepthmapLoss(nn.Module):
    def __init__(self):
        super(DepthmapLoss, self).__init__()
    
    def forward(self, depthmap_out, depthmap_gt):
        mask = ((depthmap_out > 0) * (depthmap_gt > 0)).float()
        loss = torch.abs(depthmap_out - depthmap_gt) * mask
        return loss

# forearm rotation
class ForearmLoss(nn.Module):
    def __init__(self):
        super(ForearmLoss, self).__init__()

    def forward(self, joint_out, mesh_out, joint_gt, joint_valid, elbow_gt, elbow_valid):
        is_forearm = torch.FloatTensor(np.argmax(uhm.layer.skinning_weight, 1) == uhm.joints_name.index('Forearm')).cuda() > 0
        wrist_out = joint_out[:,uhm.joints_name.index('Wrist'),:]
        forearm_out = mesh_out[:,is_forearm,:].mean(1)
        vector_out = forearm_out - wrist_out
        vector_out = F.normalize(vector_out)

        wrist_gt = joint_gt[:,uhm.joints_name.index('Wrist'),:]
        vector_gt = elbow_gt - wrist_gt
        vector_gt = F.normalize(vector_gt)
        
        dot_product = (vector_out * vector_gt).sum(1)
        loss = torch.abs(dot_product - 1) * joint_valid[:,uhm.joints_name.index('Wrist'),:].view(-1) * elbow_valid.view(-1)
        return loss

# total variation regularizer
class TVReg(nn.Module):
    def __init__(self):
        super(TVReg, self).__init__()

    def forward(self, out, mask):
        w_variance = (torch.pow(out[:,:,:,:-1] - out[:,:,:,1:], 2) * mask[:,:,:,:-1] * mask[:,:,:,1:]).mean((1,2,3))
        h_variance = (torch.pow(out[:,:,:-1,:] - out[:,:,1:,:], 2) * mask[:,:,:-1,:] * mask[:,:,1:,:]).mean((1,2,3))
        loss = (h_variance + w_variance)
        return loss


