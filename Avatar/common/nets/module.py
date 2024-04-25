# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.uhm import uhm
from nets.layer import make_conv_layers, make_linear_layers, Vert2UV
from utils.transforms import sample_joint_features
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_euler_angles, euler_angles_to_matrix, matrix_to_rotation_6d
from pytorch3d.structures import Meshes
from config import cfg

# id_code
class IDNet(nn.Module):
    def __init__(self):
        super(IDNet, self).__init__()
        self.seed = nn.Parameter(torch.randn(32).float())
        self.id_out = make_linear_layers([32, 32, uhm.id_code_dim], relu_final=False)

    def forward(self, batch_size):
        id_code = self.id_out(self.seed)[None,:].repeat(batch_size,1)
        return id_code

# color calibration
class ColorCalibNet(nn.Module):
    def __init__(self):
        super(ColorCalibNet, self).__init__()
        self.seed = nn.Parameter(torch.randn(32).float())
        self.rgb_out = make_linear_layers([32, 32, 3], relu_final=False)

    def forward(self, batch_size):
        rgb = F.softplus(self.rgb_out(self.seed)[None,:].repeat(batch_size,1))
        return rgb

# regress pose (root pose, hand pose, and root translation)
class PoseNet(nn.Module):
    def __init__(self, backbone):
        super(PoseNet, self).__init__()
        self.backbone = backbone
        self.conv = make_conv_layers([2048,512], kernel=1, stride=1, padding=0)

        self.root_pose_out = make_linear_layers([uhm.joint_num*(512+2)+uhm.id_code_dim+6, 6], relu_final=False)
        self.hand_pose_out = make_linear_layers([uhm.joint_num*(512+2)+uhm.id_code_dim+6, (uhm.joint_num-1)*6], relu_final=False)
        self.root_trans_out = make_linear_layers([512+2+uhm.id_code_dim+6, 128, 3], relu_final=False)

        self.root_pose_code_out = make_linear_layers([uhm.joint_num*(512+2)+uhm.id_code_dim+6, 6], relu_final=False)
        self.pose_code_out = make_linear_layers([uhm.joint_num*(512+2)+uhm.id_code_dim+6, uhm.pose_code_dim], relu_final=False)
        self.root_trans_code_out = make_linear_layers([512+2+uhm.id_code_dim+6, 128, 3], relu_final=False)

    def forward(self, img, depthmap, mask, joint_img, img2bb_trans, id_code):
        batch_size = img.shape[0]

        # extract joint-level features
        feat = torch.cat((img, depthmap, mask),1)
        feat = self.backbone(feat)
        feat = self.conv(feat)
        joint_feat = sample_joint_features(feat, joint_img) # batch_size, joint_num, feat_dim
        joint_feat = torch.cat((joint_feat, joint_img),2)
        root_feat = torch.cat((joint_feat[:,uhm.root_joint_idx,:], id_code, img2bb_trans.view(batch_size,6)),1)
        joint_feat = torch.cat((joint_feat.view(batch_size,-1), id_code, img2bb_trans.view(batch_size,6)),1)

        # pose and translation
        root_pose = self.root_pose_out(joint_feat)
        root_pose = rotation_6d_to_matrix(root_pose)
        hand_pose = self.hand_pose_out(joint_feat).view(batch_size,uhm.joint_num-1,6)
        hand_pose = rotation_6d_to_matrix(hand_pose)
        root_trans = self.root_trans_out(root_feat)
        root_trans = torch.cat((root_trans[:,:2], F.softplus(root_trans[:,2:])),1) # make depth positive

        # pose and translation (for the pose code)
        root_pose_for_code = self.root_pose_code_out(joint_feat)
        root_pose_for_code = rotation_6d_to_matrix(root_pose_for_code)
        pose_code = self.pose_code_out(joint_feat)
        root_trans_for_code = self.root_trans_code_out(root_feat)
        root_trans_for_code = torch.cat((root_trans_for_code[:,:2], F.softplus(root_trans_for_code[:,2:])),1) # make depth positive

        return root_pose, hand_pose, root_trans, root_pose_for_code, pose_code, root_trans_for_code

# estimate shadow
class ShadowNet(nn.Module):
    def __init__(self):
        super(ShadowNet, self).__init__()
        self.vert2uv = Vert2UV((cfg.tex_shape[0]//32, cfg.tex_shape[1]//32))
        self.pos_enc = nn.Parameter(torch.zeros((6+(uhm.joint_num-1)*6+uhm.id_code_dim+1,cfg.tex_shape[0]//32, cfg.tex_shape[1]//32)))
        self.conv1 = make_conv_layers([6+(uhm.joint_num-1)*6+uhm.id_code_dim+1, 256], norm='gn', act='silu')
        self.conv2 = nn.Sequential(make_conv_layers([256, 256], norm='gn', act='silu'))
        self.conv3 = nn.Sequential(make_conv_layers([256, 128], norm='gn', act='silu'))
        self.conv4 = nn.Sequential(make_conv_layers([128, 64], norm='gn', act='silu'))
        self.shadow_out = make_conv_layers([64, 1], norm='gn', act='silu', bnrelu_final=False)

    def compute_view_dir(self, mesh):
        batch_size = mesh.shape[0]

        # per-vertex normal
        normal = Meshes(verts=mesh, faces=torch.LongTensor(uhm.face).cuda()[None,:,:].repeat(batch_size,1,1)).verts_normals_packed().reshape(batch_size,uhm.vertex_num,3)
        normal = F.normalize(normal, dim=2)

        # camera -> vertex vector
        R = torch.eye(3)[None,:,:].repeat(batch_size,1,1).float().cuda()
        t = torch.zeros((batch_size,3)).float().cuda()
        cam_pos = -torch.bmm(torch.inverse(R), t.view(-1,3,1)).view(-1,3) # camera position
        cam_vec = F.normalize(mesh - cam_pos[:,None,:], dim=2)

        # view direction
        view_dir = torch.sum(normal * cam_vec, dim=2)
        return view_dir

    def kill_pose_dof(self, hand_pose):
        batch_size = hand_pose.shape[0]
        pose_dof_mask = torch.FloatTensor(uhm.pose_dof_mask).cuda()[None,:,:].repeat(batch_size,1,1)
        pose_dof_mask = torch.cat((pose_dof_mask[:,:uhm.root_joint_idx,:], pose_dof_mask[:,uhm.root_joint_idx+1:,:]),1) # remove root joint
        hand_pose = torch.flip(matrix_to_euler_angles(hand_pose, 'ZYX'), [2])
        hand_pose = hand_pose * pose_dof_mask
        hand_pose = euler_angles_to_matrix(torch.flip(hand_pose, [2]), 'ZYX')
        hand_pose = matrix_to_rotation_6d(hand_pose)
        return hand_pose

    def forward(self, root_pose, hand_pose, id_code, mesh):
        batch_size = root_pose.shape[0]

        root_pose = matrix_to_rotation_6d(root_pose)
        root_pose_uv = root_pose[:,:,None,None].repeat(1,1,cfg.tex_shape[0]//32,cfg.tex_shape[1]//32)
        hand_pose = self.kill_pose_dof(hand_pose)
        hand_pose_uv = hand_pose.view(batch_size,(uhm.joint_num-1)*6,1,1).repeat(1,1,cfg.tex_shape[0]//32,cfg.tex_shape[1]//32)
        id_code_uv = id_code.view(batch_size,uhm.id_code_dim,1,1).repeat(1,1,cfg.tex_shape[0]//32,cfg.tex_shape[1]//32)
        view_dir = self.compute_view_dir(mesh)
        view_dir_uv = self.vert2uv(view_dir[:,:,None]) # unwrap to UV space. (batch_size, 1, cfg.tex_shape[0]//32, cfg.tex_shape[1]//32)
        feat = torch.cat((root_pose_uv, hand_pose_uv, id_code_uv, view_dir_uv), 1) + self.pos_enc[None,:,:,:]

        feat = self.conv1(feat) # batch_size, 256, cfg.tex_shape[0]//32, cfg.tex_shape[1]//32
        feat = F.interpolate(self.conv2(feat), scale_factor=2, mode='nearest') # batch_size, 256, cfg.tex_shape[0]//16, cfg.tex_shape[1]//16
        feat = F.interpolate(self.conv3(feat), scale_factor=2, mode='nearest') # batch_size, 128, cfg.tex_shape[0]//8, cfg.tex_shape[1]//8
        feat = F.interpolate(self.conv4(feat), scale_factor=2, mode='nearest') # batch_size, 64, cfg.tex_shape[0]//4, cfg.tex_shape[1]//4
        shadow = torch.sigmoid(F.interpolate(self.shadow_out(feat), cfg.tex_shape, mode='bilinear')) # batch_size, 1, cfg.tex_shape[0], cfg.tex_shape[1]
        shadow = shadow * uhm.uv_mask.float().cuda()[None,None,:,:]
        return shadow


