# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import numpy as np
import os.path as osp
import torch
import torch.nn as nn
from torch.nn import functional as F
from pytorch3d.io import load_obj
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d, matrix_to_euler_angles, euler_angles_to_matrix
import pickle

class ID_Decoder(nn.Module):
    def __init__(self, id_code_dim, vertex_num, joint_num, joints_name, joints_name_id_decoder):
        super().__init__()
        self.id_code_dim = id_code_dim
        self.vertex_num = vertex_num
        self.joint_num = joint_num
        self.joints_name = joints_name
        self.joints_name_id_decoder = joints_name_id_decoder
        self.scale = nn.Sequential(nn.Linear(self.id_code_dim, 256), nn.ReLU(inplace=True), nn.Linear(256, 1+len(joints_name_id_decoder)))
        self.vertex = nn.Sequential(nn.Linear(self.id_code_dim, 256), nn.ReLU(inplace=True), nn.Linear(256, self.vertex_num*3))

    def forward(self, id_code):
        batch_size = id_code.shape[0]
        scale = self.scale(id_code)

        # make the global scale positive
        global_scale = 2 ** scale[:,:1]

        # change order of the skeleton corrective to that of self.joints_name
        scale = scale[:,1:] # exclude the global scale
        tgt_joints_name = sum([[name + '_X', name + '_Y', name + '_Z'] for name in self.joints_name], [])
        skeleton_corrective = torch.zeros((batch_size,self.joint_num*3)).float().cuda()
        for j in range(self.joint_num*3):
            if tgt_joints_name[j] in self.joints_name_id_decoder:
                idx = self.joints_name_id_decoder.index(tgt_joints_name[j])
                skeleton_corrective[:,j] = scale[:,idx] / 100 # centimeter to meter
        skeleton_corrective = skeleton_corrective.view(batch_size,self.joint_num,3)
        
        vertex_corrective = self.vertex(id_code).view(batch_size,self.vertex_num,3) / 1000
        return global_scale, skeleton_corrective, vertex_corrective

class Pose_Decoder(nn.Module):
    def __init__(self, joint_num, id_code_dim, vertex_num, pose_corrective_mask, pose_dof_mask, neighbor_joint_idxs, neighbor_joint_mask, root_joint_idx):
        super().__init__()
        self.joint_num = joint_num
        self.id_code_dim = id_code_dim
        self.vertex_num = vertex_num
        self.pose_corrective_mask = pose_corrective_mask
        self.pose_dof_mask = pose_dof_mask
        self.neighbor_joint_idxs = neighbor_joint_idxs
        self.neighbor_joint_mask = neighbor_joint_mask
        self.root_joint_idx = root_joint_idx
        self.vertex = nn.Sequential(nn.Conv1d((self.joint_num-1)*(18+self.id_code_dim), (self.joint_num-1)*256, kernel_size=1, groups=self.joint_num-1),\
                                    nn.ReLU(inplace=True), \
                                    nn.Conv1d((self.joint_num-1)*256, (self.joint_num-1)*self.vertex_num*3, kernel_size=1, groups=self.joint_num-1))
        self.mask = nn.Parameter(torch.FloatTensor(self.pose_corrective_mask))

    def kill_pose_dof(self, hand_pose):
        batch_size = hand_pose.shape[0]
        pose_dof_mask = torch.FloatTensor(self.pose_dof_mask).cuda()[None,:,:].repeat(batch_size,1,1)
        pose_dof_mask = torch.cat((pose_dof_mask[:,:self.root_joint_idx], pose_dof_mask[:,self.root_joint_idx+1:]),1) # remove root joint
        hand_pose = torch.flip(matrix_to_euler_angles(hand_pose, 'ZYX'), [2])
        hand_pose = hand_pose * pose_dof_mask
        hand_pose = euler_angles_to_matrix(torch.flip(hand_pose, [2]), 'ZYX')
        hand_pose = matrix_to_rotation_6d(hand_pose)
        return hand_pose

    def subtract_zero_pose(self, hand_pose):
        batch_size = hand_pose.shape[0]
        zero_hand_pose = torch.eye(3).float().cuda()[None,None,:,:].repeat(batch_size,self.joint_num-1,1,1)
        zero_hand_pose = matrix_to_rotation_6d(zero_hand_pose)
        return hand_pose - zero_hand_pose

    def add_neighbor_joints(self, hand_pose):
        hand_pose = torch.stack([hand_pose[:,self.neighbor_joint_idxs[j],:] for j in range(self.joint_num-1)],1)
        hand_pose = hand_pose.view(-1,self.joint_num-1,3,6)
        hand_pose = hand_pose * torch.FloatTensor(self.neighbor_joint_mask).cuda().view(1,self.joint_num-1,3,1)
        hand_pose = hand_pose.view(-1,self.joint_num-1,18)
        return hand_pose

    def forward(self, hand_pose, id_code):
        batch_size = hand_pose.shape[0]

        # corrective from pose and id code
        hand_pose = hand_pose.view(batch_size,self.joint_num-1,3,3)
        hand_pose = self.kill_pose_dof(hand_pose)
        hand_pose = self.subtract_zero_pose(hand_pose)
        hand_pose = self.add_neighbor_joints(hand_pose)
        feat = torch.cat((hand_pose, id_code[:,None,:].repeat(1,self.joint_num-1,1)),2)
        feat = feat.view(batch_size,(self.joint_num-1)*(18+self.id_code_dim),1)
        vertex_pose_corrective_per_joint = self.vertex(feat).view(batch_size,self.joint_num-1,self.vertex_num,3) / 1000
        
        # subtract the corrective from zero pose
        zeros = torch.zeros((batch_size,self.joint_num-1,6)).float().cuda()
        zeros = self.add_neighbor_joints(zeros)
        feat = torch.cat((zeros, id_code[:,None,:].repeat(1,self.joint_num-1,1)),2)
        feat = feat.view(batch_size,(self.joint_num-1)*(18+self.id_code_dim),1)
        vertex_pose_corrective_per_joint_zero_pose = self.vertex(feat).view(batch_size,self.joint_num-1,self.vertex_num,3) / 1000
        vertex_pose_corrective_per_joint = vertex_pose_corrective_per_joint - vertex_pose_corrective_per_joint_zero_pose
        
        # aggregate corrective per joint with mask
        pose_corrective_mask = F.relu(self.mask).view(1,self.joint_num-1,self.vertex_num,1)
        vertex_pose_corrective = (vertex_pose_corrective_per_joint * pose_corrective_mask).sum(1)
        return vertex_pose_corrective

class HandPoser_Decoder(nn.Module):
    def __init__(self, pose_code_dim, joint_num):
        super().__init__()
        self.pose_code_dim = pose_code_dim
        self.joint_num = joint_num
        self.decoder = nn.Sequential(nn.Linear(self.pose_code_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, 128), nn.ReLU(inplace=True), nn.Linear(128,(self.joint_num-1)*6))

    def forward(self, pose_code):
        batch_size = pose_code.shape[0]
        pose = self.decoder(pose_code).view(batch_size,self.joint_num-1,6)
        pose = rotation_6d_to_matrix(pose)
        return pose

class UHM(nn.Module):
    def __init__(self, root_path):
        super(UHM, self).__init__()
        # template mesh
        self.vertex, self.face, self.vertex_uv, self.face_uv = self.load_template(osp.join(root_path, 'template.obj'))
        self.vertex_num = len(self.vertex)

        # skeleton and surface
        self.joints_name = ['Wrist', 'Thumb_0', 'Thumb_1', 'Thumb_2', 'Thumb_3', 'Thumb_4', 'Index_1', 'Index_2', 'Index_3', 'Index_4', 'Middle_1', 'Middle_2', 'Middle_3', 'Middle_4', 'Ring_1', 'Ring_2', 'Ring_3', 'Ring_4', 'Pinky_0', 'Pinky_1', 'Pinky_2', 'Pinky_3', 'Pinky_4', 'Forearm']
        self.joint_num = len(self.joints_name)
        self.child_joint_idxs = ([1,6,10,14,18,23], [2], [3], [4], [5], [], [7], [8], [9], [], [11], [12], [13], [], [15], [16], [17], [], [19], [20], [21], [22], [], [])
        self.root_joint_idx = self.joints_name.index('Wrist')
        with open(osp.join(root_path, 'template.pkl'), 'rb') as f:
            template_info = pickle.load(f, encoding='latin1')
        self.local_pose_template = template_info['local_pose']
        self.local_pose_template[:,:3,3] /= 1000 # millimter to meter
        self.global_pose_template = template_info['global_pose']
        self.global_pose_template[:,:3,3] /= 1000 # millimter to meter
        self.skinning_weight = template_info['skinning_weight']
        self.joint = self.global_pose_template[:,:3,3]
        self.pose_dof_mask = self.make_pose_dof_mask()

        # decoders
        self.id_code_dim = 32 
        self.pose_code_dim = 32
        self.neighbor_joint_idxs = ([0,0,1], [0,1,2], [1,2,3], [2,3,4], [3,4,0], [0,5,6], [5,6,7], [6,7,8], [7,8,0], [0,9,10], [9,10,11], [10,11,12], [11,12,0], [0,13,14], [13,14,15], [14,15,16], [15,16,0], [0,17,18], [17,18,19], [18,19,20], [19,20,21], [20,21,0], [0,22,0]) # without root joint
        self.neighbor_joint_mask = ([0,1,1], [1,1,1], [1,1,1], [1,1,1], [0,0,0], [0,1,1], [1,1,1], [1,1,1], [0,0,0], [0,1,1], [1,1,1], [1,1,1], [0,0,0], [0,1,1], [1,1,1], [1,1,1], [0,0,0], [0,1,1], [1,1,1], [1,1,1], [1,1,1], [0,0,0], [0,0,0]) # without root joint
        self.pose_corrective_mask = self.make_pose_corrective_mask(100)
        self.joints_name_id_decoder = ('Index_1_X', 'Middle_1_X', 'Ring_1_X', 'Pinky_1_X', 'Thumb_1_X', 'Index_1_Z', 'Middle_1_Z', 'Ring_1_Z', 'Pinky_1_Z', 'Thumb_1_Z', 'Index_2_X', 'Middle_2_X', 'Ring_2_X', 'Pinky_2_X', 'Thumb_2_X', 'Index_3_X', 'Middle_3_X', 'Ring_3_X', 'Pinky_3_X', 'Thumb_3_X', 'Index_4_X', 'Middle_4_X', 'Ring_4_X', 'Pinky_4_X', 'Thumb_4_X')
        self.id_decoder = ID_Decoder(self.id_code_dim, self.vertex_num, self.joint_num, self.joints_name, self.joints_name_id_decoder)
        self.pose_decoder = Pose_Decoder(self.joint_num, self.id_code_dim, self.vertex_num, self.pose_corrective_mask, self.pose_dof_mask, self.neighbor_joint_idxs, self.neighbor_joint_mask, self.root_joint_idx)
        self.handposer_decoder = HandPoser_Decoder(self.pose_code_dim, self.joint_num)
        self.load_state_dict(torch.load(osp.join(root_path, 'ckpt.pth')), strict=True)
 
    def load_template(self, path):
        vertex_xyz, face, aux = load_obj(path)
        vertex_xyz = vertex_xyz.numpy().astype(np.float32) / 100 # centimeter to meter. (V, 3)
        face_xyz = face.verts_idx.numpy().astype(np.int64) # (F, 3). 0-based
        vertex_uv = aux.verts_uvs.numpy().astype(np.float32) # (V`, 2)
        face_uv = face.textures_idx.numpy().astype(np.int64) # (F, 3). 0-based
        return vertex_xyz, face_xyz, vertex_uv, face_uv

    def make_pose_dof_mask(self):
        dof_mask = []
        for j in range(self.joint_num):
            if '4' in self.joints_name[j]: # fingertips
                dof_mask.append([0,0,0])
            else:
                dof_mask.append([1,1,1])
        dof_mask = np.array(dof_mask, dtype=np.float32).reshape(self.joint_num,3)
        return dof_mask
       
    def make_pose_corrective_mask(self, pose_corr_mask_scale):
        joint = np.concatenate((self.joint[:self.root_joint_idx,:], self.joint[self.root_joint_idx+1:,:])) # remove root joint

        joint_label = np.argmax(self.skinning_weight, 1)
        mask = np.zeros((self.joint_num-1,self.vertex_num), dtype=np.float32)
        for v in range(self.vertex_num):
            cur_joint_name = self.joints_name[joint_label[v]]
            if ('2' in cur_joint_name) or ('3' in cur_joint_name) or ('4' in cur_joint_name):
                candidates = [cur_joint_name[:-1] + str(i) for i in range(1,5)]
                if cur_joint_name[:-1] + '0' in self.joints_name:
                    candidates += [cur_joint_name[:-1] + '0']
            else:
                candidates = [x for x in self.joints_name if ('1' in x) or ('0' in x)]
                if cur_joint_name[:-1] + '2' in self.joints_name:
                    candidates += [cur_joint_name[:-1] + '2']

            for name in candidates:
                joint_idx = (self.joints_name[:self.root_joint_idx] + self.joints_name[self.root_joint_idx+1:]).index(name) # index without root joint
                mask[joint_idx][v] = 1 / np.sqrt(np.sum((self.vertex[v] - joint[joint_idx])**2))
        
        mask = mask / pose_corr_mask_scale
        return mask

    def forward_kinematics(self, child_joint_idxs, cur_joint_idx, local_pose, global_pose):
        child_joint_idx_list = child_joint_idxs[cur_joint_idx]
        if len(child_joint_idx_list) == 0:
            return global_pose

        for joint_idx in child_joint_idx_list:
            global_pose = torch.cat((global_pose[:,:joint_idx],\
                    torch.bmm(global_pose[:,cur_joint_idx], local_pose[:,joint_idx])[:,None],\
                    global_pose[:,joint_idx+1:]),1)
            global_pose = self.forward_kinematics(child_joint_idxs, joint_idx, local_pose, global_pose)
        return global_pose
    
    def decode_pose_code(self, pose_code):
        hand_pose = self.handposer_decoder(pose_code)
        return hand_pose

    def forward(self, root_pose, hand_pose, id_code, trans=None):
        batch_size = root_pose.shape[0]

        # forward to decoders
        global_id_scale, skeleton_id_corrective, vertex_id_corrective = self.id_decoder(id_code)
        vertex_pose_corrective = self.pose_decoder(hand_pose.detach(), id_code.detach())

        # adjust local pose
        local_pose_template = torch.FloatTensor(self.local_pose_template).cuda()[None,:,:,:].repeat(batch_size,1,1,1)
        local_pose_template[:,:,:3,3] = (local_pose_template[:,:,:3,3] + skeleton_id_corrective) * global_id_scale[:,None,:]

        # adjust global pose
        global_pose_template = torch.FloatTensor(self.global_pose_template).cuda()[None,:,:,:].repeat(batch_size,1,1,1)
        global_pose_template_tmp = global_pose_template.clone()
        global_pose_template_tmp[:,:,:3,3] *= global_id_scale[:,None,:]
        global_pose_inv_template = torch.inverse(global_pose_template_tmp)

        # adjust template mesh
        mesh_template = torch.FloatTensor(self.vertex).cuda()[None,:,:].repeat(batch_size,1,1)
        mesh_template = (mesh_template + vertex_id_corrective + vertex_pose_corrective) * global_id_scale[:,None,:] # mesh_template is already root-relative. we can just multiply global_id_scale without calcelling the translation

        # combine root pose and hand pose and make it 4x4 matrix
        pose = torch.cat((root_pose[:,None], hand_pose),1)
        pose = torch.cat((pose, torch.FloatTensor([0,0,0]).cuda().view(1,1,3,1).repeat(batch_size,self.joint_num,1,1)),3)
        pose = torch.cat((pose, torch.FloatTensor([0,0,0,1]).cuda().view(1,1,1,4).repeat(batch_size,self.joint_num,1,1)),2)

        # forward kinematics
        global_pose = global_pose_template.clone()
        global_root_pose = torch.bmm(pose[:,self.root_joint_idx], global_pose[:,self.root_joint_idx])
        global_pose = torch.cat((global_pose[:,:self.root_joint_idx], global_root_pose[:,None], global_pose[:,self.root_joint_idx+1:]),1)
        global_pose = self.forward_kinematics(self.child_joint_idxs, \
                        self.root_joint_idx, \
                        torch.bmm(local_pose_template.view(-1,4,4), pose.view(-1,4,4)).view(batch_size,self.joint_num,4,4), \
                        global_pose)
        joint = global_pose[:,:,:3,3]
        joint_trans_mat = torch.bmm(global_pose.view(batch_size*self.joint_num,4,4), global_pose_inv_template.view(batch_size*self.joint_num,4,4))

        # LBS
        mesh_template_xyz1 = torch.cat([mesh_template, torch.ones_like(mesh_template[:,:,:1])],2)
        mesh_template_xyz1 = mesh_template_xyz1[:,None,:,:].repeat(1,self.joint_num,1,1).reshape(batch_size*self.joint_num,self.vertex_num,4)
        skinning_weight = torch.FloatTensor(self.skinning_weight).cuda()[None,:,:].repeat(batch_size,1,1)
        skinning_weight = skinning_weight.permute(0,2,1).reshape(batch_size*self.joint_num,self.vertex_num,1)
        mesh = skinning_weight * torch.bmm(joint_trans_mat, mesh_template_xyz1.permute(0,2,1)).permute(0,2,1)[:,:,:3]
        mesh = mesh.view(batch_size,self.joint_num,self.vertex_num,3).sum(1)

        # global translation
        if trans is not None:
            mesh = mesh + trans[:,None,:]
            joint = joint + trans[:,None,:]
        
        return mesh, joint


