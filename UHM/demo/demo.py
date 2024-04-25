# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

uhm_root_path = '..'
import sys
sys.path.insert(0, uhm_root_path)

import torch
from uhm import UHM
from pytorch3d.io import save_obj

# initialize UHM layer
uhm_layer = UHM(uhm_root_path).cuda()

# ID latent code
# randomly sample it from the normal Gaussian distribution
id_code = torch.randn(1,uhm_layer.id_code_dim).cuda()

# zero pose
# joint name is uhm_layer.joints_name
pose = torch.eye(3).float().cuda()[None,None,:,:].repeat(1,uhm_layer.joint_num,1,1) # rotation matrix
root_pose = pose[:,uhm_layer.root_joint_idx] # 3D global orientation
hand_pose = torch.cat((pose[:,:uhm_layer.root_joint_idx], pose[:,uhm_layer.root_joint_idx+1:]),1)

# pose latent code
# randomly sample it from the normal Gaussian distribution
pose_code = torch.randn(1,uhm_layer.pose_code_dim).float().cuda()
hand_pose_from_code = uhm_layer.decode_pose_code(pose_code) # decode the latent code to the rotation matrix

# 3D global translation
trans = torch.zeros((1,3)).float().cuda()

# get mesh and keypoints from id_code and pose
# joint has the same name as uhm_layer.joints_name
mesh, joint = uhm_layer(root_pose, hand_pose, id_code, trans)

# get mesh and keypoints from id_code and pose_code
# joint_from_code has the same name as uhm_layer.joints_name
mesh_from_code, joint_from_code = uhm_layer(root_pose, hand_pose_from_code, id_code, trans)

# save outputs
save_obj('mesh.obj', mesh[0].detach().cpu(), torch.LongTensor(uhm_layer.face))
save_obj('mesh_from_code.obj', mesh_from_code[0].detach().cpu(), torch.LongTensor(uhm_layer.face))
