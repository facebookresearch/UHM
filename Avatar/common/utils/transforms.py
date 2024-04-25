# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import torch
import numpy as np
from config import cfg
from torch.nn import functional as F
from pytorch3d.structures import Meshes
from pytorch3d.renderer import OrthographicCameras, PerspectiveCameras, RasterizationSettings, MeshRasterizer, RasterizationSettings

def transform_joint_to_other_db(src_joint, src_name, dst_name):
    src_joint_num = len(src_name)
    dst_joint_num = len(dst_name)

    new_joint = np.zeros(((dst_joint_num,) + src_joint.shape[1:]), dtype=np.float32)
    for src_idx in range(len(src_name)):
        name = src_name[src_idx]
        if name in dst_name:
            dst_idx = dst_name.index(name)
            new_joint[dst_idx] = src_joint[src_idx]

    return new_joint

def sample_joint_features(img_feat, joint_xy):
    height, width = img_feat.shape[2:]
    x = joint_xy[:,:,0] / (width-1) * 2 - 1
    y = joint_xy[:,:,1] / (height-1) * 2 - 1
    grid = torch.stack((x,y),2)[:,:,None,:]
    img_feat = F.grid_sample(img_feat, grid, align_corners=True)[:,:,:,0] # batch_size, channel_dim, joint_num
    img_feat = img_feat.permute(0,2,1).contiguous() # batch_size, joint_num, channel_dim
    return img_feat

def rasterize_to_uv(vertex_uv, face_uv, uvmap_shape):
    # scale UV coordinates to uvmap_shape
    vertex_uv = torch.stack((vertex_uv[:,:,0] * uvmap_shape[1], vertex_uv[:,:,1] * uvmap_shape[0]),2)
    vertex_uv = torch.cat((vertex_uv, torch.ones_like(vertex_uv[:,:,:1])),2) # add dummy depth
    vertex_uv = torch.stack((-vertex_uv[:,:,0], -vertex_uv[:,:,1], vertex_uv[:,:,2]),2) # reverse x- and y-axis following PyTorch3D axis direction
    mesh = Meshes(vertex_uv, face_uv)

    cameras = OrthographicCameras(device='cuda',
                                in_ndc=False,
                                image_size=torch.LongTensor(uvmap_shape).cuda().view(1,2))
    raster_settings = RasterizationSettings(image_size=uvmap_shape, blur_radius=0.0, faces_per_pixel=1)
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings).cuda()
    fragments = rasterizer(mesh)
    return fragments

def rasterize_to_xy(mesh, face, cam_param, render_shape, set_faces_per_bin):
    batch_size = mesh.shape[0]
    face = torch.from_numpy(face).cuda()[None,:,:].repeat(batch_size,1,1)
    mesh = torch.stack((-mesh[:,:,0], -mesh[:,:,1], mesh[:,:,2]),2) # reverse x- and y-axis following PyTorch3D axis direction
    mesh = Meshes(mesh, face)
    
    cameras = PerspectiveCameras(focal_length=cam_param['focal'], 
                                principal_point=cam_param['princpt'], 
                                device='cuda',
                                in_ndc=False,
                                image_size=torch.LongTensor(render_shape).cuda().view(1,2))

    if set_faces_per_bin:
        max_faces_per_bin = 20000
    else:
        max_faces_per_bin = None
    raster_settings = RasterizationSettings(image_size=render_shape, blur_radius=0.0, faces_per_pixel=1, max_faces_per_bin=max_faces_per_bin)
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings).cuda()
    fragments = rasterizer(mesh)
    return fragments


