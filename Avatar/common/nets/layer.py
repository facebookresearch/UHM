# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.uhm import uhm
from utils.transforms import rasterize_to_xy, rasterize_to_uv
from pytorch3d.structures import Meshes
from pytorch3d.renderer import PerspectiveCameras, RasterizationSettings, MeshRasterizer, RasterizationSettings, TexturesUV, SoftSilhouetteShader, BlendParams
from config import cfg

def make_linear_layers(feat_dims, relu_final=True, use_bn=False, bias=True):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(nn.Linear(feat_dims[i], feat_dims[i+1], bias=bias))

        # Do not use ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and relu_final):
            if use_bn:
                layers.append(nn.BatchNorm1d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

def make_conv_layers(feat_dims, kernel=3, stride=1, padding=1, bnrelu_final=True, norm='bn', act='relu'):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(
            nn.Conv2d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i+1],
                kernel_size=kernel,
                stride=stride,
                padding=padding
                ))
        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
            if norm == 'bn':
                layers.append(nn.BatchNorm2d(feat_dims[i+1]))
            elif norm == 'gn':
                layers.append(nn.GroupNorm(8, feat_dims[i+1]))
            elif norm is None:
                pass
            else:
                assert 0, 'Not implemented'

            if act == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif act == 'silu':
                layers.append(nn.SiLU(inplace=True))
            else:
                assert 0, 'Not implemented'

    return nn.Sequential(*layers)

class SoftMaskRenderer(nn.Module):
    def __init__(self):
        super(SoftMaskRenderer, self).__init__()
        self.blend_params = BlendParams(background_color=(0,0,0), sigma=1e-7, gamma=1e-1)
        self.soft_rasterizer = SoftSilhouetteShader(blend_params=self.blend_params)
    
    def get_rasterizer(self, cam_param, render_shape, set_faces_per_bin):
        cameras = PerspectiveCameras(focal_length=cam_param['focal'],
                                    principal_point=cam_param['princpt'],
                                    device='cuda',
                                    in_ndc=False,
                                    image_size=torch.LongTensor(render_shape).cuda().view(1,2))

        if set_faces_per_bin:
            max_faces_per_bin = 20000
        else:
            max_faces_per_bin = None
        raster_settings = RasterizationSettings(
            image_size=render_shape,
            blur_radius=np.log(1. / 1e-4 - 1.)*self.blend_params.sigma,
            faces_per_pixel=50,
            max_faces_per_bin=max_faces_per_bin
        )
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings).cuda()
        return rasterizer

    def forward(self, mesh, face, cam_param, render_shape, set_faces_per_bin=False):
        batch_size = mesh.shape[0]
        render_height, render_width = render_shape
        if 'R' not in cam_param:
            cam_param['R'] = torch.eye(3)[None,:,:].float().cuda().repeat(batch_size,1,1)
        if 't' not in cam_param:
            cam_param['t'] = torch.zeros((batch_size,3)).float().cuda()

        # get visible faces from mesh
        mesh = torch.bmm(cam_param['R'], mesh.permute(0,2,1)).permute(0,2,1) + cam_param['t'].view(-1,1,3) # world coordinate -> camera coordinate
        face = torch.from_numpy(face).cuda()[None,:,:].repeat(batch_size,1,1)
        mesh = torch.stack((-mesh[:,:,0], -mesh[:,:,1], mesh[:,:,2]),2) # reverse x- and y-axis following PyTorch3D axis direction
        mesh = Meshes(mesh, face)
        rasterizer = self.get_rasterizer(cam_param, render_shape, set_faces_per_bin)
        fragments = rasterizer(mesh)

        # render mask in a differentiable way
        mask = self.soft_rasterizer(fragments, mesh).permute(0,3,1,2)[:,3:,:,:]
        return mask

class DepthmapRenderer(nn.Module):
    def __init__(self):
        super(DepthmapRenderer, self).__init__()
    
    def forward(self, mesh, face, cam_param, render_shape, set_faces_per_bin=False):
        batch_size = mesh.shape[0]
        render_height, render_width = render_shape
        if 'R' not in cam_param:
            cam_param['R'] = torch.eye(3)[None,:,:].float().cuda().repeat(batch_size,1,1)
        if 't' not in cam_param:
            cam_param['t'] = torch.zeros((batch_size,3)).float().cuda()

        # get visible faces from mesh
        mesh = torch.bmm(cam_param['R'], mesh.permute(0,2,1)).permute(0,2,1) + cam_param['t'].view(-1,1,3) # world coordinate -> camera coordinate
        fragments = rasterize_to_xy(mesh, face, cam_param, render_shape, set_faces_per_bin)
        depthmap = fragments.zbuf.permute(0,3,1,2)
        return depthmap

class UVRenderer(nn.Module):
    def __init__(self, vertex_uv, face_uv):
        super(UVRenderer, self).__init__()
        self.vertex_uv = torch.FloatTensor(vertex_uv).cuda()
        self.face_uv = torch.LongTensor(face_uv).cuda()
    
    def forward(self, mesh, face, cam_param, render_shape, uvmap, bkg_color=[0,0,0], set_faces_per_bin=False):
        batch_size = mesh.shape[0]
        render_height, render_width = render_shape
        if 'R' not in cam_param:
            cam_param['R'] = torch.eye(3)[None,:,:].float().cuda().repeat(batch_size,1,1)
        if 't' not in cam_param:
            cam_param['t'] = torch.zeros((batch_size,3)).float().cuda()

        # get visible faces from mesh
        mesh = torch.bmm(cam_param['R'], mesh.permute(0,2,1)).permute(0,2,1) + cam_param['t'].view(-1,1,3) # world coordinate -> camera coordinate
        fragments = rasterize_to_xy(mesh, face, cam_param, render_shape, set_faces_per_bin)

        # render from UV texture map
        vertex_uv = torch.stack((self.vertex_uv[:,0], 1 - self.vertex_uv[:,1]),1)[None,:,:].repeat(batch_size,1,1) # flip y-axis following PyTorch3D convention
        renderer = TexturesUV(uvmap.permute(0,2,3,1), self.face_uv[None,:,:].repeat(batch_size,1,1), vertex_uv)
        render = renderer.sample_textures(fragments) # batch_size, render_height, render_width, faces_per_pixel, uvmap_dim
        render = render[:,:,:,0,:].permute(0,3,1,2) # batch_size, uvmap_dim, render_height, render_width
        
        # fg mask
        pix_to_face = fragments.pix_to_face # batch_size, render_height, render_width, faces_per_pixel. invalid: -1
        pix_to_face_xy = pix_to_face[:,:,:,0] # Note: this is a packed representation

        # packed -> unpacked
        is_valid = (pix_to_face_xy != -1).float()
        pix_to_face_xy = (pix_to_face_xy - torch.arange(batch_size)[:,None,None].cuda() * face.shape[0]) * is_valid + (-1) * (1 - is_valid)
        pix_to_face_xy = pix_to_face_xy.long()
        
        # set backgroud color
        is_valid = is_valid[:,None,:,:]
        render = render * is_valid + torch.FloatTensor(bkg_color)[None,:,None,None].cuda() * (1 - is_valid)
        return render, is_valid

class Vert2UV(nn.Module):
    def __init__(self, uvmap_shape):
        super(Vert2UV, self).__init__()
        vertex_uv = torch.from_numpy(uhm.vertex_uv).float().cuda()[None,:,:]
        face_uv = torch.from_numpy(uhm.face_uv).long().cuda()[None,:,:]
        fragments = rasterize_to_uv(vertex_uv, face_uv, (uvmap_shape[0], uvmap_shape[1]))
        pix_to_face = fragments.pix_to_face # batch_size, uvmap_shape[0], uvmap_shape[1], faces_per_pixel. invalid: -1
        bary_coords = fragments.bary_coords # batch_size, uvmap_shape[0], uvmap_shape[1], faces_per_pixel, 3. invalid: -1
        self.pix_to_face_uv = pix_to_face[0,:,:,0]
        self.bary_coords_uv = bary_coords[0,:,:,0,:]
        self.uvmap_shape = uvmap_shape
    
    def forward(self, vert_feat):
        batch_size, _, feat_dim = vert_feat.shape
        is_valid = (self.pix_to_face_uv != -1)
        pix_to_vert_idx_uv = torch.from_numpy(uhm.face).long().cuda()[self.pix_to_face_uv,:].view(self.uvmap_shape[0], self.uvmap_shape[1], 3)
        vert_feat_uv = vert_feat[:,pix_to_vert_idx_uv.view(-1),:].view(batch_size,self.uvmap_shape[0],self.uvmap_shape[1],3,feat_dim)
        vert_feat_uv = torch.sum(vert_feat_uv * self.bary_coords_uv[None,:,:,:,None], dim=3) # batch_size, self.uvmap_shape[0], self.uvmap_shape[1], feat_dim
        vert_feat_uv = vert_feat_uv.permute(0,3,1,2) # batch_size, feat_dim, self.uvmap_shape[0], self.uvmap_shape[1]
        vert_feat_uv = vert_feat_uv * is_valid[None,None,:,:]
        return vert_feat_uv

class XY2UV(nn.Module):
    def __init__(self, uvmap_shape):
        super(XY2UV, self).__init__()
        vertex_uv = torch.from_numpy(uhm.vertex_uv).float().cuda()[None,:,:]
        face_uv = torch.from_numpy(uhm.face_uv).long().cuda()[None,:,:]
        fragments = rasterize_to_uv(vertex_uv, face_uv, uvmap_shape)
        pix_to_face = fragments.pix_to_face # batch_size, uvmap_shape[0], uvmap_shape[1], faces_per_pixel. invalid: -1
        bary_coords = fragments.bary_coords # batch_size, uvmap_shape[0], uvmap_shape[1], faces_per_pixel, 3. invalid: -1
        self.pix_to_face_uv = pix_to_face[0,:,:,0]
        self.bary_coords_uv = bary_coords[0,:,:,0,:]
        self.uvmap_shape = uvmap_shape

    def forward(self, img, mesh, face, cam_param, set_faces_per_bin=False):
        batch_size, channel_dim, img_height, img_width = img.shape
        if 'R' not in cam_param:
            cam_param['R'] = torch.eye(3)[None,:,:].float().cuda().repeat(batch_size,1,1)
        if 't' not in cam_param:
            cam_param['t'] = torch.zeros((batch_size,3)).float().cuda()

        # project mesh
        mesh_cam = torch.bmm(cam_param['R'], mesh.permute(0,2,1)).permute(0,2,1) + cam_param['t'].view(-1,1,3) # world coordinate -> camera coordinate
        x = mesh_cam[:,:,0] / mesh_cam[:,:,2] * cam_param['focal'][:,None,0] + cam_param['princpt'][:,None,0]
        y = mesh_cam[:,:,1] / mesh_cam[:,:,2] * cam_param['focal'][:,None,1] + cam_param['princpt'][:,None,1]
        mesh_img = torch.stack((x,y),2)

        # get visible faces from mesh
        fragments = rasterize_to_xy(mesh_cam, face, cam_param, (img_height, img_width), set_faces_per_bin)
        pix_to_face = fragments.pix_to_face # batch_size, img_height, img_width, faces_per_pixel. invalid: -1
        pix_to_face_xy = pix_to_face[:,:,:,0] # Note: this is a packed representation!
        
        # get 2D coordinates of visible vertices
        mesh_img_0, mesh_img_1, mesh_img_2, invisible_uv = [], [], [], []
        for i in range(batch_size):
            # get visible face idxs
            visible_faces = torch.unique(pix_to_face_xy[i])
            visible_faces[visible_faces != -1] = visible_faces[visible_faces != -1] - face.shape[0] * i # packed -> unpacked
            valid_face_mask = torch.zeros((face.shape[0])).float().cuda()
            valid_face_mask[visible_faces] = 1.0
            
            # mask idxs of invisible vertices to -1
            _face = torch.from_numpy(face).cuda()
            _face[valid_face_mask==0,:] = -1
            vertex_idx_0 = _face[self.pix_to_face_uv.view(-1),0].view(self.uvmap_shape[0],self.uvmap_shape[1])
            vertex_idx_1 = _face[self.pix_to_face_uv.view(-1),1].view(self.uvmap_shape[0],self.uvmap_shape[1])
            vertex_idx_2 = _face[self.pix_to_face_uv.view(-1),2].view(self.uvmap_shape[0],self.uvmap_shape[1])
            invisible_uv.append((vertex_idx_0 == -1))
            
            # get 2D coordinates
            mesh_img_0.append(mesh_img[i,vertex_idx_0.view(-1),:].view(self.uvmap_shape[0], self.uvmap_shape[1], 2))
            mesh_img_1.append(mesh_img[i,vertex_idx_1.view(-1),:].view(self.uvmap_shape[0], self.uvmap_shape[1], 2))
            mesh_img_2.append(mesh_img[i,vertex_idx_2.view(-1),:].view(self.uvmap_shape[0], self.uvmap_shape[1], 2))
        mesh_img_0 = torch.stack(mesh_img_0) # batch_size, self.uvmap_shape[0], self.uvmap_shape[1], 2
        mesh_img_1 = torch.stack(mesh_img_1)
        mesh_img_2 = torch.stack(mesh_img_2)
        invisible_uv = torch.stack(invisible_uv).float() # batch_size, self.uvmap_shape[0], self.uvmap_shape[1]

        # prepare coordinates to perform grid_sample
        mesh_img = mesh_img_0 * self.bary_coords_uv[None,:,:,0,None] + mesh_img_1 * self.bary_coords_uv[None,:,:,1,None] + mesh_img_2 * self.bary_coords_uv[None,:,:,2,None]
        mesh_img = torch.stack((mesh_img[:,:,:,0]/(img_width-1)*2-1, mesh_img[:,:,:,1]/(img_height-1)*2-1), 3) # [-1,1] normalization
       
        # fill UV map
        uvmap = F.grid_sample(img, mesh_img, align_corners=True)
        uvmap[self.pix_to_face_uv[None,None,:,:].repeat(batch_size,channel_dim,1,1) == -1] = -1
        uvmap = uvmap * (1 - invisible_uv)[:,None,:,:] - invisible_uv[:,None,:,:]
        return uvmap

