# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import torch
import torch.nn as nn
from torch.nn import functional as F
from nets.resnet import ResNetBackbone
from nets.layer import SoftMaskRenderer, DepthmapRenderer, UVRenderer, XY2UV
from nets.module import IDNet, ColorCalibNet, PoseNet, ShadowNet
from nets.loss import PoseLoss, VertexLoss, Joint3DLoss, VGGLoss, DepthmapLoss, ForearmLoss, TVReg
from utils.uhm import uhm
from config import cfg

class Model(nn.Module):
    def __init__(self, id_net, color_calib_net, pose_net, shadow_net):
        super(Model, self).__init__()
        # trainable modules and parameters
        self.id_net = id_net
        self.color_calib_net = color_calib_net
        self.pose_net = pose_net
        self.shadow_net = shadow_net
        if cfg.use_tex:
            self.texture = nn.Parameter(uhm.unwrapped_tex)
            self.register_buffer('texture_mask', uhm.unwrapped_tex_mask)
        if cfg.fit_pose_to_test:
            # will be initialized with snapshot_0.pth from the training stage
            self.texture = nn.Parameter(torch.zeros((3,cfg.tex_shape[0],cfg.tex_shape[1])).float()) 
            self.register_buffer('texture_mask', torch.zeros(cfg.tex_shape).float())
        self.trainable_params = [{'params': self.pose_net.parameters()}]
        self.trainable_params.append({'params': self.shadow_net.parameters()})
        # optimize id_code and texture only when cfg.fit_pose_to_test is False
        if not cfg.fit_pose_to_test:
            self.trainable_params.append({'params': self.id_net.parameters()})
            self.trainable_params.append({'params': self.color_calib_net.parameters()})
            if cfg.use_tex:
                self.trainable_params.append({'params': self.texture, 'lr': 1e-2})
 
        # UHM
        self.uhm = uhm.layer
  
        # renderer
        self.soft_mask_renderer = SoftMaskRenderer()
        self.depthmap_renderer = DepthmapRenderer()
        self.uv_renderer = UVRenderer(uhm.vertex_uv, uhm.face_uv)
        self.xy2uv = XY2UV(cfg.tex_shape)

        # loss functions
        self.pose_loss = PoseLoss()
        self.vertex_loss = VertexLoss()
        self.joint_3d_loss = Joint3DLoss()
        self.vgg_loss = VGGLoss()
        self.depthmap_loss = DepthmapLoss()
        self.forearm_loss = ForearmLoss()
        self.tv_reg = TVReg()
        self.eval_modules = [self.uhm, self.vgg_loss]
        if cfg.fit_pose_to_test:
            self.eval_modules.append(self.shadow_net)
    
    def project_coord(self, joint_cam, cam_param, do_downsample):
        x = joint_cam[:,:,0] / (joint_cam[:,:,2] + 1e-4) * cam_param['focal'][:,None,0] + cam_param['princpt'][:,None,0]
        y = joint_cam[:,:,1] / (joint_cam[:,:,2] + 1e-4) * cam_param['focal'][:,None,1] + cam_param['princpt'][:,None,1]
        if do_downsample:
            x = x / cfg.render_target_shape[1] * cfg.joint_proj_shape[1]
            y = y / cfg.render_target_shape[0] * cfg.joint_proj_shape[0]
        joint_proj = torch.stack((x,y),2)
        return joint_proj

    def forward(self, data, mode):
        batch_size = data['img'].shape[0]
        if (cfg.mode == 'train') or (cfg.mode == 'eval'):
            bkg_color = [0,0,0]
            render_shape = cfg.render_target_shape
        else:
            bkg_color = [1,1,1]
            render_shape = [int(x) for x in data['img_orig'].shape[2:]]

        # ID code (global parameter)
        id_code = self.id_net(batch_size)

        # regress pose and translation
        img_input = F.interpolate(data['img'], size=cfg.input_shape, mode='bilinear')
        depthmap_input = F.interpolate(data['depthmap'], size=cfg.input_shape, mode='bilinear')
        mask_input = F.interpolate(data['mask'], size=cfg.input_shape, mode='bilinear')
        root_pose, hand_pose, root_trans, root_pose_for_code, pose_code, root_trans_for_code = self.pose_net(img_input, depthmap_input, mask_input, data['joint_img'], data['img2bb_trans'], id_code.detach())
        hand_pose_from_code = self.uhm.handposer_decoder(pose_code)

        # get coordinates in the camera coordinate system
        mesh_cam, joint_cam = self.uhm(root_pose, hand_pose, id_code)
        trans = -joint_cam[:,uhm.root_joint_idx,:] + root_trans
        mesh_cam, joint_cam = mesh_cam + trans[:,None,:], joint_cam + trans[:,None,:]

        # get coordinates in the camera coordinate system (using the pose code and its dedicated root pose and translation)
        mesh_cam_from_code, joint_cam_from_code = self.uhm(root_pose_for_code, hand_pose_from_code, id_code.detach())
        trans_for_code = -joint_cam_from_code[:,uhm.root_joint_idx,:] + root_trans_for_code
        mesh_cam_from_code, joint_cam_from_code = mesh_cam_from_code + trans_for_code[:,None,:], joint_cam_from_code + trans_for_code[:,None,:]
     
        # project 3D -> 2D
        mesh_proj = self.project_coord(mesh_cam, data['cam_param'], mode=='train')
        joint_proj = self.project_coord(joint_cam, data['cam_param'], mode=='train')
        mesh_proj_from_code = self.project_coord(mesh_cam_from_code, data['cam_param'], mode=='train')
        joint_proj_from_code = self.project_coord(joint_cam_from_code, data['cam_param'], mode=='train')

        # render
        if (mode == 'test') or (cfg.depthmap_loss_weight > 0):
            depthmap_render = self.depthmap_renderer(mesh_cam, uhm.face, data['cam_param'], render_shape, set_faces_per_bin=(mode=='test'))
        if (mode == 'test') or (cfg.mask_loss_weight > 0):
            mask_render = self.soft_mask_renderer(mesh_cam, uhm.face, data['cam_param'], render_shape, set_faces_per_bin=(mode=='test'))
        if (mode == 'test') or (cfg.shadow_loss_weight > 0):
            # shadow estimation and render
            shadow_uv = self.shadow_net(root_pose.detach(), hand_pose.detach(), id_code.detach(), mesh_cam.detach())
            shadow_render, render_fg_mask = self.uv_renderer(mesh_cam.detach(), uhm.face, data['cam_param'], render_shape, shadow_uv, bkg_color, set_faces_per_bin=(mode=='test'))
            
            # color calibrate
            img_calib = self.color_calib_net(batch_size)[:,:,None,None] * render_fg_mask
            img_calib = img_calib + torch.FloatTensor(bkg_color)[None,:,None,None].cuda() * (1 - render_fg_mask)
        if cfg.use_tex:
            # render image
            unwrapped_tex_refine = (self.texture * uhm.uv_mask.float().cuda()[None,None,:,:]).repeat(batch_size,1,1,1)
            img_render_refine, render_fg_mask = self.uv_renderer(mesh_cam, uhm.face, data['cam_param'], render_shape, unwrapped_tex_refine, bkg_color, set_faces_per_bin=(mode=='test'))
   
        if mode == 'train':
            # loss functions
            loss = {}
            loss['pose_code_reg'] = pose_code ** 2 * 0.01
            loss['root_pose_from_code'] = torch.abs(root_pose - root_pose_for_code.detach()) * cfg.from_code_loss_weight
            loss['pose_from_code'] = self.pose_loss(hand_pose, hand_pose_from_code.detach()) * cfg.from_code_loss_weight
            loss['mesh_from_code'] = self.vertex_loss(mesh_cam, mesh_cam_from_code.detach(), joint_cam, joint_cam_from_code.detach()) * cfg.from_code_loss_weight
            loss['joint_proj'] = torch.abs(joint_proj - data['joint_img']) * data['joint_img_valid']
            loss['joint_proj_code'] = torch.abs(joint_proj_from_code - data['joint_img']) * data['joint_img_valid'] * cfg.joint_proj_code_loss_weight
            loss['joint_cam_code'] = self.joint_3d_loss(joint_cam_from_code, data['joint_cam'], data['joint_cam_valid'])
            loss['forearm'] = self.forearm_loss(joint_proj, mesh_proj, data['joint_img'], data['joint_img_valid'], data['elbow_img'], data['elbow_valid'])
            loss['forearm_code'] = self.forearm_loss(joint_proj_from_code, mesh_proj_from_code, data['joint_img'], data['joint_img_valid'], data['elbow_img'], data['elbow_valid']) * cfg.forearm_code_loss_weight

            if cfg.depthmap_loss_weight > 0:
                loss['depthmap'] = self.depthmap_loss(depthmap_render, data['depthmap']) * cfg.depthmap_loss_weight

            if cfg.mask_loss_weight > 0:
                loss['mask'] = torch.abs(mask_render - data['mask']) * cfg.mask_loss_weight

            if (cfg.shadow_loss_weight > 0) and (not cfg.fit_pose_to_test):
                loss['img_calib'] = torch.abs(img_calib - data['img']) * data['mask'] * cfg.shadow_loss_weight
                loss['img_calib_shadow'] = torch.abs(img_calib*shadow_render - data['img']) * data['mask'] * cfg.shadow_loss_weight
                loss['img_calib_shadow_vgg'] = self.vgg_loss(img_calib*shadow_render, data['img'], data['mask']) * cfg.shadow_loss_weight
                loss['shadow_reg'] = F.binary_cross_entropy(shadow_uv, torch.ones_like(shadow_uv)) * uhm.uv_mask.float().cuda()[None,None,:,:] * cfg.shadow_reg_weight
                loss['shadow_tv'] = self.tv_reg(shadow_uv, uhm.uv_mask.float().cuda()[None,None,:,:]) * cfg.shadow_tv_reg_weight

            if cfg.img_loss_weight > 0:
                loss['img'] = torch.abs(img_render_refine*shadow_render - data['img']) * data['mask'] * cfg.img_loss_weight
                loss['img_vgg'] = self.vgg_loss(img_render_refine*shadow_render, data['img'], data['mask']) * cfg.img_loss_weight
                loss['tex_tv'] = self.tv_reg(unwrapped_tex_refine, ((1 - self.texture_mask) * uhm.uv_mask.float().cuda())[None,None,:,:]) * cfg.tex_tv_reg_weight
            
            return loss

        else:
            # test output
            out = {}
            out['mesh'] = mesh_cam
            out['mesh_from_code'] = mesh_cam_from_code
            out['depthmap_render'] = depthmap_render
            out['mask_render'] = mask_render
            out['img_calib'] = img_calib
            out['shadow_uv'] = shadow_uv
            out['shadow_render'] = shadow_render
            out['joint_proj'] = joint_proj
            out['root_pose'] = root_pose
            out['hand_pose'] = hand_pose
            out['trans'] = trans
            
            if cfg.mode != 'eval':
                # unwrap texture
                unwrapped_tex = self.xy2uv(data['img_orig'], mesh_cam, uhm.face, data['cam_param'], set_faces_per_bin=(mode=='test')) * 255
                is_unwrap_fg = self.xy2uv(data['mask_orig'], mesh_cam, uhm.face, data['cam_param'], set_faces_per_bin=(mode=='test'))
                is_unwrap_valid = ((unwrapped_tex != -1) * (is_unwrap_fg == 1)).float()
                unwrapped_tex = unwrapped_tex * is_unwrap_valid

                # unwrap texture (without shadow)
                unwrapped_tex_wo_shadow = self.xy2uv(data['img_orig'] / (shadow_render + 1e-4), mesh_cam, uhm.face, data['cam_param'], set_faces_per_bin=(mode=='test')) * 255
                unwrapped_tex_wo_shadow = unwrapped_tex_wo_shadow * is_unwrap_valid

                out['unwrapped_tex'] = unwrapped_tex
                out['unwrapped_tex_wo_shadow'] = unwrapped_tex_wo_shadow
                out['unwrapped_tex_valid'] = is_unwrap_valid

            if cfg.use_tex:
                # render from unwrapped texture
                img_render, _ = self.uv_renderer(mesh_cam, uhm.face, data['cam_param'], render_shape, uhm.unwrapped_tex.cuda().repeat(batch_size,1,1,1), bkg_color, set_faces_per_bin=(mode=='test'))
                out['unwrapped_tex_refine'] = unwrapped_tex_refine
                out['img_render'] = img_render
                out['img_render_shadow'] = img_render * shadow_render
                out['img_render_refine'] = img_render_refine
                out['img_render_refine_shadow'] = img_render_refine * shadow_render
                out['render_fg_mask'] = render_fg_mask
            return out

    def get_id(self):
        out = {}

        id_code = self.id_net(1)
        zero_root_pose = torch.eye(3)[None,:,:].float().cuda()
        zero_hand_pose = torch.eye(3)[None,None,:,:].float().cuda().repeat(1,uhm.joint_num-1,1,1)
        mesh, _ = self.uhm(zero_root_pose, zero_hand_pose, id_code)

        out['id_code'] = id_code
        out['mesh'] = mesh
        return out

def init_weights(m):
    try:
        if type(m) == nn.ConvTranspose2d:
            nn.init.normal_(m.weight,std=0.001)
        elif type(m) == nn.Conv2d:
            nn.init.normal_(m.weight,std=0.001)
            nn.init.constant_(m.bias, 0)
        elif type(m) == nn.BatchNorm2d:
            nn.init.constant_(m.weight,1)
            nn.init.constant_(m.bias,0)
        elif type(m) == nn.Linear:
            nn.init.normal_(m.weight,std=0.01)
            nn.init.constant_(m.bias,0)
    except AttributeError:
        pass

def get_model(mode):
    id_net = IDNet()
    color_calib_net = ColorCalibNet()
    posenet_backbone = ResNetBackbone(50, input_channel=5)
    pose_net = PoseNet(posenet_backbone)
    shadow_net = ShadowNet()

    if mode == 'train':
        id_net.apply(init_weights)
        color_calib_net.apply(init_weights)
        pose_net.apply(init_weights)
        posenet_backbone.init_weights()
        shadow_net.apply(init_weights)

    model = Model(id_net, color_calib_net, pose_net, shadow_net)
    return model
