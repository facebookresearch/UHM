# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import numpy as np
import torch
import torch.utils.data
import cv2
import os.path as osp
from glob import glob
from tqdm import tqdm
from config import cfg
from utils.preprocessing import get_bbox, process_bbox, generate_patch_image
from utils.transforms import transform_joint_to_other_db
from utils.uhm import uhm
from utils.mano import mano
import json
from PIL import Image
import pillow_avif

class Goliath(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.data_split = data_split
        if not cfg.fit_pose_to_test:
            self.root_path = osp.join('..', 'data', 'Goliath', 'data', cfg.subject_id)
        else:
            self.root_path = osp.join('..', 'data', 'Goliath', 'data', cfg.subject_id[:-5]) # remove '_test'
        self.transform = transform
        self.joint_set = {
                        'joint_num': 21,
                        'joints_name': ('Thumb_4', 'Thumb_3', 'Thumb_2', 'Thumb_1', 'Index_4', 'Index_3', 'Index_2', 'Index_1', 'Middle_4', 'Middle_3', 'Middle_2', 'Middle_1', 'Ring_4', 'Ring_3', 'Ring_2', 'Ring_1', 'Pinky_4', 'Pinky_3', 'Pinky_2', 'Pinky_1', 'Wrist'), 
                        'R_Elbow_idx': 8,
                        'L_Elbow_idx': 7,
                        'R_Hand_idx': [x for x in range(21,42)],
                        'L_Hand_idx': [x for x in range(42,63)]
                        }
        self.depthmap_orig_shape = (640, 360)
        self.cam_params, self.img_paths, self.mask_paths, self.depthmap_paths, self.kpts, self.mano_param_paths, self.frame_idx_list = self.load_data()
        if cfg.use_tex:
            uhm.load_unwrapped_texture(cfg.result_dir)

    def load_data(self):
        img_paths, mask_paths, depthmap_paths, kpts, mano_param_paths, frame_idx_list = {}, {}, {}, {}, {}, []

        # load frame index
        if not cfg.fit_pose_to_test:
            frame_list_path = osp.join(self.root_path, 'frame_list_train.txt')
        else:
            frame_list_path = osp.join(self.root_path, 'frame_list_test.txt')
        with open(frame_list_path) as f:
            lines = f.readlines()
            for line in lines:
                seq_name, frame_idx_min, frame_idx_max = line.split()
                for frame_idx in range(int(frame_idx_min), int(frame_idx_max)+1):
                    frame_idx_list.append({'seq_name': seq_name, 'frame_idx': frame_idx})
 
        # camera parameter
        cam_params = self.load_cam_params(osp.join(self.root_path, 'camera_calibration.txt'))
        
        for x in frame_idx_list:
            seq_name, frame_idx = x['seq_name'], x['frame_idx']
            
            # image paths
            if seq_name not in img_paths:
                img_paths[seq_name] = {}
            img_paths[seq_name][frame_idx] = osp.join(self.root_path, seq_name, 'rgb', '%06d.avif' % frame_idx)

            # mask paths
            if seq_name not in mask_paths:
                mask_paths[seq_name] = {}
            mask_paths[seq_name][frame_idx] = osp.join(self.root_path, seq_name, 'segmentations_fgbg', '%06d.png' % frame_idx)
     
            # depthmap paths
            if seq_name not in depthmap_paths:
                depthmap_paths[seq_name] = {}
            depthmap_paths[seq_name][frame_idx] = osp.join(self.root_path, seq_name, 'depth', '%06d.bin' % frame_idx)

            # load keypoints
            kpt = np.fromfile(osp.join(self.root_path, seq_name, 'keypoints_2d', '%06d.bin' % frame_idx), dtype=np.float32).reshape(-1,3)
            if seq_name not in kpts:
                kpts[seq_name] = {}
            kpts[seq_name][frame_idx] = kpt
           
            # do not use invalid frames
            joint_valid = kpt[self.joint_set['R_Hand_idx'],2]
            if np.sum(joint_valid > 0.1) == 0:
                continue

            # initial 3D estimations
            mano_param_paths[frame_idx] = osp.join(self.root_path, seq_name, 'mano', 'params', '%06d_right.json' % frame_idx)
        
        return cam_params, img_paths, mask_paths, depthmap_paths, kpts, mano_param_paths, frame_idx_list

    def __len__(self):
        return len(self.frame_idx_list)
    
    def __getitem__(self, idx):
        seq_name, frame_idx = self.frame_idx_list[idx]['seq_name'], self.frame_idx_list[idx]['frame_idx']

        # load 2D keypoint
        joint_img = self.kpts[seq_name][frame_idx]
        joint_img_valid = (joint_img[:,2:] > 0.1)
        joint_img = joint_img[:,:2]
        elbow_img = joint_img[self.joint_set['R_Elbow_idx']]
        elbow_valid = joint_img_valid[self.joint_set['R_Elbow_idx']]
        joint_img = joint_img[self.joint_set['R_Hand_idx']]
        joint_img_valid = joint_img_valid[self.joint_set['R_Hand_idx']]
        joint_img = transform_joint_to_other_db(joint_img, self.joint_set['joints_name'], uhm.joints_name)
        joint_img_valid = transform_joint_to_other_db(joint_img_valid, self.joint_set['joints_name'], uhm.joints_name)
        
        # load image
        img = Image.open(self.img_paths[seq_name][frame_idx])
        img_orig = img.copy()
        img_height, img_width = img.shape[0], img.shape[1]
        bbox = get_bbox(joint_img, joint_img_valid[:,0])
        bbox = process_bbox(bbox)
        img, img2bb_trans, bb2img_trans = generate_patch_image(img_orig, bbox, cfg.render_target_shape)
        img_orig = self.transform(img_orig.astype(np.float32))/255.
        img = self.transform(img.astype(np.float32))/255.

        # mask
        mask = cv2.imread(self.mask_paths[seq_name][frame_idx]) / 255.
        mask_orig = (mask > 0.5).astype(np.float32).copy()
        kernel = np.ones((5, 5), np.uint8)
        mask_orig = cv2.erode(mask_orig, kernel, iterations=1) # erode to prevent background color leaking during the unwrapping
        mask_orig = self.transform(mask_orig.astype(np.float32))[:1,:,:]
        mask, _, _  = generate_patch_image(mask, bbox, cfg.render_target_shape)
        mask = self.transform((mask > 0.5).astype(np.float32))[:1,:,:]
      
        # load depthmap
        depthmap = np.fromfile(self.depthmap_paths[seq_name][frame_idx], dtype=np.float32).reshape(*self.depthmap_orig_shape)[:,:,None] / 1000 # millimeter to meter
        # depthmap filter
        root_img = joint_img[uhm.root_joint_idx].copy()
        root_img[0] = root_img[0] / img_width * depthmap.shape[1]
        root_img[1] = root_img[1] / img_height * depthmap.shape[0]
        root_img[0] = min(depthmap.shape[1] - 1, max(0, root_img[0]))
        root_img[1] = min(depthmap.shape[0] - 1, max(0, root_img[1]))
        root_depth = float(depthmap[int(root_img[1]), int(root_img[0])])
        depthmap[depthmap > float(root_depth) + cfg.hand_size] = 0
        depthmap[depthmap < float(root_depth) - cfg.hand_size] = 0

        # depthmap crop and resize
        bbox_depthmap = bbox.reshape(2,2).copy()
        bbox_depthmap[:,0] = bbox_depthmap[:,0] / img_width * depthmap.shape[1]
        bbox_depthmap[:,1] = bbox_depthmap[:,1] / img_height * depthmap.shape[0]
        bbox_depthmap = bbox_depthmap.reshape(4)
        depthmap, _, _ = generate_patch_image(depthmap, bbox_depthmap, cfg.render_target_shape)
        depthmap = self.transform(depthmap.astype(np.float32))
      
        # joint affine transformation
        joint_img_xy1 = np.concatenate((joint_img, np.ones_like(joint_img[:,:1])),1)
        joint_img = np.dot(img2bb_trans, joint_img_xy1.transpose(1,0)).transpose(1,0)
        joint_img[:,0] = joint_img[:,0] / cfg.render_target_shape[1] * cfg.joint_proj_shape[1]
        joint_img[:,1] = joint_img[:,1] / cfg.render_target_shape[0] * cfg.joint_proj_shape[0]

        # elbow affine transformation
        x, y = elbow_img
        elbow_img_xy1 = np.array([x,y,1], dtype=np.float32)
        x, y = np.dot(img2bb_trans, elbow_img_xy1)
        x = x / cfg.render_target_shape[1] * cfg.joint_proj_shape[1]
        y = y / cfg.render_target_shape[0] * cfg.joint_proj_shape[0]
        elbow_img = np.array([x,y], dtype=np.float32)
        
        # regressed 3D joint coordinates
        mano_param_path = self.mano_param_paths[frame_idx]
        with open(mano_param_path) as f:
            mano_param = json.load(f)
        root_pose = torch.FloatTensor(mano_param['root_pose']).view(1,-1)
        hand_pose = torch.FloatTensor(mano_param['hand_pose']).view(1,-1)
        shape = torch.FloatTensor(mano_param['shape']).view(1,-1)
        with torch.no_grad():
            output = mano.layer(global_orient=root_pose, hand_pose=hand_pose, betas=shape)
        mesh_cam = output.vertices[0].numpy()
        joint_cam = np.dot(mano.joint_regressor, mesh_cam)
        joint_cam_valid = np.ones_like(joint_cam[:,:1])
        joint_cam = transform_joint_to_other_db(joint_cam, mano.joints_name, uhm.joints_name)
        joint_cam_valid = transform_joint_to_other_db(joint_cam_valid, mano.joints_name, uhm.joints_name)

        # camera parameters (depthmap camera parameters are just downsampled ones from the rgb camera prameters. just use rgb parameters)
        focal = self.cam_params['rgb']['focal']
        princpt = self.cam_params['rgb']['princpt']
       
        # modify intrincis to directly render to cfg.render_target_shape space
        if (cfg.mode == 'train') or (cfg.mode == 'eval'):
            focal = np.array([focal[0] / bbox[2] * cfg.render_target_shape[1], focal[1] / bbox[3] * cfg.render_target_shape[0]], dtype=np.float32)
            princpt = np.array([(princpt[0] - bbox[0]) / bbox[2] * cfg.render_target_shape[1], (princpt[1] - bbox[1]) / bbox[3] * cfg.render_target_shape[1]], dtype=np.float32)
        cam_param = {'focal': focal, 'princpt': princpt}
       
        data = {'img': img, 'mask': mask, 'depthmap': depthmap, 'joint_img': joint_img, 'elbow_img': elbow_img, 'img2bb_trans': img2bb_trans, 'joint_img_valid': joint_img_valid, 'elbow_valid': elbow_valid, 'joint_cam': joint_cam, 'joint_cam_valid': joint_cam_valid, 'cam_param': cam_param, 'seq_name': seq_name, 'frame_idx': frame_idx}
        if cfg.mode == 'test':
            data['img_orig'] = img_orig
            data['mask_orig'] = mask_orig
        return data

    def load_cam_params(self, path):
        cam_params = {'rgb': {}, 'depth': {}}
        with open(path) as f:
            krts = f.readlines()
        cam_params['rgb']['focal'] = np.array((float(krts[1].split()[0]), float(krts[2].split()[1])), dtype=np.float32)
        cam_params['rgb']['princpt'] = np.array((float(krts[1].split()[2]), float(krts[2].split()[2])), dtype=np.float32)
        cam_params['depth']['focal'] = np.array((float(krts[10].split()[0]), float(krts[11].split()[1])), dtype=np.float32)
        cam_params['depth']['princpt'] = np.array((float(krts[10].split()[2]), float(krts[11].split()[2])), dtype=np.float32)
        return cam_params
