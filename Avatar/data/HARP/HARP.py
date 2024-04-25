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
from utils.preprocessing import load_img, get_bbox, process_bbox, generate_patch_image
from utils.transforms import transform_joint_to_other_db
from utils.uhm import uhm
import pickle
import json

class HARP(torch.utils.data.Dataset):
    def __init__(self, transform):
        if not cfg.fit_pose_to_test:
            self.root_path = osp.join('..', 'data', 'HARP', 'data', cfg.subject_id)
        else:
            self.root_path = osp.join('..', 'data', 'HARP', 'data', cfg.subject_id[:-5]) # remove '_test'
        self.transform = transform
        self.img_shape = (448, 448)
        self.focal = (2000, 2000)
        self.princpt = (224, 244)
        self.joint_set = {
                        'joint_num': 21,
                        'joints_name': ('Wrist', 'Thumb_1', 'Thumb_2', 'Thumb_3', 'Thumb_4', 'Index_1', 'Index_2', 'Index_3', 'Index_4', 'Middle_1', 'Middle_2', 'Middle_3', 'Middle_4', 'Ring_1', 'Ring_2', 'Ring_3', 'Ring_4', 'Pinky_1', 'Pinky_2', 'Pinky_3', 'Pinky_4'),
                        }
        self.img_paths, self.mask_paths, self.kpt_paths, self.mano_param_paths, self.frame_idx_list = self.load_data()
        if cfg.use_tex:
            uhm.load_unwrapped_texture(cfg.result_dir)

    def load_data(self):
        img_paths, mask_paths, kpt_paths, mano_param_paths, frame_idx_list = {}, {}, {}, {}, []
        
        if not cfg.fit_pose_to_test:
            seq_names = ['1', '2', '3', '4', '5']
        else:
            seq_names = ['6', '7', '8', '9']
      
        for seq_name in seq_names:
            img_paths[seq_name] = {}
            mask_paths[seq_name] = {}
            kpt_paths[seq_name] = {}
            mano_param_paths[seq_name] = {}
            _frame_idx_list = [int(x.split('/')[-1][:-4]) for x in glob(osp.join(self.root_path, seq_name, 'unscreen_cropped', '*.jpg'))]
            for frame_idx in _frame_idx_list:

                # image paths
                img_path = osp.join(self.root_path, seq_name, 'unscreen_cropped', '%04d.jpg' % frame_idx)
                if not osp.isfile(img_path):
                    continue
                img_paths[seq_name][frame_idx] = img_path

                # mask paths
                mask_path = osp.join(self.root_path, seq_name, 'mask', '%04d_mask.jpg' % frame_idx)
                if not osp.isfile(mask_path):
                    continue
                mask_paths[seq_name][frame_idx] = mask_path

                # keypoint paths
                kpt_path = osp.join(self.root_path, seq_name, 'keypoints_from_mediapipe', str(frame_idx) + '.json')
                if not osp.isfile(kpt_path):
                    continue
                kpt_paths[seq_name][frame_idx] = kpt_path
         
                # mano parameter paths
                mano_param_path = osp.join(self.root_path, seq_name, 'metro_mano_smooth', '%04d_mano.pkl' % frame_idx)
                if not osp.isfile(mano_param_path):
                    continue
                mano_param_paths[seq_name][frame_idx] = mano_param_path

                frame_idx_list.append({'seq_name': seq_name, 'frame_idx': frame_idx})

        return img_paths, mask_paths, kpt_paths, mano_param_paths, frame_idx_list

    def __len__(self):
        return len(self.frame_idx_list)
    
    def __getitem__(self, idx):
        seq_name, frame_idx = self.frame_idx_list[idx]['seq_name'], self.frame_idx_list[idx]['frame_idx']
        
        # load image
        img = load_img(self.img_paths[seq_name][frame_idx])
        img_orig = img.copy()
        img_height, img_width = img.shape[0], img.shape[1]
        bbox = np.array([0,0,img_width,img_height], dtype=np.float32)
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

        # dummy depthmap
        depthmap = torch.zeros((1,cfg.render_target_shape[0],cfg.render_target_shape[1])).float()

        # load 2D keypoint
        with open(self.kpt_paths[seq_name][frame_idx]) as f:
            joint_img = np.array(json.load(f), dtype=np.float32).reshape(-1,2)
        joint_img_valid = np.ones_like(joint_img[:,:1])
        joint_img_xy1 = np.concatenate((joint_img, np.ones_like(joint_img[:,:1])),1)
        joint_img = np.dot(img2bb_trans, joint_img_xy1.transpose(1,0)).transpose(1,0)
        joint_img[:,0] = joint_img[:,0] / cfg.render_target_shape[1] * cfg.joint_proj_shape[1]
        joint_img[:,1] = joint_img[:,1] / cfg.render_target_shape[0] * cfg.joint_proj_shape[0]
        joint_img = transform_joint_to_other_db(joint_img, self.joint_set['joints_name'], uhm.joints_name)
        joint_img_valid = transform_joint_to_other_db(joint_img_valid, self.joint_set['joints_name'], uhm.joints_name)
        elbow_img = np.zeros((2), dtype=np.float32)
        elbow_valid = np.zeros((1), dtype=np.float32)

        # load mano parameter
        with open(self.mano_param_paths[seq_name][frame_idx], 'rb') as f:
            mano_param = pickle.load(f, encoding='latin1')
        joint_cam = mano_param['joints'].reshape(-1,3) / 1000 # millimeter to meter
        joint_cam_valid = np.ones_like(joint_cam[:,:1])
        joint_cam = transform_joint_to_other_db(joint_cam, self.joint_set['joints_name'], uhm.joints_name)
        joint_cam_valid = transform_joint_to_other_db(joint_cam_valid, self.joint_set['joints_name'], uhm.joints_name)

        # camera parameters 
        focal = np.array(self.focal, dtype=np.float32)
        princpt = np.array(self.princpt, dtype=np.float32)
       
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

