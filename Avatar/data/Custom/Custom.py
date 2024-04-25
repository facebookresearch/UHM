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
from utils.mano import mano
import pickle
import json

class Custom(torch.utils.data.Dataset):
    def __init__(self, transform):
        if not cfg.fit_pose_to_test:
            self.root_path = osp.join('..', 'data', 'Custom', 'data', cfg.subject_id)
        else:
            self.root_path = osp.join('..', 'data', 'Custom', 'data', cfg.subject_id[:-5]) # remove '_test'
        self.transform = transform
        self.joint_set = {
                        'joint_num': 21,
                        'joints_name': ('Wrist', 'Thumb_1', 'Thumb_2', 'Thumb_3', 'Thumb_4', 'Index_1', 'Index_2', 'Index_3', 'Index_4', 'Middle_1', 'Middle_2', 'Middle_3', 'Middle_4', 'Ring_1', 'Ring_2', 'Ring_3', 'Ring_4', 'Pinky_1', 'Pinky_2', 'Pinky_3', 'Pinky_4'),
                        }
        self.img_paths, self.mask_paths, self.depthmap_paths, self.kpt_paths, self.mano_param_paths, self.cam_params, self.frame_idx_list = self.load_data()
        if cfg.use_tex:
            uhm.load_unwrapped_texture(cfg.result_dir)

    def load_data(self):
        img_paths, mask_paths, depthmap_paths, kpt_paths, mano_param_paths, cam_params, frame_idx_list = {}, {}, {}, {}, {}, {}, []
        
        if not cfg.fit_pose_to_test:
            with open(osp.join(self.root_path, 'frame_list_train.txt')) as f:
                _frame_idx_list = f.readlines()
        else:
            with open(osp.join(self.root_path, 'frame_list_test.txt')) as f:
                _frame_idx_list = f.readlines()
        _frame_idx_list = [{'seq_name': x.split()[0], 'frame_idx': x.split()[1]} for x in _frame_idx_list]

        for frame in _frame_idx_list: 
            seq_name, frame_idx = frame['seq_name'], frame['frame_idx']

            # image path
            if seq_name not in img_paths:
                img_paths[seq_name] = {}
            img_path = osp.join(self.root_path, seq_name, 'frames', str(frame_idx) + '.png')
            if not osp.isfile(img_path):
                continue
            img_paths[seq_name][frame_idx] = img_path

            # mask path
            if seq_name not in mask_paths:
                mask_paths[seq_name] = {}
            mask_path = osp.join(self.root_path, seq_name, 'masks', str(frame_idx) + '.png')
            if not osp.isfile(mask_path):
                continue
            mask_paths[seq_name][frame_idx] = mask_path
 
            # depthmap path (optional)
            if seq_name not in depthmap_paths:
                depthmap_paths[seq_name] = {}
            depthmap_path = osp.join(self.root_path, seq_name, 'depthmaps', str(frame_idx) + '.pkl')
            depthmap_paths[seq_name][frame_idx] = depthmap_path

            # keypoint path
            if seq_name not in kpt_paths:
                kpt_paths[seq_name] = {}
            kpt_path = osp.join(self.root_path, seq_name, 'keypoints', str(frame_idx) + '.json')
            if not osp.isfile(kpt_path):
                continue
            kpt_paths[seq_name][frame_idx] = kpt_path
     
            # mano parameter path
            if seq_name not in mano_param_paths:
                mano_param_paths[seq_name] = {}
            mano_param_path = osp.join(self.root_path, seq_name, 'mano_params', 'params', str(frame_idx) + '.json')
            if not osp.isfile(mano_param_path):
                continue
            mano_param_paths[seq_name][frame_idx] = mano_param_path

            # camera parameter path (optional)
            if seq_name not in cam_params:
                cam_param_path = osp.join(self.root_path, seq_name, 'cam_param.json')
                if osp.isfile(cam_param_path): # camera parameter exist
                    with open(cam_param_path) as f:
                        cam_params[seq_name] = {k: np.array(v, dtype=np.float32) for k,v in json.load(f).items()}
                else: # camera parameter not exist
                    cam_params[seq_name] = None
            
            # frame indices
            frame_idx_list.append({'seq_name': seq_name, 'frame_idx': frame_idx})

        return img_paths, mask_paths, depthmap_paths, kpt_paths, mano_param_paths, cam_params, frame_idx_list

    def __len__(self):
        return len(self.frame_idx_list)
    
    def __getitem__(self, idx):
        seq_name, frame_idx = self.frame_idx_list[idx]['seq_name'], self.frame_idx_list[idx]['frame_idx']
 
        # load 2D keypoint
        with open(self.kpt_paths[seq_name][frame_idx]) as f:
            joint_img = np.array(json.load(f), dtype=np.float32).reshape(-1,2)
        joint_img_valid = np.ones_like(joint_img[:,:1])
        joint_img = transform_joint_to_other_db(joint_img, self.joint_set['joints_name'], uhm.joints_name)
        joint_img_valid = transform_joint_to_other_db(joint_img_valid, self.joint_set['joints_name'], uhm.joints_name)
        elbow_img = np.zeros((2), dtype=np.float32)
        elbow_valid = np.zeros((1), dtype=np.float32)
        
        # get hand bbox from 2D keypoint
        bbox = get_bbox(joint_img, joint_img_valid[:,0])
        bbox = process_bbox(bbox)

        # load image
        img = load_img(self.img_paths[seq_name][frame_idx])
        img_orig = img.copy()
        img_height, img_width = img.shape[0], img.shape[1]
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

        # depthmap (optional)
        depthmap_path = self.depthmap_paths[seq_name][frame_idx]
        if osp.isfile(depthmap_path): # depthmap exist
            # depthmap load
            with open(depthmap_path, 'rb') as f:
                depthmap = pickle.load(f, encoding='latin1') # meter
                if len(depthmap.shape) == 2:
                    depthmap = depthmap[:,:,None]

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

        else: # depthmap not exist. dummpy depthmap.
            depthmap = torch.zeros((1,cfg.render_target_shape[0],cfg.render_target_shape[1])).float()

        # joint affine transformation
        joint_img_xy1 = np.concatenate((joint_img, np.ones_like(joint_img[:,:1])),1)
        joint_img = np.dot(img2bb_trans, joint_img_xy1.transpose(1,0)).transpose(1,0)
        joint_img[:,0] = joint_img[:,0] / cfg.render_target_shape[1] * cfg.joint_proj_shape[1]
        joint_img[:,1] = joint_img[:,1] / cfg.render_target_shape[0] * cfg.joint_proj_shape[0]

        # regressed 3D joint coordinates
        mano_param_path = self.mano_param_paths[seq_name][frame_idx]
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

        # camera parameter (optional)
        cam_param = self.cam_params[seq_name]
        if cam_param is not None: # camera parameter exists
            focal = cam_param['focal']
            princpt = cam_param['princpt']
        else: # camera parameter not exist. use virtual one.
            focal = np.array((2000,2000), dtype=np.float32)
            princpt = np.array((img_width/2,img_height/2), dtype=np.float32)
       
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

