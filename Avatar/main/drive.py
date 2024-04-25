# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import torch
import argparse
import numpy as np
from config import cfg
from utils.uhm import uhm
import json
import cv2
import os
import os.path as osp
from glob import glob
from tqdm import tqdm
from nets.layer import UVRenderer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_id', type=str, dest='subject_id')
    args = parser.parse_args()

    assert args.subject_id, "Please set subject id."
    return args

def main():

    args = parse_args()
    subject_id = args.subject_id
    cfg.set_args('test', subject_id)

    # camera parameters
    render_shape = (2048, 1334)
    cam_params = [\
            {'R': [[0.82409604139436798, -0.2133101099047629, 0.52475185713873385], 
                    [-0.050813395101184974, 0.89481350260251979, 0.4435389435421665], 
                    [-0.56416638807750952, -0.39218311102421055, 0.72657325438729758]],
            't': [-525.74112221873668, -483.22261864905471, 282.44602753241713],
            'focal': [5060.1925, 5053.005],
            'princpt': [571.1872, 974.18325]},

            {'R': [[0.99402306710114208, -0.04559214889210686, 0.099194243937026402],
                    [0.0570877495352939, 0.99156920400719883, -0.11632498664313579],
                    [ -0.093054451391826984, 0.12129249615672902, 0.98824541458700221]],
            't': [-90.557101324343648, 128.51548463915734, 13.564557506517362],
            'focal': [5080.657, 5077.043],
            'princpt': [614.44405, 945.43695]},

            {'R': [[0.77466111438414909, 0.25398290743090907, -0.57913110829418479],
                    [0.11889249336584827, 0.84097473618005203, 0.52785042211609889],
                    [0.6210996159090425, -0.4777595376754234, 0.62127376516138666]],
            't': [628.52129651695759, -562.05129004247544, 425.77812669437139],
            'focal': [5081.3745, 5076.878],
            'princpt': [743.85195, 947.6756]}
            ]
    for i in range(len(cam_params)):
        for k in cam_params[i].keys():
            cam_params[i][k] = torch.FloatTensor(cam_params[i][k]).cuda()
        cam_params[i]['t'] /= 1000 # millimeter to meter

    # UHM and renderer
    bkg_color = [0, 0, 0] # range: (0-1). black
    uhm_layer = uhm.layer.cuda()
    uv_renderer = UVRenderer(uhm.vertex_uv, uhm.face_uv).cuda()

    # load ID information
    with open(osp.join(cfg.result_dir, 'id_code.json')) as f:
        id_code = torch.FloatTensor(json.load(f))[None,:].cuda()
    texture = cv2.imread(osp.join(cfg.result_dir, 'unwrapped_texture_refine.png'))[:,:,::-1].transpose(2,0,1)/255
    texture = torch.from_numpy(texture).float().cuda()[None,:,:,:]

    # get frame index
    pose_param_root_path = './driving_poses/UHM/PoseParams' 
    pose_param_path_list = glob(osp.join(pose_param_root_path, '*.json'))
    frame_idx_list = sorted([int(x.split('/')[-1][:-5]) for x in pose_param_path_list])
    
    # make a video
    save_root_path = './drive_results'
    os.makedirs(save_root_path, exist_ok=True)
    video_out = cv2.VideoWriter(osp.join(save_root_path, subject_id + '.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (render_shape[1]*len(cam_params), render_shape[0]))
    for frame_idx in tqdm(frame_idx_list):

        # load pose parameter
        pose_param_path = osp.join(pose_param_root_path, str(frame_idx) + '.json')
        with open(pose_param_path) as f:
            pose_param = json.load(f)
        root_pose = torch.FloatTensor(pose_param['root_pose']).cuda()[None]
        hand_pose = torch.FloatTensor(pose_param['hand_pose']).cuda()[None]
        trans = torch.FloatTensor(pose_param['trans']).cuda()[None]
        
        img_renders = []
        with torch.no_grad():
            # forward to UHM layer
            mesh, _ = uhm_layer(root_pose, hand_pose, id_code, trans)

            # render
            for cam_idx in range(len(cam_params)):
                cam_param = {k: v[None] for k,v in cam_params[cam_idx].items()}
                img_render, _ = uv_renderer(mesh, uhm.face, cam_param, render_shape, texture, bkg_color)
                img_renders.append(img_render[0])

        # save
        img = torch.cat(img_renders, 2).cpu().numpy().transpose(1,2,0)[:,:,::-1].copy()*255
        img = img.astype(np.uint8)
        img = cv2.putText(img, str(frame_idx), (int(0.02*img.shape[1]), int(0.1*img.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 2.5, [51,51,255], 3, 2)
        video_out.write(img)

    video_out.release()

if __name__ == "__main__":
    main()
