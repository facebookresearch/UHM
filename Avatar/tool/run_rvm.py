# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import torch
from model import MattingNetwork
from inference import convert_video
import argparse
import cv2
import os
import os.path as osp
from glob import glob
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, dest='root_path')
    args = parser.parse_args()
    assert args.root_path, "Please set root_path."
    return args

args = parse_args()
root_path = args.root_path

# load model
model = MattingNetwork('resnet50').eval().cuda()
model.load_state_dict(torch.load('rvm_resnet50.pth'))

seq_path_list = [x for x in glob(osp.join(root_path, '*')) if osp.isdir(x)]
for seq_path in seq_path_list:

    save_root_path = osp.join(seq_path, 'masks')
    os.makedirs(save_root_path, exist_ok=True)
    
    # run model
    convert_video(
        model,                                              # The model, can be on any device (cpu or cuda).
        input_source=osp.join(seq_path, 'video.mp4'),       # A video file or an image sequence directory.
        output_type='png_sequence',                         # Choose "video" or "png_sequence"
        output_composition=save_root_path,                  # File path if video; directory path if png sequence.
        downsample_ratio=None,                              # A hyperparameter to adjust or use None for auto.
        seq_chunk=12,                                       # Process n frames at once for better parallelism.
    )
    
    # change masked images -> masks
    # change 0000.png -> 0.png
    output_path_list = glob(osp.join(save_root_path, '*.png'))
    img_height, img_width = cv2.imread(output_path_list[0]).shape[:2]
    for output_path in output_path_list:
        output = cv2.imread(output_path, -1)
        os.system('rm ' + output_path)

        frame_idx = int(output_path.split('/')[-1][:-4])
        cv2.imwrite(osp.join(save_root_path, str(frame_idx) + '.png'), output[:,:,3:])

    # make video
    video_out = cv2.VideoWriter(osp.join(seq_path, 'masks.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (img_width, img_height))
    frame_idx_list = sorted([int(x.split('/')[-1][:-4]) for x in glob(osp.join(save_root_path, '*.png'))])
    for frame_idx in frame_idx_list:
        out = cv2.imread(osp.join(save_root_path, str(frame_idx) + '.png'))
        out = cv2.putText(out, str(frame_idx), (int(img_width*0.1), int(img_height*0.1)), cv2.FONT_HERSHEY_PLAIN, 2.0, (0,0,255), 3)
        video_out.write(out)
    video_out.release()
