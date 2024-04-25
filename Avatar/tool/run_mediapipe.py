# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import cv2
import mediapipe as mp
import os
import os.path as osp
import numpy as np
from tqdm import tqdm
from glob import glob
import json
import matplotlib.pyplot as plt
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, dest='root_path')
    args = parser.parse_args()
    assert args.root_path, "Please set root_path."
    return args

def vis_keypoints_with_skeleton(img, kpt):
    skeleton = ( (0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8), (0,9), (9,10), (10,11), (11,12), (0,13), (13,14), (14,15), (15,16), (0,17), (17,18), (18,19), (19,20) )

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(skeleton) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Draw the keypoints.
    for l in range(len(skeleton)):
        i1 = skeleton[l][0]
        i2 = skeleton[l][1]
        p1 = kpt[i1,0].astype(np.int32), kpt[i1,1].astype(np.int32)
        p2 = kpt[i2,0].astype(np.int32), kpt[i2,1].astype(np.int32)
        cv2.line(
            img, p1, p2,
            color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        cv2.circle(
            img, p1,
            radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(
            img, p2,
            radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    return img

args = parse_args()
root_path = args.root_path

seq_path_list = [x for x in glob(osp.join(root_path, '*')) if osp.isdir(x)]
for seq_path in seq_path_list:

    save_root_path = osp.join(seq_path, 'keypoints')
    os.makedirs(save_root_path, exist_ok=True)

    mp_hands = mp.solutions.hands
    img_path_list = glob(osp.join(seq_path, 'frames', '*.png'))
    frame_idx_list = sorted([int(x.split('/')[-1][:-4]) for x in img_path_list])
    img_height, img_width = cv2.imread(img_path_list[0]).shape[:2]

    video_out = cv2.VideoWriter(osp.join(seq_path, 'keypoints.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (img_width, img_height))
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        for frame_idx in tqdm(frame_idx_list):
            # load img
            img_orig = cv2.imread(osp.join(seq_path, 'frames', str(frame_idx) + '.png'))

            # Flip it around y-axis for correct handedness output.
            img_input = cv2.flip(img_orig, 1)

            # Convert the BGR image to RGB before processing.
            results = hands.process(cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB))
            
            # if a hand it not detected, skip this frame
            if not results.multi_hand_landmarks:
              continue

            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
              h = results.multi_handedness[i].classification[0].label # Right, Left
              if h == 'Right':
                  xy = []
                  for j in range(len(hand_landmarks.landmark)):
                      x = hand_landmarks.landmark[j].x * img_width
                      y = hand_landmarks.landmark[j].y * img_height
                      xy.append([x,y])
                  xy = np.array(xy).reshape(-1,2)
                  xy[:,0] = img_width - xy[:,0] - 1
            
            # dump keypoint
            with open(osp.join(save_root_path, str(frame_idx) + '.json'), 'w') as f:
                json.dump(xy.tolist(), f)

            # vis keypoints
            img_vis = vis_keypoints_with_skeleton(img_orig, xy)
            
            # put frame_idx
            img_vis = cv2.putText(img_vis, str(frame_idx), (int(img_width*0.1), int(img_height*0.1)), cv2.FONT_HERSHEY_PLAIN, 2.0, (0,0,255), 3)
            video_out.write(img_vis)

    video_out.release()
