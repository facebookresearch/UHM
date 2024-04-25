# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import argparse
import torch
import numpy as np
from config import cfg
from base import Tester
from utils.uhm import uhm
import json
import cv2
import os
import os.path as osp
from pytorch3d.io import save_obj

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_id', type=str, dest='subject_id')
    parser.add_argument('--test_epoch', type=str, dest='test_epoch')
    args = parser.parse_args()

    assert args.subject_id, "Please set subject id."
    assert args.test_epoch, 'Test epoch is required.'
    return args

def main():

    # argument parse and create log
    args = parse_args()
    cfg.set_args('test', args.subject_id)

    tester = Tester(args.test_epoch)
    tester._make_model()

    # forward
    with torch.no_grad():
        out = tester.model.module.get_id()
    id_code = out['id_code'].cpu().numpy()[0]
    mesh = out['mesh'].cpu()[0]

    # save
    save_root_path = cfg.result_dir
    os.makedirs(save_root_path, exist_ok=True)
    with open(osp.join(save_root_path, 'id_code.json'), 'w') as f:
        json.dump(id_code.tolist(), f)
    save_path = osp.join(save_root_path, 'personalized_zero_pose.ply')
    save_obj(save_path, mesh, torch.LongTensor(uhm.face))

if __name__ == "__main__":
    main()
