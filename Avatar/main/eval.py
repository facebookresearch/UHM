# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import torch
import argparse
from tqdm import tqdm
import numpy as np
from config import cfg
from base import Tester
from utils.uhm import uhm
import os
import os.path as osp
import cv2
import pickle
import json
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_id', type=str, dest='subject_id')
    parser.add_argument('--fit_pose_to_test', dest='fit_pose_to_test', action='store_true')
    parser.add_argument('--test_epoch', type=str, dest='test_epoch')
    args = parser.parse_args()

    assert args.subject_id, "Please set subject id."
    assert args.test_epoch, 'Test epoch is required.'
    return args

def main():

    # argument parse and create log
    args = parse_args()
    cfg.set_args('eval', args.subject_id, True, args.fit_pose_to_test)

    tester = Tester(args.test_epoch)
    tester._make_batch_generator()
    tester._make_model()
 
    psnr = PeakSignalNoiseRatio(data_range=1).cuda()
    ssim = StructuralSimilarityIndexMeasure(data_range=1).cuda()
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex").cuda()

    eval_result = {'image_psnr': [], 'image_lpips': [], 'image_ssim': [], 'mask_iou': [], 'depthmap_l1': []}
    for itr, data in enumerate(tqdm(tester.batch_generator)):

        # forward
        with torch.no_grad():
            out = tester.model(data, 'test')
        
        # save
        img_render = out['img_render'].cpu().numpy()
        img_render_shadow = out['img_render_shadow'].cpu().numpy()
        img_render_refine = out['img_render_refine'].cpu().numpy()
        img_render_refine_shadow = out['img_render_refine_shadow'].cpu().numpy()
        render_fg_mask = out['render_fg_mask'].cpu().numpy()
        depthmap_render = out['depthmap_render'].cpu().numpy()
        batch_size = img_render.shape[0]
        for i in range(batch_size):
            img_out = img_render_refine_shadow[i]
            img_gt = data['img'][i].cpu().numpy()
            mask_out = np.tile(render_fg_mask[i], (3,1,1))
            mask_gt = np.tile(data['mask'][i].cpu().numpy(), (3,1,1))
            
            # PSNR and SSIM
            _mask = mask_out * mask_gt
            _img_out = torch.clamp(torch.from_numpy(img_out*_mask).float().cuda()[None,:,:,:], min=0, max=1)
            _img_gt = torch.clamp(torch.from_numpy(img_gt*_mask).float().cuda()[None,:,:,:], min=0, max=1)
            eval_result['image_psnr'].append(psnr(_img_out, _img_gt).detach().cpu().numpy().mean())
            eval_result['image_ssim'].append(ssim(_img_out, _img_gt).detach().cpu().numpy().mean())
            
            # LPIPS
            _mask = mask_out * mask_gt
            _img_out = torch.clamp(torch.from_numpy((img_out*2-1)*_mask).float().cuda()[None,:,:,:], min=-1, max=1) # [0,1] -> [-1,1]
            _img_gt = torch.clamp(torch.from_numpy((img_gt*2-1)*_mask).float().cuda()[None,:,:,:], min=-1, max=1) # [0,1] -> [-1,1]
            eval_result['image_lpips'].append(lpips(_img_out, _img_gt).detach().cpu().numpy().mean())

            # Mask IoU
            eval_result['mask_iou'].append((mask_out * mask_gt).sum() / (mask_out + mask_gt - mask_out * mask_gt).sum()*100)
            
            # Depthmap
            depthmap_out = depthmap_render[i]
            depthmap_gt = data['depthmap'][i].cpu().numpy()
            if np.sum((depthmap_out>0)*(depthmap_gt>0)) > 0:
                _depthmap_out = depthmap_out[(depthmap_out>0)*(depthmap_gt>0)]
                _depthmap_gt = depthmap_gt[(depthmap_out>0)*(depthmap_gt>0)]
                _depthmap_out = _depthmap_out - np.mean(_depthmap_out) + np.mean(_depthmap_gt)
                eval_result['depthmap_l1'].append(np.abs(_depthmap_out - _depthmap_gt).mean()*1000) # meter to millimeter
            
    save_path = osp.join('.', 'eval_results')
    os.makedirs(save_path, exist_ok=True)
    with open(osp.join(save_path, cfg.subject_id + '.json'), 'w') as f:
        json.dump({k: float(np.mean(v)) for k,v in eval_result.items()}, f)
    print(cfg.subject_id, {k: np.mean(v) for k,v in eval_result.items()})
  
if __name__ == "__main__":
    main()
