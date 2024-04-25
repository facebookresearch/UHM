# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import argparse
import os
import os.path as osp
import torch
from tqdm import tqdm
from glob import glob
import numpy as np
from config import cfg
from base import Tester
from utils.vis import render_mesh
from utils.uhm import uhm
from pytorch3d.io import save_obj
import json
import cv2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_id', type=str, dest='subject_id')
    parser.add_argument('--use_tex', dest='use_tex', action='store_true')
    parser.add_argument('--fit_pose_to_test', dest='fit_pose_to_test', action='store_true')
    parser.add_argument('--test_epoch', type=str, dest='test_epoch')
    args = parser.parse_args()

    assert args.subject_id, "Please set subject id."
    assert args.test_epoch, 'Test epoch is required.'
    return args

def main():

    # argument parse and create log
    args = parse_args()
    cfg.set_args('test', args.subject_id, args.use_tex, args.fit_pose_to_test)

    tester = Tester(args.test_epoch)
    tester._make_batch_generator()
    tester._make_model()
 
    unwrapped_tex_save = 0
    unwrapped_tex_wo_shadow_save = 0
    unwrapped_tex_mask_save = 0
    unwrapped_tex_grad_zero_save = 0
    for itr, data in enumerate(tqdm(tester.batch_generator)):

        # forward
        with torch.no_grad():
            out = tester.model(data, 'test')
        
        # save
        img_orig = data['img_orig'].cpu().numpy()
        cam_param = data['cam_param']
        mesh = out['mesh'].cpu()
        mesh_from_code = out['mesh_from_code'].cpu()
        mask_render = out['mask_render'].cpu().numpy()
        img_calib = out['img_calib'].cpu().numpy()
        shadow_uv = out['shadow_uv'].cpu().numpy()
        shadow_render = out['shadow_render'].cpu().numpy()
        joint_proj = out['joint_proj'].cpu().numpy()
        root_pose = out['root_pose'].cpu().numpy()
        hand_pose = out['hand_pose'].cpu().numpy()
        trans = out['trans'].cpu().numpy()
        unwrapped_tex = out['unwrapped_tex'].cpu().numpy()
        unwrapped_tex_wo_shadow = out['unwrapped_tex_wo_shadow'].cpu().numpy()
        unwrapped_tex_valid = out['unwrapped_tex_valid'].cpu().numpy()
        if cfg.use_tex:
            img_render = out['img_render'].cpu().numpy()
            img_render_shadow = out['img_render_shadow'].cpu().numpy()
            img_render_refine = out['img_render_refine'].cpu().numpy()
            img_render_refine_shadow = out['img_render_refine_shadow'].cpu().numpy()
            unwrapped_tex_refine = out['unwrapped_tex_refine'].cpu().numpy()[0].transpose(1,2,0)[:,:,::-1]*255
            cv2.imwrite(osp.join(cfg.result_dir, 'unwrapped_texture_refine.png'), unwrapped_tex_refine)
        batch_size = mesh.shape[0]
        for i in range(batch_size):
            seq_name = data['seq_name'][i]
            frame_idx = int(data['frame_idx'][i])
            save_root_path = osp.join(cfg.result_dir, seq_name)
            os.makedirs(save_root_path, exist_ok=True)
            
            # unwrapped texture
            unwrapped_tex_save += unwrapped_tex[i]
            unwrapped_tex_wo_shadow_save += unwrapped_tex_wo_shadow[i]
            unwrapped_tex_mask_save += unwrapped_tex_valid[i]
            grad_x = np.abs(cv2.Sobel(unwrapped_tex_valid[i][0], cv2.CV_64F, 1, 0, ksize=5)) # x-axis gradient
            grad_y = np.abs(cv2.Sobel(unwrapped_tex_valid[i][0], cv2.CV_64F, 0, 1, ksize=5)) # y-axis gradient
            unwrapped_tex_grad_zero_save += (grad_x == 0) * (grad_y == 0) * unwrapped_tex_valid[i][0]

            # save mesh
            save_path = osp.join(save_root_path, 'Meshes')
            os.makedirs(save_path, exist_ok=True)
            save_obj(osp.join(save_path, str(frame_idx) + '.obj'), torch.FloatTensor(mesh[i]), torch.LongTensor(uhm.face))
            
            # save visualized results
            save_path = osp.join(save_root_path, 'Renders')
            os.makedirs(save_path, exist_ok=True)
            cv2.imwrite(osp.join(save_path, str(frame_idx) + '_gt.png'), img_orig[i].transpose(1,2,0)[:,:,::-1]*255)
            render = render_mesh(mesh[i].numpy(), uhm.face, {'focal': cam_param['focal'][i].numpy(), 'princpt': cam_param['princpt'][i].numpy()}, img_orig[i].transpose(1,2,0)[:,:,::-1]*255, 0.75)
            cv2.imwrite(osp.join(save_path, str(frame_idx) + '_mesh.png'), render)
            render = render_mesh(mesh_from_code[i].numpy(), uhm.face, {'focal': cam_param['focal'][i].numpy(), 'princpt': cam_param['princpt'][i].numpy()}, img_orig[i].transpose(1,2,0)[:,:,::-1]*255, 0.75)
            cv2.imwrite(osp.join(save_path, str(frame_idx) + '_mesh_from_code.png'), render)
            cv2.imwrite(osp.join(save_path, str(frame_idx) + '_mask_render.png'), mask_render[i].transpose(1,2,0)[:,:,::-1]*255)
            cv2.imwrite(osp.join(save_path, str(frame_idx) + '_render_calib.png'), img_calib[i].transpose(1,2,0)[:,:,::-1]*255)
            cv2.imwrite(osp.join(save_path, str(frame_idx) + '_shadow_uv.png'), shadow_uv[i].transpose(1,2,0)*255)
            cv2.imwrite(osp.join(save_path, str(frame_idx) + '_shadow_render.png'), shadow_render[i].transpose(1,2,0)*255)
            cv2.imwrite(osp.join(save_path, str(frame_idx) + '_unwrapped_texture.png'), unwrapped_tex[i].transpose(1,2,0)[:,:,::-1])
            cv2.imwrite(osp.join(save_path, str(frame_idx) + '_unwrapped_texture_wo_shadow.png'), unwrapped_tex_wo_shadow[i].transpose(1,2,0)[:,:,::-1])
            vis = img_orig[i].transpose(1,2,0)[:,:,::-1].copy()*255
            for j in range(uhm.joint_num):
                vis = cv2.circle(vis, (int(joint_proj[i][j][0]), int(joint_proj[i][j][1])), 3, (255,0,0), -1)
            cv2.imwrite(osp.join(save_path, str(frame_idx) + '_joint.png'), vis)
            if cfg.use_tex:
                cv2.imwrite(osp.join(save_path, str(frame_idx) + '_render.png'), img_render[i].transpose(1,2,0)[:,:,::-1]*255)
                cv2.imwrite(osp.join(save_path, str(frame_idx) + '_render_shadow.png'), img_render_shadow[i].transpose(1,2,0)[:,:,::-1]*255)
                cv2.imwrite(osp.join(save_path, str(frame_idx) + '_render_refine.png'), img_render_refine[i].transpose(1,2,0)[:,:,::-1]*255)
                cv2.imwrite(osp.join(save_path, str(frame_idx) + '_render_refine_shadow.png'), img_render_refine_shadow[i].transpose(1,2,0)[:,:,::-1]*255)
            
            # save pose parameters
            save_path = osp.join(save_root_path, 'PoseParams')
            os.makedirs(save_path, exist_ok=True)
            save_path = osp.join(save_path, str(frame_idx) + '.json')
            with open(save_path, 'w') as f:
                json.dump({'root_pose': root_pose[i].tolist(), 'hand_pose': hand_pose[i].tolist(), 'trans': trans[i].tolist()}, f)

    # unwrapped mask
    unwrapped_is_valid = unwrapped_tex_mask_save
    cv2.imwrite(osp.join(cfg.result_dir, 'unwrapped_texture_mask.png'), (unwrapped_is_valid.sum(0) > 0).astype(np.uint8)*255)
    
    # unwrapped gradient
    unwrapped_grad_zero = (unwrapped_tex_grad_zero_save > 0).astype(np.uint8) * 255
    cv2.imwrite(osp.join(cfg.result_dir, 'unwrapped_texture_grad_mask.png'), unwrapped_grad_zero)

    # unwrapped texture (average)
    unwrapped_tex = unwrapped_tex_save / (unwrapped_is_valid + 1e-4)
    unwrapped_tex = np.clip(unwrapped_tex, a_min=0, a_max=255)
    unwrapped_tex = unwrapped_tex.transpose(1,2,0)[:,:,::-1].astype(np.uint8)
    cv2.imwrite(osp.join(cfg.result_dir, 'unwrapped_texture.png'), unwrapped_tex)
    do_inpaint = (unwrapped_is_valid.sum(0) == 0).astype(np.uint8)
    unwrapped_tex = cv2.inpaint(unwrapped_tex, do_inpaint, 3, cv2.INPAINT_TELEA)
    unwrapped_tex = unwrapped_tex * uhm.uv_mask.numpy()[:,:,None]
    cv2.imwrite(osp.join(cfg.result_dir, 'unwrapped_texture_inpaint.png'), unwrapped_tex)
    
    # unwrapped texutre without shadow (average)
    unwrapped_tex_wo_shadow = unwrapped_tex_wo_shadow_save / (unwrapped_is_valid + 1e-4)
    unwrapped_tex_wo_shadow = np.clip(unwrapped_tex_wo_shadow, a_min=0, a_max=255)
    unwrapped_tex_wo_shadow = unwrapped_tex_wo_shadow.transpose(1,2,0)[:,:,::-1].astype(np.uint8)
    cv2.imwrite(osp.join(cfg.result_dir, 'unwrapped_texture_wo_shadow.png'), unwrapped_tex_wo_shadow)
    do_inpaint = (unwrapped_is_valid.sum(0) == 0).astype(np.uint8)
    unwrapped_tex_wo_shadow = cv2.inpaint(unwrapped_tex_wo_shadow, do_inpaint, 3, cv2.INPAINT_TELEA)
    unwrapped_tex_wo_shadow = unwrapped_tex_wo_shadow * uhm.uv_mask.numpy()[:,:,None]
    cv2.imwrite(osp.join(cfg.result_dir, 'unwrapped_texture_wo_shadow_inpaint.png'), unwrapped_tex_wo_shadow)

    # make video
    seq_path_list = [x for x in glob(osp.join(cfg.result_dir, '*')) if osp.isdir(x)]
    for seq_path in seq_path_list:
        render_path_list = glob(osp.join(seq_path, 'Renders', '*_mesh.png'))
        frame_idx_list = sorted([int(x.split('/')[-1].split('_')[0]) for x in render_path_list])

        img_height, img_width, _ = cv2.imread(osp.join(osp.join(seq_path, 'Renders', str(frame_idx_list[0]) + '_mesh.png'))).shape
        if not cfg.use_tex:
            video_out = cv2.VideoWriter(osp.join(seq_path, 'video_init_wo_tex.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (img_width*2, img_height))
        else:
            video_out = cv2.VideoWriter(osp.join(seq_path, 'video_refine_with_tex.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (img_width*4, img_height))

        for frame_idx in tqdm(frame_idx_list):
            orig_img = cv2.imread(osp.join(seq_path, 'Renders', str(frame_idx) + '_gt.png'))
            mesh_img = cv2.imread(osp.join(seq_path, 'Renders', str(frame_idx) + '_mesh.png'))
            cv2.putText(orig_img, 'image', (int(1/3*orig_img.shape[1]), int(0.1*orig_img.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 2, [51,51,255], 3, 2)
            cv2.putText(mesh_img, 'mesh', (int(1/3*mesh_img.shape[1]), int(0.1*mesh_img.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 2, [51,51,255], 3, 2)
            if not cfg.use_tex:
                frame = np.concatenate((orig_img, mesh_img),1)
            else:
                render = cv2.imread(osp.join(seq_path, 'Renders', str(frame_idx) + '_render_refine.png'))
                render_shadow = cv2.imread(osp.join(seq_path, 'Renders', str(frame_idx) + '_render_refine_shadow.png'))
                cv2.putText(render, 'render', (int(1/3*render.shape[1]), int(0.1*render.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 2, [51,51,255], 3, 2)
                cv2.putText(render_shadow, 'render_w_shadow', (int(1/5*render_shadow.shape[1]), int(0.1*render_shadow.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 2, [51,51,255], 3, 2)
                frame = np.concatenate((orig_img, mesh_img, render, render_shadow),1)

            cv2.putText(frame, str(frame_idx), (int(0.02*frame.shape[1]), int(0.1*frame.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 2, [51,51,255], 3, 2) 
            video_out.write(frame)

        video_out.release()


if __name__ == "__main__":
    main()
