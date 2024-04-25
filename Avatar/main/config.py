# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import os
import os.path as osp
import sys

class Config:
    
    ## input, output
    render_target_shape = (512, 512)
    input_shape = (256, 256)
    joint_proj_shape = (8, 8)
    tex_shape = (1024, 1024)
    hand_size = 0.15

    ## training config
    stage_epochs = [21, 41]
    lr_dec_factor = 10

    ## testing config
    test_batch_size = 2

    ## others
    dataset = 'HARP' # 'HARP', 'Ours', 'Custom'
    num_thread = 16
    num_gpus = 1
    continue_train = False
    
    ## directory
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, '..')
    data_dir = osp.join(root_dir, 'data')
    output_dir = osp.join(root_dir, 'output')
    model_dir = osp.join(output_dir, 'model_dump')
    vis_dir = osp.join(output_dir, 'vis')
    log_dir = osp.join(output_dir, 'log')
    result_dir = osp.join(output_dir, 'result')
    mano_root_path = osp.join(root_dir, 'common', 'utils', 'human_model_files')
    uhm_root_path = osp.join(root_dir, 'common', 'utils', 'UHM')

    def set_args(self, mode, subject_id, use_tex=False, fit_pose_to_test=False, continue_train=False):
        self.mode = mode
        self.subject_id = subject_id
        self.use_tex = use_tex
        self.fit_pose_to_test = fit_pose_to_test
        self.continue_train = continue_train

        self.model_dir = osp.join(self.model_dir, subject_id)
        make_folder(self.model_dir)
        self.result_dir = osp.join(self.result_dir, subject_id)
        make_folder(self.result_dir)
        
        if not self.use_tex:
            self.lr = 1e-4
            self.lr_dec_epoch = [80]
            self.end_epoch = 100
            self.train_batch_size = 4
        else:
            self.lr = 1e-5
            self.lr_dec_epoch = [130]
            self.end_epoch = 150
            self.train_batch_size = 2

    def set_stage(self, epoch):
        if epoch < self.stage_epochs[0]:
            self.from_code_loss_weight = 10
            self.joint_proj_code_loss_weight = 0.01
            self.forearm_code_loss_weight = 0.01
            self.depthmap_loss_weight = 0
            self.mask_loss_weight = 0
            self.shadow_loss_weight = 0
            self.shadow_reg_weight = 0
            self.shadow_tv_reg_weight = 0

            self.img_loss_weight = 0
            self.tex_tv_reg_weight = 0

        elif epoch < self.stage_epochs[1]:
            self.from_code_loss_weight = 1
            self.joint_proj_code_loss_weight = 0.1
            self.forearm_code_loss_weight = 0.1
            self.depthmap_loss_weight = 1
            self.mask_loss_weight = 1
            self.shadow_loss_weight = 1
            self.shadow_reg_weight = 0.1
            self.shadow_tv_reg_weight = 100
            
            self.img_loss_weight = 0
            self.tex_tv_reg_weight = 0

        else:
            self.from_code_loss_weight = 1
            self.joint_proj_code_loss_weight = 1
            self.forearm_code_loss_weight = 1
            self.depthmap_loss_weight = 10
            self.mask_loss_weight = 10
            self.shadow_loss_weight = 1
            self.shadow_reg_weight = 0.1
            self.shadow_tv_reg_weight = 100
            
            if self.use_tex:
                self.img_loss_weight = 10
                self.tex_tv_reg_weight = 100
            else:
                self.img_loss_weight = 0
                self.tex_tv_reg_weight = 0

cfg = Config()

sys.path.insert(0, osp.join(cfg.root_dir, 'common'))
from utils.dir import add_pypath, make_folder
add_pypath(osp.join(cfg.data_dir))
add_pypath(osp.join(cfg.data_dir, cfg.dataset))
make_folder(cfg.model_dir)
make_folder(cfg.vis_dir)
make_folder(cfg.log_dir)
make_folder(cfg.result_dir)
