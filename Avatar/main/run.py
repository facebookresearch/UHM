# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import os
import os.path as osp
import torch

subject_id_list = ['subject_1'] # HARP dataset
#subject_id_list = ['AXE977', 'QVC422', 'QZX685', 'XKT970'] # Goliath dataset

for subject_id in subject_id_list:
    
    # training start
    cmd = 'python train.py --subject_id ' + subject_id
    print(cmd)
    os.system(cmd)
    
    cmd = 'python test.py --subject_id ' + subject_id + ' --test_epoch 99'
    print(cmd)
    os.system(cmd)

    cmd = 'python train.py --subject_id ' + subject_id + ' --use_tex --continue'
    print(cmd)
    os.system(cmd)
    
    cmd = 'python test.py --subject_id ' + subject_id + ' --use_tex --test_epoch 149'
    print(cmd)
    os.system(cmd)

    cmd = 'python get_id.py --subject_id ' + subject_id + ' --test_epoch 149'
    print(cmd)
    os.system(cmd)
    # training end
    
    # fit to the test set start
    model_path = osp.join('../output/model_dump', subject_id, 'snapshot_149.pth')
    model = torch.load(model_path)
    model['epoch'] = 0
    
    model_path = osp.join('../output/model_dump', subject_id + '_test')
    os.makedirs(model_path, exist_ok=True)
    torch.save(model, osp.join(model_path, 'snapshot_0.pth'))

    cmd = 'python train.py --subject_id ' + subject_id + '_test --fit_pose_to_test --continue'
    print(cmd)
    os.system(cmd)

    cmd = 'python test.py --subject_id ' + subject_id + '_test --fit_pose_to_test --test_epoch 99'
    print(cmd)
    os.system(cmd)

    cmd = 'python train.py --subject_id ' + subject_id + '_test --use_tex --fit_pose_to_test --continue'
    print(cmd)
    os.system(cmd)

    cmd = 'python test.py --subject_id ' + subject_id + '_test --use_tex --fit_pose_to_test --test_epoch 149'
    print(cmd)
    os.system(cmd)
    # fit to the test set end

    #cmd = 'python eval.py --subject_id ' + subject_id + ' --test_epoch 149' # eval on the training set
    cmd = 'python eval.py --subject_id ' + subject_id + '_test --fit_pose_to_test --test_epoch 149' # eval on the testing set
    print(cmd)
    os.system(cmd)
    
    # only for the Goliath dataset
    #cmd = 'python eval_3d.py --subject_id ' + subject_id
    #print(cmd)
    #os.system(cmd)

