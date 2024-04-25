# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import argparse
import torch
import os
from config import cfg
from base import Trainer
from utils.uhm import uhm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_id', type=str, dest='subject_id')
    parser.add_argument('--use_tex', dest='use_tex', action='store_true')
    parser.add_argument('--fit_pose_to_test', dest='fit_pose_to_test', action='store_true')
    parser.add_argument('--continue', dest='continue_train', action='store_true')
    args = parser.parse_args()
    
    assert args.subject_id, "Please set subject id."
    return args

def main():
    
    # argument parse and create log
    args = parse_args()
    cfg.set_args('train', args.subject_id, args.use_tex, args.fit_pose_to_test, args.continue_train)

    trainer = Trainer()
    trainer._make_batch_generator()
    trainer._make_model()
    
    # train
    for epoch in range(trainer.start_epoch, cfg.end_epoch):
        # adjust weight of loss functions
        cfg.set_stage(epoch)

        trainer.set_lr(epoch)
        trainer.tot_timer.tic()
        trainer.read_timer.tic()
        for itr, data in enumerate(trainer.batch_generator):
            trainer.read_timer.toc()
            trainer.gpu_timer.tic()

            # forward
            trainer.optimizer.zero_grad()
            loss = trainer.model(data, 'train')
            loss = {k:loss[k].mean() for k in loss}
            sum(loss[k] for k in loss).backward()
            trainer.optimizer.step()

            # backward
            trainer.gpu_timer.toc()
            screen = [
                'Epoch %d/%d itr %d/%d:' % (epoch, cfg.end_epoch, itr, trainer.itr_per_epoch),
                'lr: %g' % (trainer.get_lr()),
                'speed: %.2f(%.2fs r%.2f)s/itr' % (
                    trainer.tot_timer.average_time, trainer.gpu_timer.average_time, trainer.read_timer.average_time),
                '%.2fh/epoch' % (trainer.tot_timer.average_time / 3600. * trainer.itr_per_epoch),
                ]
            screen += ['%s: %.4f' % ('loss_' + k, v.detach()) for k,v in loss.items()]
            trainer.logger.info(' '.join(screen))
            print(cfg.model_dir)

            trainer.tot_timer.toc()
            trainer.tot_timer.tic()
            trainer.read_timer.tic()

        # save model
        if ((epoch % 10 == 0) and epoch > 0) or (epoch == cfg.end_epoch - 1):
            trainer.save_model({
                'epoch': epoch,
                'network': trainer.model.module.state_dict(),
                'optimizer': trainer.optimizer.state_dict(),
            }, epoch)
        

if __name__ == "__main__":
    main()
