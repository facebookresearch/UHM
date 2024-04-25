# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import os
import os.path as osp
import math
import time
import glob
import abc
from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms
from timer import Timer
from logger import colorlogger
from torch.nn.parallel.data_parallel import DataParallel
from config import cfg
from model import get_model

# dynamic dataset import
exec('from ' + cfg.dataset + ' import ' + cfg.dataset)

class Base(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, log_name='logs.txt'):
        
        self.cur_epoch = 0

        # timer
        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()

        # logger
        self.logger = colorlogger(cfg.log_dir, log_name=log_name)

    @abc.abstractmethod
    def _make_batch_generator(self):
        return

    @abc.abstractmethod
    def _make_model(self):
        return

class Trainer(Base):
    def __init__(self):
        super(Trainer, self).__init__(log_name = 'train_logs.txt')
 
    def get_optimizer(self, trainable_params):
        optimizer = torch.optim.Adam(trainable_params, lr=cfg.lr)
        return optimizer
   
    def set_lr(self, epoch):
        if len(cfg.lr_dec_epoch) == 0:
            return cfg.lr
        for e in cfg.lr_dec_epoch:
            if epoch < e:
                break
        if epoch < cfg.lr_dec_epoch[-1]:
            idx = cfg.lr_dec_epoch.index(e)
            for g in self.optimizer.param_groups:
                g['lr'] = cfg.lr / (cfg.lr_dec_factor ** idx)
        else:
            for g in self.optimizer.param_groups:
                g['lr'] = cfg.lr / (cfg.lr_dec_factor ** len(cfg.lr_dec_epoch))

    def get_lr(self):
        for g in self.optimizer.param_groups:
            cur_lr = g['lr']
        return cur_lr
    
    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        trainset_loader = eval(cfg.dataset)(transforms.ToTensor())
        
        self.itr_per_epoch = math.ceil(len(trainset_loader) / cfg.num_gpus / cfg.train_batch_size)
        self.batch_generator = DataLoader(dataset=trainset_loader, batch_size=cfg.num_gpus*cfg.train_batch_size, shuffle=True, num_workers=cfg.num_thread, pin_memory=True)

    def _make_model(self):
        # prepare network
        self.logger.info("Creating graph and optimizer...")
        model = get_model('train')
        model = DataParallel(model).cuda()
        optimizer = self.get_optimizer(model.module.trainable_params)

        if cfg.continue_train:
            start_epoch, model = self.load_model(model)
        else:
            start_epoch = 0
        model.train()
        for module in model.module.eval_modules:
            module.eval()

        self.start_epoch = start_epoch
        self.model = model
        self.optimizer = optimizer

    def save_model(self, state, epoch):
        file_path = osp.join(cfg.model_dir,'snapshot_{}.pth'.format(str(epoch)))
        torch.save(state, file_path)
        self.logger.info("Write snapshot into {}".format(file_path))

    def load_model(self, model):
        model_file_list = glob.glob(osp.join(cfg.model_dir,'*.pth'))
        cur_epoch = max([int(file_name[file_name.find('snapshot_') + 9 : file_name.find('.pth')]) for file_name in model_file_list])
        model_path = osp.join(cfg.model_dir, 'snapshot_' + str(cur_epoch) + '.pth')
        self.logger.info('Load checkpoint from {}'.format(model_path))
        ckpt = torch.load(model_path, map_location='cpu')
        start_epoch = ckpt['epoch'] + 1
        model.module.load_state_dict(ckpt['network'], strict=False)
        return start_epoch, model

class Tester(Base):
    def __init__(self, test_epoch):
        super(Tester, self).__init__(log_name = 'test_logs.txt')
        self.test_epoch = int(test_epoch)

    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        testset_loader = eval(cfg.dataset)(transforms.ToTensor())
        batch_generator = DataLoader(dataset=testset_loader, batch_size=cfg.num_gpus*cfg.test_batch_size, shuffle=False, num_workers=cfg.num_thread, pin_memory=True)
        
        self.testset = testset_loader
        self.batch_generator = batch_generator

    def _make_model(self):
        model_path = os.path.join(cfg.model_dir, 'snapshot_%d.pth' % self.test_epoch)
        assert os.path.exists(model_path), 'Cannot find model at ' + model_path
        self.logger.info('Load checkpoint from {}'.format(model_path))
        
        # prepare network
        self.logger.info("Creating graph...")
        model = get_model('test')
        model = DataParallel(model).cuda()
        ckpt = torch.load(model_path)
        model.module.load_state_dict(ckpt['network'], strict=False)
        model.eval()

        self.model = model
    
