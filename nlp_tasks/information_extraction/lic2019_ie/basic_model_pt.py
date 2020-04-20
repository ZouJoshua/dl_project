#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 4/19/20 7:50 PM
@File    : basic_model_pt.py
@Desc    : 

"""

import torch
import time


class BasicModule(torch.nn.Module):
    '''
    封装了nn.Module,主要是提供了save和load两个方法
    '''

    def __init__(self):
        super(BasicModule, self).__init__()

    def load(self, path):
        '''
        可加载指定路径的模型
        '''
        self.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
        # self.load_state_dict(torch.load(path))

    def save(self, opt, epoch=0):
        '''
        保存模型，默认使用“模型名字+时间”作为文件名
        '''
        name_str = './checkpoints/{}_sl:{}_k:{}_fn:{}_lam:{}_lr:{}_epoch:{}'
        name = name_str.format(opt.model, opt.seq_length, opt.filters, opt.filter_num, opt.lam, opt.lr, epoch)
        torch.save(self.state_dict(), name)
        return name