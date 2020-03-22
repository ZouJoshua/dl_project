#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 3/22/20 9:49 PM
@File    : basic_predict.py
@Desc    : 

"""




class PredictorBase(object):
    def __init__(self, config):

        self.output_path = config.get("output_dir")

    def load_vocab(self):
        """
        加载词汇表
        :return:
        """
        raise NotImplementedError

    def sentence_to_idx(self, text):
        """
        创建数据对象
        :return:
        """
        raise NotImplementedError

    def create_model(self):
        """
        根据config文件选择对应的模型，并初始化
        :return:
        """
        raise NotImplementedError

    def predict(self, text):
        """
        训练模型
        :return:
        """
        raise NotImplementedError