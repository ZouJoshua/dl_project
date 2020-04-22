#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 3/22/20 9:53 PM
@File    : basic_data.py
@Desc    : 

"""


class DataBase(object):
    def __init__(self, config):
        self.config = config
        self.sepcial_tokens()

    def sepcial_tokens(self):
        """
        定义特殊token
        :return:
        """
        self._num_token = "<NUM>"
        self._en_token = "<ENG>"
        self._unk_token = "<UNK>"
        self._pad_token = "<PAD>"
        self._sep_token = "<SEP>"
        self._cls_token = "<CLS>"
        self._mask_token = "<MASK>"

    def read_data(self, file, mode=None):
        """
        读取数据
        :param file:
        :param mode: train,eval,test
        :return:
        """
        raise NotImplementedError

    def remove_stop_word(self, inputs):
        """
        去除低频词和停用词
        :return:
        """
        raise NotImplementedError

    def get_word_embedding(self, words):
        """
        加载词向量
        :return:
        """
        raise NotImplementedError

    def gen_vocab(self):
        """
        生成词汇表
        :return:
        """
        raise NotImplementedError

    def load_vocab(self):
        """
        生成词汇表
        :return:
        """
        raise NotImplementedError

    @staticmethod
    def trans_to_index(inputs, word_to_index):
        """
        将数据转换成索引表示
        """
        raise NotImplementedError

    def padding(self, inputs, sequence_length):
        """
        对序列进行补全
        :return:
        """
        raise NotImplementedError

    def gen_data(self):
        """
        生成可导入到模型中的数据
        :return:
        """
        raise NotImplementedError

    def next_batch(self, x, y, batch_size):
        """
        生成batch数据
        :return:
        """
        raise NotImplementedError
