#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 2/27/20 3:47 PM
@File    : thuc_news_classifier.py
@Desc    : 文本分类baseline

"""

import os
from setting import DATA_PATH, CONFIG_PATH
from model_normal.fasttext_model import FastTextClassifier
from utils.logger import Logger
from setting import LOG_PATH




class ThucCategoryModel(object):

    def __init__(self, model_path, config_file, config_section, corpus_path, log, name_mark):
        self.mp = model_path
        self.cf = config_file
        self.cs = config_section
        self.cp = corpus_path
        self.loger = log
        self.nm = name_mark
        self.model = self.train()

    def train(self):
        model = FastTextClassifier(self.mp, self.cf, self.cs, train=True, data_path=self.cp, name_mark=self.nm, logger=self.loger).model
        return model



def participle_train():
    config_file = os.path.join(CONFIG_PATH, "fasttext_train.conf")
    config_section = "fasttext.args_thuc_news"
    corpus_path = os.path.join(DATA_PATH, "corpus", "thuc_news")
    model_path = os.path.join(DATA_PATH, "model", "thuc_news")
    log_file = os.path.join(model_path, 'train_log')
    log = Logger("fasttext_train_log", log2console=False, log2file=True, logfile=log_file).get_logger()
    # jieba 分词训练版本
    model = ThucCategoryModel(model_path, config_file, config_section, corpus_path, log)


def single_char_train():
    config_file = os.path.join(CONFIG_PATH, "fasttext_train.conf")
    config_section = "fasttext.args_thuc_news_single_char"
    corpus_path = os.path.join(DATA_PATH, "corpus", "thuc_news")
    model_path = os.path.join(DATA_PATH, "model", "thuc_news")
    log_file = os.path.join(model_path, 'train_log')
    log = Logger("fasttext_train_log", log2console=False, log2file=True, logfile=log_file).get_logger()
    # 单字训练版本
    model = ThucCategoryModel(model_path, config_file, config_section, corpus_path, log, name_mark="single_char.")


def main():
    # participle_train()
    single_char_train()


if __name__ == '__main__':
    main()

