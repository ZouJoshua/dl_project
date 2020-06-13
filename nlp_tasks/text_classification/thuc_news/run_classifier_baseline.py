#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 3/13/20 5:52 PM
@File    : run_classifier_baseline.py
@Desc    : 文本分类baseline的训练\测试

"""


import os
from setting import DATA_PATH, CONFIG_PATH
from model_normal.fasttext_model import FastTextClassifier, Config
from utils.logger import Logger




class ThucCategoryModel(object):

    def __init__(self, model_path, config_file, config_section, corpus_path, log, name_mark):
        self.mp = model_path
        self.cp = corpus_path
        self.loger = log
        self.nm = name_mark
        self.config = Config(config_file, section=config_section)
        self.model = self.train()

    def train(self):
        model = FastTextClassifier(self.mp, self.config, train=True, data_path=self.cp, name_mark=self.nm, logger=self.loger).model
        return model



def participle_train():
    config_file = os.path.join(CONFIG_PATH, "fasttext_train.conf")
    config_section = "fasttext.args_thuc_news"
    corpus_path = os.path.join(DATA_PATH, "corpus", "thuc_news")
    model_path = os.path.join(DATA_PATH, "model", "thuc_news")
    log_file = os.path.join(model_path, 'train_log')
    log = Logger("fasttext_train_log", log2console=False, log2file=True, logfile=log_file).get_logger()
    # jieba 分词训练版本
    ThucCategoryModel(model_path, config_file, config_section, corpus_path, log)


def single_char_train():
    config_file = os.path.join(CONFIG_PATH, "fasttext.conf")
    config_section = "fasttext.args_thuc_news_single_char"
    corpus_path = os.path.join(DATA_PATH, "corpus", "thuc_news")
    model_path = os.path.join(DATA_PATH, "model", "thuc_news")
    log_file = os.path.join(model_path, 'train_log')
    log = Logger("fasttext_train_log", log2console=False, log2file=True, logfile=log_file).get_logger()
    # 单字训练版本
    ThucCategoryModel(model_path, config_file, config_section, corpus_path, log, name_mark="single_char.")


def main():
    # participle_train()
    single_char_train()


if __name__ == '__main__':
    main()