#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 2020/6/15 5:05 下午
@File    : test_fasttext_model.py
@Desc    : 用公开数据集测试文本分类

"""

import os
from setting import DATA_PATH, CONFIG_PATH
from model_normal.fasttext_model import FastTextClassifier, Config
from utils.logger import Logger
from setting import LOG_PATH





class TestCategoryModel(object):

    def __init__(self, model_path, config_file, config_section, corpus_path, log, name_mark):
        self.mp = model_path
        self.cp = corpus_path
        self.loger = log
        self.nm = name_mark
        self.config = Config(config_file, section=config_section)
        self.model = self.train()

    def train(self):
        model = FastTextClassifier(self.mp, self.config, train=True, data_path=self.cp, name_mark=self.nm,
                                   logger=self.loger).model
        return model


def main():
    model_path = os.path.join(DATA_PATH, "model", "test_sample")
    corpus_path = os.path.join(DATA_PATH, "common", "test_sample")
    log_file = os.path.join(model_path, 'train_log')
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    log = Logger("test_fasttext_train_log", log2console=False, log2file=True, logfile=log_file).get_logger()

    config_file = os.path.join(CONFIG_PATH, "fasttext.conf")
    config_section = "fasttext.args_thuc_news_token_test_sample"
    TestCategoryModel(model_path, config_file, config_section, corpus_path, log, name_mark="")


if __name__ == "__main__":
    main()