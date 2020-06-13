#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 12/18/19 4:33 PM
@File    : fasttext_train.py
@Desc    : 

"""

import os
from setting import DATA_PATH, CONFIG_PATH
from model_normal.fasttext_model import FastTextClassifier, Config
from utils.logger import Logger
from setting import LOG_PATH





class VivoCategoryModel(object):

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
    model_path = os.path.join(DATA_PATH, "model", "vivo_news")
    corpus_path = os.path.join(DATA_PATH, "corpus", "vivo_news")
    log_file = os.path.join(model_path, 'train_log')
    log = Logger("fasttext_train_log", log2console=False, log2file=True, logfile=log_file).get_logger()

    config_file = os.path.join(CONFIG_PATH, "fasttext.conf")
    config_section = "fasttext.args_vivo_news_token"
    model = VivoCategoryModel(model_path, config_file, config_section, corpus_path, log, name_mark="")


if __name__ == "__main__":
    main()


