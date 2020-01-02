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
from model_normal.fasttext_model import FastTextClassifier
from evaluate.eval_calculate import EvaluateModel


config_file = os.path.join(CONFIG_PATH, "fasttext_train.conf")
config_section = "fasttext.args_zh"
corpus_path = os.path.join(DATA_PATH, "sogou")



class SogouCategoryModel(object):

    def __init__(self, model_path, config_file, config_section, corpus_path):
        self.mp = model_path
        self.cf = config_file
        self.cs = config_section
        self.cp = corpus_path
        self.model = self.train()

    def train(self):
        model = FastTextClassifier(self.mp, self.cf, self.cs, train=True, file_path=self.cp).model
        return model


def main():
    model = SogouCategoryModel(corpus_path, config_file, config_section, corpus_path)


if __name__ == "__main__":
    main()


