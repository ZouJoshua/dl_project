#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 1/4/20 3:56 PM
@File    : text_category_baseline.py
@Desc    : fasttext文本分类

"""

import os
from setting import DATA_PATH, CONFIG_PATH
from nlp_tasks.text_classification.brief_news.preprocess import DataSet
from model_normal.fasttext_model import FastTextClassifier
from evaluate.eval_calculate import EvaluateModel


config_file = os.path.join(CONFIG_PATH, "fasttext_train.conf")
config_section = "fasttext.args_zh"
corpus_path = os.path.join(DATA_PATH, "brief_news")



cat_list = ["car", "entertainment", "finance", "sports", "military", "technology"]
ds = DataSet(DATA_PATH, cat_list)
# x, y = zip(*ds.data)



class BriefNewsCategoryModel(object):

    def __init__(self, model_path, config_file, config_section, corpus_path):
        self.mp = model_path
        self.train_file = os.path.join(corpus_path, "train.txt")
        self.test_file = os.path.join(corpus_path, "test.txt")
        if not (os.path.exists(self.train_file) and os.path.exists(self.test_file)):
            self.write_file()
        self.cf = config_file
        self.cs = config_section
        self.cp = corpus_path
        self.model = self.train()

    def train(self):
        model = FastTextClassifier(self.mp, self.cf, self.cs, train=True, data_path=self.cp).model
        return model

    def write_file(self):

        with open(self.train_file, "w") as f:
            for train in zip(ds.x_train, ds.y_train):
                line = "__label__".join(train)
                f.write(line + "\n")
        with open(self.test_file, "w") as f:
            for train in zip(ds.x_test, ds.y_test):
                line = "__label__".join(train)
                f.write(line + "\n")

def main():
    model = BriefNewsCategoryModel(corpus_path, config_file, config_section, corpus_path)


if __name__ == "__main__":
    main()

