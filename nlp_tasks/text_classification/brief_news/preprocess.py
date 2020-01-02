#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 1/2/20 10:10 PM
@File    : preprocess.py
@Desc    : 预处理

"""


import os
import random
import jieba
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score
import numpy as np


def read_news_from_csv(data_path):
    df = pd.read_csv(data_path, encoding='utf-8')
    df = df.dropna()
    lines = df.content.values.tolist()[:20000]
    return lines

def read_stopwords(stopwords_path):
    stopwords = pd.read_csv(stopwords_path, index_col=False, quoting=3, sep="\t", names=['stopword'], encoding='utf-8')
    stopwords = stopwords['stopword'].values
    return stopwords

def preprocess_text(content_lines, sentences, category, stopwords):
    for line in content_lines:
        try:
            segs=jieba.lcut(line)
            segs = filter(lambda x: len(x) > 1, segs)
            segs = filter(lambda x: x not in stopwords, segs)
            sentences.append((" ".join(list(segs)), category))
        except Exception as e:
            print(line)
            continue


class DataSet(object):
    """
    数据准备
    """
    def __init__(self, data_dir, category_list):
        if category_list:
            self.category_list = category_list
        else:
            self.category_list = ["car", "entertainment", "finance", "sports", "military", "technology","society","home","house","international"]
        self.data_path = os.path.join(data_dir, "brief_news")

        self.file_list = [os.path.join(self.data_path, "{}_news.csv".format(cat)) for cat in category_list]
        self.stopwords = read_stopwords(os.path.join(self.data_path, "stopwords.txt"))
        self.data = self.corpus()
        self.x_train, self.x_test, self.y_train, self.y_test = self.split_data(self.data)

    def corpus(self):
        data = list()
        for file in self.file_list:
            cat = os.path.split(file)[1].split("_")[0]
            sentences = read_news_from_csv(file)
            preprocess_text(sentences, data, cat, self.stopwords)
        random.shuffle(data)
        return data

    def split_data(self, data):
        x, y = zip(*data)
        return train_test_split(x, y, random_state=1234)


def stratifiedkfold_cv(x, y, clf_class, shuffle=True, n_folds=5, **kwargs):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=shuffle)
    y_pred = y[:]
    for train_index, test_index in skf.split(x, y):
        X_train, X_test = x[train_index], x[test_index]
        y_train = y[train_index]
        clf = clf_class(**kwargs)
        clf.fit(X_train, y_train)
        y_pred[test_index] = clf.predict(X_test)
    return y_pred