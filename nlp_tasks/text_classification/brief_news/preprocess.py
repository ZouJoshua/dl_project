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
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold


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
            segs = jieba.lcut(line)
            segs = filter(lambda x: len(x) > 1, segs)
            segs = filter(lambda x: x not in stopwords, list(segs))
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
            self.category_list = ["car", "entertainment", "finance", "sports", "military", "technology", "society","home","house","international"]
        self.data_path = os.path.join(data_dir, "brief_news")

        self.file_list = [os.path.join(self.data_path, "{}_news.csv".format(cat)) for cat in category_list]
        self.stopwords = read_stopwords(os.path.join(self.data_path, "stopwords.txt"))
        self.data = self.corpus()
        # self.x_train, self.x_test, self.y_train, self.y_test = self.split_data(self.data)
        self.x_train, self.x_test, self.y_train, self.y_test = self.split_data_kflod(self.data)

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

    def split_data_kflod(self, data):
        x, y = zip(*data)
        skf = StratifiedKFold(n_splits=5, shuffle=True)
        index = [(train_index, test_index) for train_index, test_index in skf.split(x, y)]
        X_train, X_test = [x[i] for i in index[0][0]], [x[i] for i in index[0][1]]
        Y_train, Y_test = [y[i] for i in index[0][0]], [y[i] for i in index[0][1]]
        return X_train, X_test, Y_train, Y_test



def dump_data(data, file):
    """
    序列化数据
    :param data:
    :param file:
    :return:
    """
    print("序列化数据到文件:{}".format(file))
    outfile = open(file, "wb")
    pickle.dump(data, outfile)
    outfile.close()




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