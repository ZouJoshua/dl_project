#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 1/2/20 5:25 PM
@File    : text_category_ml.py
@Desc    : 文本分类(机器学习)

"""

import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

from setting import DATA_PATH
from nlp_tasks.text_classification.brief_news.preprocess import DataSet, dump_data



class TextClassifier(object):

    def __init__(self, classifier=None, use_bow=True, ngram=True):
        self.classifier = classifier
        if use_bow:
            self.vectorizer = self.bow_vector(ngram=ngram)
        else:
            self.vectorizer = self.tfidf_vector()

    def bow_vector(self, ngram=None):
        """
        文本抽取词袋模型特征
        :return:
        """
        if not ngram:
            vec = CountVectorizer(
                analyzer='word',  # tokenise by character ngrams
                max_features=4000,  # keep the most common 1000 ngrams
            )
        else:
            vec = CountVectorizer(
                analyzer='word',  # tokenise by character ngrams
                ngram_range=(1, 3),  # use ngrams of size 1 and 2
                max_features=20000,  # keep the most common 1000 ngrams
            )

        return vec

    def tfidf_vector(self):
        """
        文本抽取tfidf模型特征
        :return:
        """
        return TfidfVectorizer(analyzer='word', ngram_range=(1, 3), max_features=12000)

    def features(self, X):
        return self.vectorizer.transform(X)

    def fit(self, X, y):
        self.vectorizer.fit(X)
        self.classifier.fit(self.features(X), y)

    def predict(self, x):
        return self.classifier.predict(self.features([x]))

    def score(self, X, y):
        return self.classifier.score(self.features(X), y)


def CategoryNB():
    cat_list = ["car", "entertainment", "finance", "sports", "military", "technology"]
    data_path = os.path.join(DATA_PATH, "brief_news")
    train_dump_file = os.path.join(data_path, "train.pkl")
    test_dump_file = os.path.join(data_path, "test.pkl")
    if os.path.exists(train_dump_file) and os.path.exists(test_dump_file):
        with open(train_dump_file, 'rb') as file_train, open(test_dump_file, "rb") as file_test:
            train_data = pickle.load(file_train)
            test_data = pickle.load(file_test)
    else:
        ds = DataSet(DATA_PATH, cat_list)
        # x, y = zip(*ds.data)
        train_data = (ds.x_train, ds.y_train)
        test_data = (ds.x_test, ds.y_test)
        dump_data(train_data, train_dump_file)
        dump_data(test_data, test_dump_file)

    x_train, y_train = train_data
    x_test, y_test = test_data
    # text_classifier_v1 = TextClassifier(classifier=MultinomialNB())
    # text_classifier_v1.fit(x_train, y_train)
    # print("一元词语朴素贝叶斯分类准确率:")
    # print(text_classifier_v1.score(x_test, y_test))
    text_classifier_v2 = TextClassifier(classifier=MultinomialNB(), ngram=True)
    text_classifier_v2.fit(x_train, y_train)
    print("多元词语朴素贝叶斯分类准确率:")
    print(text_classifier_v2.score(x_test, y_test))

    # y_pred = stratifiedkfold_cv(vec.transform(x), np.array(y), classifier)
    # precision_score(y, y_pred, average='macro')

def CategorySVM():
    cat_list = ["car", "entertainment", "finance", "sports", "military", "technology"]

    data_path = os.path.join(DATA_PATH, "brief_news")
    train_dump_file = os.path.join(data_path, "train.pkl")
    test_dump_file = os.path.join(data_path, "test.pkl")
    if os.path.exists(train_dump_file) and os.path.exists(test_dump_file):
        with open(train_dump_file, 'rb') as file_train, open(test_dump_file, "rb") as file_test:
            train_data = pickle.load(file_train)
            test_data = pickle.load(file_test)
    else:
        ds = DataSet(DATA_PATH, cat_list)
        # x, y = zip(*ds.data)
        train_data = (ds.x_train, ds.y_train)
        test_data = (ds.x_test, ds.y_test)
        dump_data(train_data, train_dump_file)
        dump_data(test_data, test_dump_file)

    x_train, y_train = train_data
    x_test, y_test = test_data
    # text_classifier_v1 = TextClassifier(classifier=SVC(kernel='linear'))
    # text_classifier_v1.fit(x_train, y_train)
    # print("一元词语支持向量机分类准确率:")
    # print(text_classifier_v1.score(x_test, y_test))
    text_classifier_v2 = TextClassifier(classifier=SVC(kernel='linear'), ngram=True)
    text_classifier_v2.fit(x_train, y_train)
    print("多元词语支持向量机分类准确率:")
    print(text_classifier_v2.score(x_test, y_test))

    # y_pred = stratifiedkfold_cv(vec.transform(x), np.array(y), classifier)
    # precision_score(y, y_pred, average='macro')



if __name__ == "__main__":
    # CategoryNB()
    CategorySVM()
