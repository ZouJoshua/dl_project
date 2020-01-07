#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 19-7-16 下午7:29
@File    : gensim_word2vec.py
@Desc    : 
"""


from gensim.models import word2vec
from preprocess.tools import split_text, read_txt_file


class GensimWord2VecModel:

    def __init__(self, train_file, model_path):
        """
        用gensim word2vec 训练词向量
        :param train_file: 分好词的文本
        :param model_path: 模型保存的路径
        """
        self.train_file = train_file
        self.model_path = model_path
        self.model = self.load()
        if not self.model:
            self.model = self.train()
            self.save(self.model_path)

    def train(self):
        sentences = Sentences(self.train_file)
        model = word2vec.Word2Vec(sentences, min_count=2, window=3, size=300, workers=4)
        return model

    def vector(self, word):
        return self.model.wv.get_vector(word)

    def similar(self, word):
        return self.model.wv.similar_by_word(word, topn=10)

    def save(self, model_path):
        self.model.save(model_path)

    def load(self):
        # 加载模型文件
        try:
            model = word2vec.Word2Vec.load(self.model_path)
        except FileNotFoundError:
            model = None
        return model

class Sentences(object):

    def __init__(self, filename):
        self.file = filename

    def __iter__(self):
        print(">>>>> 正在读取embed语料")
        _doc_count = 0
        for doc in read_txt_file(self.file):
            _doc_count += 1
            word_list = split_text(doc)
            yield word_list
        print("<<<<< 已读取{}文档".format(_doc_count))
