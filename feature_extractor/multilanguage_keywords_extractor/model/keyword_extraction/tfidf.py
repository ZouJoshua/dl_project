#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 12/5/19 1:24 PM
@File    : tfidf.py
@Desc    : 

"""


import os
import gensim

from feature_extractor.multilanguage_keywords_extractor.model.keyword_extraction.ke_base import KeywordExtraction

class KeywordExtractionTFIDF(KeywordExtraction):
    def __init__(self):
        self.__dictionary = None
        self.__model = None
        self.__dictFileName = 'dict'
        self.__modelFileName = 'model'

    def train(self, docs):
        self.__dictionary = gensim.corpora.Dictionary(docs)
        corpus = [self.__dictionary.doc2bow(cor) for cor in docs]
        self.__model = gensim.models.TfidfModel(corpus)

    # doc: List(Token)
    def extract(self, doc, topK=-1):
        assert self.__model is not None, 'Please training model first.'
        words = self.__model[self.__dictionary.doc2bow(doc)]
        keywords = [(self.__dictionary.__getitem__(kv[0]), kv[1]) for kv in sorted(words, key=lambda d: d[1], reverse=True)]
        return keywords[:topK if topK != -1 else len(keywords)]

    def saveModel(self, dirPath):
        assert self.__model is not None, 'Please training model first.'
        if os.path.exists(dirPath):
            assert not os.path.isfile(dirPath), 'Required directory, but {} is a file.'.format(dirPath)
        else:
            os.makedirs(dirPath)
        self.__dictionary.save_as_text(dirPath + '/' + self.__dictFileName)
        self.__model.save(dirPath + '/' + self.__modelFileName)

    def loadModel(self, dirPath):
        self.__dictionary.load_from_text(dirPath + '/' + self.__dictFileName)
        self.__model.load(dirPath + '/' + self.__modelFileName)


