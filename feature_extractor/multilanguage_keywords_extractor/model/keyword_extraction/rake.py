#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 12/5/19 1:15 PM
@File    : rake.py
@Desc    : 

"""

from feature_extractor.multilanguage_keywords_extractor.core import base
from feature_extractor.multilanguage_keywords_extractor.model.keyword_extraction.third_party import rake

from feature_extractor.multilanguage_keywords_extractor.model.keyword_extraction.ke_base import KeywordExtraction

class KeywordExtractionRake(KeywordExtraction):
    def __init__(self, stopwordPath=base.englishStopwordPath, minLength=5, maxWordNum=3, minTimes=2):
        self.__stopwordPath = stopwordPath
        self.__minLength = minLength
        self.__maxWordNum = maxWordNum
        self.__minTimes = minTimes

    def train(self, docs):
        pass

    # text: string text
    def extract(self, text, topK=-1):
        rake_object = rake.Rake(self.__stopwordPath, self.__minLength, self.__maxWordNum, self.__minTimes)
        keywords = rake_object.run(text)
        return keywords[:topK if topK != -1 else len(keywords)]

    def saveModel(self, dirPath):
        pass

    def loadModel(self, dirPath):
        pass