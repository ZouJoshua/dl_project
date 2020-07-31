#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 12/5/19 1:13 PM
@File    : ke_base.py
@Desc    : 

"""

from abc import ABCMeta, abstractmethod

class KeywordExtraction(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, docs):
        pass

    @abstractmethod
    def extract(self, doc, topK):
        pass

    @abstractmethod
    def saveModel(self, filePath):
        pass

    @abstractmethod
    def loadModel(self, filePath):
        pass