#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 12/5/19 1:39 PM
@File    : stem_base.py
@Desc    : 

"""


from abc import ABCMeta, abstractmethod

class Stem(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def stem(self, words, language):
        pass

    @abstractmethod
    def supportLanguages(self):
        pass