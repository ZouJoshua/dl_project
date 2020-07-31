#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 12/5/19 1:28 PM
@File    : base.py
@Desc    : 

"""

from abc import ABCMeta, abstractmethod

class LanguageDetection(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def detect(self, text):
        pass

    @abstractmethod
    def supportLanguages(self):
        pass