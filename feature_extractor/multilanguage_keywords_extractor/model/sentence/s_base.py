#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 12/5/19 1:34 PM
@File    : s_base.py
@Desc    : 

"""

from abc import ABCMeta, abstractmethod

class Sentence(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def token(self, text):
        pass

    @abstractmethod
    def supportLanguages(self):
        pass