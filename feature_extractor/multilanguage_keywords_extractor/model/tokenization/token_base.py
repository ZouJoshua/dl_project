#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 12/5/19 1:41 PM
@File    : token_base.py
@Desc    : 

"""


from abc import ABCMeta, abstractmethod

class Tokenization(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def token(self, text):
        pass

    @abstractmethod
    def supportLanguages(self):
        pass