#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 12/5/19 1:43 PM
@File    : tpos_base.py
@Desc    : 

"""

from abc import ABCMeta, abstractmethod

class TokenizationPos(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def tokenPos(self, text, language):
        pass

    @abstractmethod
    def supportLanguages(self):
        pass