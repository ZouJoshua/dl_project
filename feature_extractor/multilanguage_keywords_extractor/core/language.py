#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 12/5/19 1:11 PM
@File    : language.py
@Desc    : 

"""


class Language(object):
    def __init__(self, code, name):
        self.code = code
        self.name = name

    def __eq__(self, other):
        return self.code == other.code

    def __hash__(self):
        return self.code.__hash__()

    def __str__(self):
        return self.code + '|' + self.name

unknownLanguage = Language('un', 'unknown')