#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 12/5/19 1:46 PM
@File    : data_utils.py
@Desc    : 

"""


def reverseDictionary(map):
    return dict([(kv[1], kv[0]) for kv in map.items()])