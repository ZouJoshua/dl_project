#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 4/27/20 12:16 PM
@File    : sort02_select.py
@Desc    : 选择排序

"""


def select_sort(alist):
    """
    选择排序
    平均时间复杂度O(n^2)
    最优时间复杂度O(n^2)
    最坏时间复杂度O(n^2)
    不稳定
    :param alist:
    :return:
    """

    n = len(alist)
    for i in range(n-1):
        min_index = i
        for j in range(i+1, n):
            if alist[min_index] > alist[j]:
                min_index = j
        alist[i], alist[min_index] = alist[min_index], alist[i]