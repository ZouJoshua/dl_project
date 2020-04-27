#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 4/27/20 11:45 AM
@File    : sort01_bubble.py
@Desc    : 冒泡排序

"""


def bubble_sort(alist):
    """
    冒泡排序
    最坏时间复杂度O(n^2)
    :param alist:
    :return:
    """
    n = len(alist)
    for i in range(0, n-1):
        for j in range(0, n-1-i):
            if alist[j] > alist[j+1]:
                alist[j], alist[j+1] = alist[j+1] + alist[j]

def bubble_sort_v1(alist):
    """
    冒泡排序(优化,如果序列是有序的)
    平均时间复杂度O(n^2)
    最优时间复杂度(序列是有序的O(n))
    最坏时间复杂度O(n^2)
    空间复杂度O(1)
    不稳定
    :param alist:
    :return:
    """
    n = len(alist)
    for i in range(0, n-1):
        count = 0
        for j in range(0, n-1-i):
            if alist[j] > alist[j+1]:
                alist[j], alist[j+1] = alist[j+1] + alist[i]
        if count == 0:
            return