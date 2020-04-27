#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 4/27/20 4:20 PM
@File    : search01_binary.py
@Desc    : 二分查找

"""


def binary_search(alist, item):
    """
    二分查找(递归)
    最优时间复杂度:O(1)
    最坏时间复杂度:O(logn)
    :param alist:
    :return:
    """
    n = len(alist)
    if n > 0:
        mid = n // 2
        if alist[mid] == item:
            return True
        elif alist[mid] > item:
            return binary_search(alist[:mid], item)
        else:
            return binary_search(alist[mid+1:], item)
    return False


def binary_search_v1(alist, item):
    """
    二分查找(非递归)
    最优时间复杂度:O(1)
    最坏时间复杂度:O(logn)
    空间复杂度:O()
    :param alist:
    :param item:
    :return:
    """
    n = len(alist)
    start, end = 0, n-1
    while start <= end:
        mid = (start+end) // 2
        if alist[mid] == item:
            return True
        elif alist[mid] > item:
            end = mid - 1
        else:
            start = mid + 1
    return False