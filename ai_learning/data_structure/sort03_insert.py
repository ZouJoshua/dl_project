#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 4/27/20 12:40 PM
@File    : sort03_insert.py
@Desc    : 插入算法

"""



def insert_sort(alist):
    """
    插入算法
    平均时间复杂度O(n^2)
    最优时间复杂度(序列是有序的O(n))
    最坏时间复杂度O(n^2)
    空间复杂度O(1)
    稳定
    :param alist:
    :return:
    """

    n = len(alist)
    # 从右边的无序序列中取出多少个元素执行这个过程
    for i in range(1, n):
        j = i
        # 从右边无序序列中取出第一个元素,即j的位置.然后将其插入到前面的正确序列中
        while j > 0:
            if alist[j] < alist[j-1]:
                alist[j], alist[j-1] = alist[j-1], alist[j]
                j -= 1
            else:   # 已经优化
                break
