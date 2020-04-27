#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 4/27/20 1:11 PM
@File    : sort04_shell.py
@Desc    : 希尔排序

"""


def shell_sort(alist):
    """
    希尔排序(插入序列的改进版)
    平均时间复杂度:O(nlogn)~O(n^2)
    最优时间复杂度:(序列是有序的O(n^1.3))  根据gap序列的长度而定
    最坏时间复杂度:O(n^2)
    空间复杂度:O(1)
    不稳定
    :param alist:
    :return:
    """
    n = len(alist)
    gap = n // 2

    # gap变化到0之前,插入算法执行的次数
    while gap >= 1:
        # 插入算法与普通插入算法的区别就是gap的步长
        for i in range(gap, n):
            j = i
            while j > 0:
                if alist[j] < alist[j-gap]:
                    alist[j-gap], alist[j] = alist[j], alist[j-gap]
                    j -= gap
                else:
                    break

        # 缩短gap步长
        gap //= 2