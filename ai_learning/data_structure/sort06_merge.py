#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 4/27/20 2:37 PM
@File    : sort06_merge.py
@Desc    : 归并排序

"""


def merge_sort(alist):
    """
    归并排序
    平均时间复杂度:O(nlogn)
    最优时间复杂度:(nlogn)
    最坏时间复杂度:O(nlogn)
    空间复杂度:O(n)
    稳定
    :param alist:
    :return:
    """
    n = len(alist)
    if n < 2:
        return alist
    mid = n // 2
    # 采用归并排序后形成的新的有序的列表
    left_li = merge_sort(alist[:mid])
    right_li = merge_sort(alist[mid:])
    left, right = 0, 0
    result = list()
    while left < len(left_li) and right < len(right_li):
        if left_li[left] <= right_li[right]:
            result.append(left_li[left])
            left += 1
        else:
            result.append(right_li[right])
            right += 1

    result += left_li[left:]
    result += right_li[right:]
    return result



def merge(left, right):
    result = []
    while left and right:
        if left[0] < right[0]:
            result.append(left.pop(0))
        else:
            result.append(right.pop(0))
    while left:
        result.append(left.pop(0))
    while right:
        result.append(right.pop(0))

    return result


if __name__ == "__main__":
    a = [3, 42, 5, 1, 55, 23, 44, 54, 32, 8, 10]
    s = merge_sort(a)
    # s = quick_sort_v1(a, 0, len(a)-1)
    print(s)