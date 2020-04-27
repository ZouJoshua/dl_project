#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 4/27/20 1:47 PM
@File    : sort05_quick.py
@Desc    : 快速排序

"""



def quick_sort(alist, l, r):
    """
    快速排序算法
    平均时间复杂度O(nlogn)
    最优时间复杂度(序列是有序的O(n))
    最坏时间复杂度O(n^2)
    稳定
    :param alist:
    :param left:
    :param right:
    :return:
    """
    if l >= r:
        return
    mid_value = alist[l]
    left = l
    right = r
    while left < right:
        while left < right and alist[right] >= mid_value:
            right -= 1
        alist[left] = alist[right]
        while left < right and alist[left] < mid_value:
            left += 1
        alist[right] = alist[left]
    # 从循环退出时left和right是相等的
    alist[left] = mid_value

    quick_sort(alist, l, left-1)
    quick_sort(alist, left+1, r)


def quick_sort_v1(alist, left, right):
    """
    快速排序算法
    平均时间复杂度:O(nlogn)
    最优时间复杂度:(序列是有序的O(n))
    最坏时间复杂度:O(n^2)
    空间复杂度:O(logn)~O(n)
    不稳定
    :param alist:
    :param left:
    :param right:
    :return:
    """

    def quick(alist, left, right):
        mid_value = alist[left]
        while left < right:
            while left < right and alist[right] >= mid_value:
                right -= 1
            alist[left] = alist[right]
            while left < right and alist[left] < mid_value:
                left += 1
            alist[right] = alist[left]
        # 从循环退出时left和right是相等的
        alist[left] = mid_value
        return left
    if left < right:
        mid_value = quick(alist, left, right)
        quick_sort(alist, mid_value, mid_value-1)
        quick_sort(alist, mid_value+1, right)
    return alist


if __name__ == "__main__":
    a = [3, 42, 5, 1, 55, 23, 44, 54, 32, 8, 10]
    quick_sort(a, 0, len(a)-1)
    # s = quick_sort_v1(a, 0, len(a)-1)
    print(a)