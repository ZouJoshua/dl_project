#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 4/27/20 11:02 AM
@File    : data07_queue.py
@Desc    : 队列

"""


class Queue(object):
    """
    队列
    """
    def __init__(self):
        self.__list = list()

    def enqueue(self, item):
        """
        往队列里添加一个元素
        :param item:
        :return:
        """
        self.__list.append(item)  # 时间复杂度O(1)
        # self.__list.insert(0, item)  # 时间复杂度O(n)

    def dequeue(self):
        """
        从队列头部删除一个元素
        :return:
        """
        return self.__list.pop(0)   # 时间复杂度O(n)
        # return self.__list.pop      # 时间复杂度O(1),根据应用出队多还是入队多选择时间复杂度

    def is_empty(self):
        """
        判断一个队列是否为空
        :return:
        """
        return self.__list == []

    def size(self):
        """
        返回队列的大小
        :return:
        """
        return len(self.__list)