#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 4/27/20 11:30 AM
@File    : data08_dequeue.py
@Desc    : 双端队列

"""

class Dequeue(object):
    """
    双端队列
    """

    def __init__(self):
        self.__list = list()

    def add_front(self, item):
        """
        往队列里头部添加一个元素
        :param item:
        :return:
        """
        # self.__list.append(item)  # 时间复杂度O(1)
        self.__list.insert(0, item)  # 时间复杂度O(n)

    def add_rear(self, item):
        """
        往队列里尾部添加一个元素
        :param item:
        :return:
        """
        self.__list.append(item)  # 时间复杂度O(1)
        # self.__list.insert(0, item)  # 时间复杂度O(n)


    def pop_front(self):
        """
        从队列头部删除一个元素
        :return:
        """
        return self.__list.pop(0)  # 时间复杂度O(n)
        # return self.__list.pop      # 时间复杂度O(1),根据应用出队多还是入队多选择时间复杂度

    def pop_rear(self):
        """
        从队列尾部删除一个元素
        :return:
        """
        # return self.__list.pop(0)  # 时间复杂度O(n)
        return self.__list.pop()      # 时间复杂度O(1),根据应用出队多还是入队多选择时间复杂度

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