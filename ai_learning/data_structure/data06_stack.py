#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 4/27/20 12:01 AM
@File    : data06_stack.py
@Desc    : 栈

"""




class Stack(object):
    """
    栈
    """
    def __init__(self):
        self.__list = list()


    def push(self, item):
        """
        添加一个新的元素item到栈顶
        :param item:
        :return:
        """
        self.__list.append(item)

    def pop(self):
        """
        弹出栈顶元素
        :return:
        """
        return self.__list.pop()


    def peek(self):
        """
        返回栈顶元素
        :return:
        """
        if self.__list:
            return self.__list[-1]
        else:
            return None

    def is_empty(self):
        """
        判断栈是否为空
        :return:
        """
        return self.__list == []

    def size(self):
        """
        返回栈的元素个数
        :return:
        """
        return len(self.__list)