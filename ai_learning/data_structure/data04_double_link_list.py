#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 4/26/20 9:35 PM
@File    : data04_double_link_list.py
@Desc    : 双向循环链表

"""

from ai_learning.data_structure.data03_single_link_list import SingleLinkList

class Node(object):
    """
    节点
    """
    def __init__(self, elem):
        self.elem = elem
        self.next = None
        self.prev = None



class DoubleLinkList(SingleLinkList):
    """

    """
    def __init__(self, node=None):
        super(DoubleLinkList, self).__init__(node)


    def add(self, item):
        """
        链表头部添加元素,头插法
        时间复杂度O(1)
        :param item:
        :return:
        """
        node = Node(item)
        node.next = self._head
        self._head = node
        node.next.prev = node


    def append(self, item):
        """
        链表尾部添加元素,尾插法
        时间复杂度O(n)
        :param item:
        :return:
        """
        node = Node(item)
        if self.is_empty():
            self._head = node
        else:
            cur = self._head
            while cur.next != None:
                cur = cur.next
            cur.next = node
            node.prev = cur


    def insert(self, pos, item):
        """
        指定位置添加元素
        :param pos: 从0开始
        :param item:
        :return:
        """
        if pos <= 0:
            self.add(item)
        elif pos > (self.length() - 1):
            self.append(item)
        else:
            cur = self._head
            count = 0
            while count <= (pos - 1):
                count += 1
                cur = cur.next

            node = Node(item)
            node.next = cur
            node.prev = cur.prev
            cur.prev.next = node
            cur.prev = node



    def remove(self, item):
        """
        删除节点
        :return:
        """
        cur = self._head
        while cur != None:
            if cur.elem == item:
                # 先判断是否是头结点
                if cur == self._head:
                    self.__head = cur.next
                    if cur.next:
                        # 判断链表是否只有一个节点
                        cur.next.prev = None
                else:
                    cur.prev.next = cur.next
                    if cur.next:
                        cur.next.prev = cur.prev
            else:
                cur = cur.next



    def search(self, item):
        """
        查找节点是否存在
        :param item:
        :return:
        """
        cur = self._head
        while cur != None:
            if cur.elem == item:
                return True
            else:
                cur = cur.next
        return False