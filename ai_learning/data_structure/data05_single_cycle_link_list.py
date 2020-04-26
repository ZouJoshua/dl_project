#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 4/26/20 9:36 PM
@File    : data05_single_cycle_link_list.py
@Desc    : 单向循环链表

"""



class Node(object):
    def __init__(self, elem):
        self.elem = elem
        self.next = None


class SingleCycleLinkList(object):
    """
    单向循环链表
    """
    def __init__(self, node=None):

        self.__head = None
        if node:
            node.next = node

    def is_empty(self):
        """
        链表是否为空
        :return:
        """
        return self.__head == None

    def length(self):
        """
        链表长度
        :return:
        """
        if self.is_empty():
            return 0
        # cur游标,用来移动遍历节点
        cur = self.__head
        count = 1
        while cur.next != self.__head:
            count += 1
            cur = cur.next
        return count

    def travel(self):
        """
        遍历整个链表
        :return:
        """
        if self.is_empty():
            return
        cur = self.__head
        while cur.next != self.__head:
            print(cur.elem, end=" ")
            cur = cur.next
        # 退出循环.cur指向尾节点,单尾节点的元素未打印
        print(cur.elem)


    def add(self, item):
        """
        链表头部添加元素,头插法
        时间复杂度O(1)
        :param item:
        :return:
        """
        node = Node(item)
        if self.is_empty():
            self.__head = node
            node.next = node
        else:
            cur = self.__head
            while cur.next != self.__head:
                cur = cur.next
            # 退出循环,cur指向尾节点
            node.next = self.__head
            self.__head = node
            cur.next = self.__head


    def append(self, item):
        """
        链表尾部添加元素,尾插法
        时间复杂度O(n)
        :param item:
        :return:
        """
        node = Node(item)
        if self.is_empty():
            self.__head = node
            node.next = node
        else:
            cur = self.__head
            while cur.next != self.__head:
                cur = cur.next
            node.next = self.__head
            cur.next = node


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

        pre = self.__head
        count = 0
        while count <= (pos - 1):
            count += 1
            pre = pre.next

        node = Node(item)
        node.next = pre.next
        pre.next = node



    def remove(self, item):
        """
        删除节点
        :return:
        """
        if self.is_empty():
            return
        cur = self.__head
        pre = None

        while cur != self.__head:
            if cur.elem == item:
                # 先判断是否是头结点
                if cur == self.__head:
                    # 头节点的情况
                    # 找尾节点
                    rear = self.__head
                    while rear.next != self.__head:
                        rear = rear.next
                    self.__head = cur.next
                    rear.next = self.__head
                else:
                    # 中间节点
                    pre.next = cur.next
                return
            else:
                pre = cur
                cur = cur.next

        # 退出循环,cur指向尾节点
        if cur.elem == item:
            if cur == self.__head:
                # 链表只有一个节点
                self.__head = None
            else:
                pre.next = self.__head



    def search(self, item):
        """
        查找节点是否存在
        :param item:
        :return:
        """
        if self.is_empty():
            return False
        cur = self.__head

        while cur != self.__head:
            if cur.elem == item:
                return True
            else:
                cur = cur.next
        # 退出循环,cur指向尾节点
        if cur.elem == item:
            return True
        return False