#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 4/26/20 6:43 PM
@File    : data03_single_link_list.py
@Desc    : 链表

"""



class Node(object):
    def __init__(self, elem):
        self.elem = elem
        self.next = None


class SingleLinkList(object):
    """
    单链表
    """
    def __init__(self, node=None):
        self.__head = node


    def is_empty(self):
        """
        链表是否为空
        :return:
        """
        return self._head is None

    def length(self):
        """
        链表长度
        :return:
        """
        # cur游标,用来移动遍历节点
        cur = self._head
        count = 0
        while cur != None:
            count += 1
            cur = cur.next
        return count

    def travel(self):
        """
        遍历整个链表
        :return:
        """
        cur = self._head
        while cur != None:
            print(cur.elem)
            cur = cur.next



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

        pre = self._head
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
        cur = self._head
        pre = None
        while cur != None:
            if cur.elem == item:
                # 先判断是否是头结点
                if cur == self._head:
                    self.__head = cur.next
                else:
                    pre.next = cur.next
            else:
                pre = cur
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



if __name__ == "__main__":
    ll = SingleLinkList()
    print(ll.is_empty())
    print(ll.length())
    ll.append(1)
    print(ll.is_empty())
    print(ll.length())

    ll.append(2)
    ll.append(3)
    ll.append(4)
    ll.append(5)
    ll.append(6)
    print(ll)
    ll.travel()
