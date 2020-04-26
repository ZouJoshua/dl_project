#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 4/26/20 1:43 PM
@File    : data02_list.py
@Desc    : 

"""
from timeit import Timer



def t1():
    """
    测试使用append
    :return:
    """
    li = []
    for i in range(10000):
        li.append(li)


def t2():
    """
    测试使用列表加和的方法
    :return:
    """
    li = []
    for i in range(10000):
        li += [i]

def t3():
    """
    测试使用列表生成器
    :return:
    """
    li = [i for i in range(10000)]


def t4():
    """
    测试使用列表转换
    :return:
    """
    li = list(range(10000))


def t5():
    """
    测试使用extend的方法
    :return:
    """
    li = list()
    for i in range(10000):
        li.extend([i])


def t6():
    """
    测试使用从头部insert
    :return:
    """
    li = list()
    for i in range(10000):
        li.insert(0, i)


timer1 = Timer("t1()", "from __main__ import t1")
print("append:", timer1.timeit(1000))
timer2 = Timer("t2()", "from __main__ import t2")
print("+:", timer1.timeit(1000))
timer3 = Timer("t3()", "from __main__ import t3")
print("[i for i in range]", timer1.timeit(1000))
timer4 = Timer("t4()", "from __main__ import t4")
print("list(range())", timer1.timeit(1000))