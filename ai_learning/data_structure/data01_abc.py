#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 4/26/20 1:38 PM
@File    : data01_abc.py
@Desc    : 

"""

"""
a + b +c = 1000,  且 a^2+ b^2 = c^2，求出所有abc的组合
"""


import time
# 枚举法(163.82774901390076s)
s_time = time.time()
for a in range(0, 1001):
    for b in range(0, 1001):
        for c in range(0, 1001):
            if a + b + c == 1000 and a **2 + b**2 == c**2:
                print(a, b, c)
e_time = time.time()
print("耗时:{}".format(e_time - s_time))



# 优化方法 (1.0311992168426514s)

s_time = time.time()
for a in range(0, 1001):
    for b in range(0, 1001):
        c = 1000 - a - b
        if a **2 + b **2 == c**2:
            print(a, b, c)
e_time = time.time()
print("耗时:{}".format(e_time - s_time))
