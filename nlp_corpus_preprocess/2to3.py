#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 12/10/19 7:41 PM
@File    : 2to3.py
@Desc    : python2代码批量转换python3

"""


import sys
from lib2to3.main import main

sys.exit(main("lib2to3.fixes"))