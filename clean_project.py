#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 2020/9/23 4:37 下午
@File    : clean_project.py
@Desc    : 

"""

import os
import shutil

def clear(filepath):
    files = os.listdir(filepath)
    for fd in files:
        cur_path = os.path.join(filepath, fd)
        if os.path.isdir(cur_path):
            if fd == "__pycache__":
                print("rm -rf %s" % cur_path)
                # os.system("rm -rf %s" % cur_path)
                shutil.rmtree(cur_path, ignore_errors=True)

            else:
                clear(cur_path)

if __name__ == "__main__":
    clear("./")