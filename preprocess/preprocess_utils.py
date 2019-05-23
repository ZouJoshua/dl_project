#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 19-5-15 下午6:37
@File    : preprocess_utils.py
@Desc    : 预处理数据
"""


import json
import os
from pyquery import PyQuery

def read_json_format_file(file):
    """
    读取每行为json格式的文本
    :param file: 文件名
    :return: 每行文本
    """
    if not os.path.exists(file):
        raise FileNotFoundError("【{}】文件未找到，请检查".format(file))
    print(">>>>> 正在读原始取数据文件：{}".format(file))
    with open(file, 'r') as f:
        while True:
            _line = f.readline()
            if not _line:
                break
            else:
                line = json.loads(_line.strip())
                yield line

def read_txt_file(file):
    """
    读取txt格式的文本
    :param file:
    :return:
    """
    if not os.path.exists(file):
        raise FileNotFoundError("【{}】文件未找到，请检查".format(file))
    print(">>>>> 正在读原始取数据文件：{}".format(file))
    with open(file, 'r') as f:
        while True:
            _line = f.readline()
            if not _line:
                break
            else:
                yield _line.strip()


def get_text_from_html(html):
    """
    从html中解析出文本内容
    :param html: html（str）
    :return: text（str）
    """
    return PyQuery(html).text().strip()


def split_text(text, lower=True, stop=None):
    """
    切分字符串（默认按空格切）
    :param text: 文本
    :param lower: 大小写（默认小写）
    :param stop: 停用词
    :return:
    """
    _text = text
    if lower:
        _text = text.lower()
    if stop:
        for i in stop:
            _text = _text.replace(i, "")
    word_list = _text.split()
    return word_list

def sort_by_value(d):
    items = d.items()
    backitems=[[v[1],v[0]] for v in items]
    backitems.sort(reverse=True)
    return [backitems[i][1] for i in range(0, len(backitems))]
