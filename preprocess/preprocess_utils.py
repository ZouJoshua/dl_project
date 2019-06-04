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

def write_json_format_file(source_data, file):
    """
    写每行为json格式的文件
    :param source_data:
    :param file:
    :return:
    """
    print(">>>>> 正在写入目标数据文件：{}".format(file))
    f = open(file, "w")
    _count = 0
    for _line in source_data:
        _count += 1
        if _count % 100000 == 0:
            print("<<<<< 已写入{}行".format(_count))
        if isinstance(_line, dict):
            line = json.dumps(_line)
            f.write(line + "\n")
        elif isinstance(_line, str):
            f.write(_line + "\n")
    f.close()


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


def get_ngrams(sentence, n_gram=3):
    """
     # 将一句话转化为(uigram,bigram,trigram)后的字符串
    :param sentence: string. example:'w17314 w5521 w7729 w767 w10147 w111'
    :param n_gram:
    :return:string. example:'w17314 w17314w5521 w17314w5521w7729 w5521 w5521w7729 w5521w7729w767 w7729 w7729w767 w7729w767w10147 w767 w767w10147 w767w10147w111 w10147 w10147w111 w111'
    """
    result = list()
    word_list = sentence.split(" ") #[sentence[i] for i in range(len(sentence))]
    unigram = ''
    bigram = ''
    trigram = ''
    fourgram = ''
    length_sentence = len(word_list)
    for i, word in enumerate(word_list):
        unigram = word                           # ui-gram
        word_i = unigram
        if n_gram >= 2 and i+2 <= length_sentence: #bi-gram
            bigram = "".join(word_list[i:i+2])
            word_i = word_i + ' ' + bigram
        if n_gram >= 3 and i+3 <= length_sentence: #tri-gram
            trigram = "".join(word_list[i:i+3])
            word_i = word_i + ' ' + trigram
        if n_gram >= 4 and i+4 <= length_sentence: #four-gram
            fourgram = "".join(word_list[i:i+4])
            word_i = word_i + ' ' + fourgram
        if n_gram >= 5 and i+5 <= length_sentence: #five-gram
            fivegram = "".join(word_list[i:i+5])
            word_i = word_i + ' ' + fivegram
        result.append(word_i)
    result = " ".join(result)
    return result
