#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 19-8-8 下午2:46
@File    : preprocess_sensitive_words.py
@Desc    : 预处理敏感词
"""

import json


def en_sensitive_words_merge(raw_file, sensitive_type_file, type_file, result_file):
    with open(raw_file, 'r') as f:
        lines = f.readlines()
    word_list1 = [line.strip() for line in lines]
    with open(sensitive_type_file, 'r') as f:
        word_dict = json.load(f)
    word_list2 = list(word_dict.keys())
    sensitive_word_list = word_list1 + word_list2
    sensitive_words = sorted(set(sensitive_word_list))
    sensitive_type_dict = dict()
    for k, v in word_dict.items():
        if v in sensitive_type_dict:
            sensitive_type_dict[v].append(k)
        else:
            sensitive_type_dict[v] = list()
            sensitive_type_dict[v].append(k)
    with open(type_file, "w") as tf:
        tf.write(json.dumps(sensitive_type_dict, ensure_ascii=False, indent=4))

    with open(result_file, "w") as rf:
        for word in sensitive_words:
            rf.write(word + "\n")


def main():
    en_raw_file = "/home/zoushuai/algoproject/tf_project/data/sensitive_words/en_raw"
    en_sensitive_type_file = "/home/zoushuai/algoproject/tf_project/data/sensitive_words/en_sensitive_type.json"
    sensitive_type_file = "/home/zoushuai/algoproject/tf_project/data/sensitive_words/sensitive_type.json"
    result_file = "/home/zoushuai/algoproject/tf_project/data/sensitive_words/en"
    en_sensitive_words_merge(en_raw_file, en_sensitive_type_file, sensitive_type_file, result_file)


if __name__ == '__main__':
    main()