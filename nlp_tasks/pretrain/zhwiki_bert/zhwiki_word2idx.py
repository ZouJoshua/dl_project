#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 2/22/20 4:58 PM
@File    : zhwiki_word2idx.py
@Desc    : 

"""

import os
import json
from setting import DATA_PATH


def load_all_data(train_file, test_file):
    """
    加载所有的语料,可能需要等到20分钟左右
    :return:
    """

    with open(train_file, "r", encoding="utf-8") as f:
        all_wiki_corpus = [i for i in f.readlines()]
    with open(test_file, "r", encoding="utf-8") as f:
        all_wiki_corpus += [i for i in f.readlines()]
    print(len(all_wiki_corpus))  # 8667666

    # 因为这里上下句有重复的, 所以需要去重, 之后制作字典
    # 注意这里可能会很慢, 可能需要等到5分钟
    all_text = []
    for dic in all_wiki_corpus:
        dic = eval(dic)
        all_text += [v for _, v in dic.items()]
    all_text = list(set(all_text))
    print(len(all_text))  # 9045453
    return all_text

def get_word2tf(corpus_list):
    """
    制作所有字出现频率的dict, 然后可以舍去出现频率非常低的字
    word2tf是记录字频的dict
    :param corpus_list:
    :return:
    """
    word2tf = {}
    for text in corpus_list:
        for char in list(text):
            char = char.lower()
            word2tf = update_dic(char, word2tf)
    return word2tf

def update_dic(char, word2tf):
    if word2tf.get(char) is None:
        word2tf[char] = 1
    else:
        word2tf[char] += 1
    return word2tf

def write_word2idx(all_text, word2idx_file):
    word2tf = get_word2tf(all_text)
    print(len(word2tf))  # 19211
    # 可以根据需要舍去字频较低的字
    # word2idx是我们将要制作的字典
    word2idx = {}
    # 定义一些特殊token
    pad_index = 0  # 用来补长度和空白
    unk_index = 1  # 用来表达未知的字, 如果字典里查不到
    cls_index = 2  # CLS#
    sep_index = 3  # SEP#
    mask_index = 4  # 用来做Masked LM所做的遮罩
    num_index = 5  # (可选) 用来替换语句里的所有数字, 例如把 "23.9" 直接替换成 #num#
    word2idx["#PAD#"] = pad_index
    word2idx["#UNK#"] = unk_index
    word2idx["#SEP#"] = sep_index
    word2idx["#CLS#"] = cls_index
    word2idx["#MASK#"] = mask_index
    word2idx["#NUM#"] = num_index
    # 留出前20个token做备用, 实际字的token从序号20开始
    idx = 20
    for char, v in word2tf.items():
        word2idx[char] = idx
        idx += 1
    print(len(word2idx))

    # 写入json
    with open(word2idx_file, 'w+', encoding='utf-8') as f:
        f.write(json.dumps(word2idx, ensure_ascii=False))
    print("Word2idx File Finished!")

def main():
    corpus_path = os.path.join(DATA_PATH, "corpus", "zhwiki")
    train_file = os.path.join(corpus_path, "bert_train_wiki.txt")
    test_file = os.path.join(corpus_path, "bert_test_wiki.txt")
    word2idx_file = os.path.join(corpus_path, "bert_word2idx.join")
    all_text = load_all_data(train_file, test_file)
    write_word2idx(all_text, word2idx_file)


if __name__ == "__main__":
    main()