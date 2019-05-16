#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 19-5-15 下午6:13
@File    : word2vec_embedding.py
@Desc    : 
"""


import os
import time
from gensim.models import word2vec

import sys
cwd = os.path.realpath(__file__)
root_dir = os.path.dirname(os.path.dirname(cwd))
sys.path.append(os.path.join(root_dir, cwd))

from preprocess.preprocess_data import split_text
from preprocess.preprocess_data import read_json_format_file


def get_word2vec_corpus(file):
    """
    生成词向量训练语料
    :param file: 原始数据文件
    :return:
    """
    print(">>>>> 正在获取分词list语料")
    doc_word_list = list()
    _doc_count = 0
    for doc in read_json_format_file(file):
        _doc_count += 1
        if _doc_count % 100000 ==0:
            print(">>>>> 已处理{}篇文档".format(_doc_count))
        if 'title' and 'content' in doc.keys():
            title = doc["title"].strip().replace("\t", "").replace("\n", "").replace("\r", "")
            content = doc["content"].strip()
            text = title + "." + content
            word_list = split_text(text)
            doc_word_list.append(word_list)
        else:
            continue
    return doc_word_list


def train_word2vec_embed_by_gensim(doc_word_list, save_path, model_file="word2vec.model", word2vec_file="word2vec.bin"):
    """
    gensim训练词向量
    :param doc_word_list:
    :param save_path:
    :param model_file:
    :param word2vec_file:
    :return:
    """
    #skip gram
    if save_path:
        model_path = os.path.join(save_path, model_file)
        vector_path = os.path.join(save_path, word2vec_file)
    else:
        model_path = model_file
        vector_path = word2vec_file
    print(">>>>> 正在使用skip gram模型训练词向量")
    model = word2vec.Word2Vec(doc_word_list, size=300, workers=4, sg=1, iter=50)  # 默认训练词向量的时候把频次小于5的单词从词汇表中剔除掉
    model.save(model_path)
    model.wv.save_word2vec_format(vector_path, binary=True)
    print("<<<<< 词向量模型已保存【{}】".format(model_path))
    print("<<<<< 词向量embedding已保存【{}】".format(vector_path))

def main():
    # corpus_data_path = "/data/in_hi_news/raw_data/raw_data"
    corpus_data_path = "/data/zoushuai/hi_news/raw_data"
    file = os.path.join(corpus_data_path, "part-00000-69676dc0-8d50-4410-864d-79709f3f4960-c000.json")
    # doc_word_list_all = list()
    # for file in os.listdir(corpus_data_path):
    #     if file.startswith("part"):
    #         doc_word_list = get_word2vec_corpus(file)
    #         doc_word_list_all += doc_word_list
    doc_word_list_all = get_word2vec_corpus(file)
    s = time.time()
    train_word2vec_embed_by_gensim(doc_word_list_all)
    e = time.time()
    print(">>>>> 训练{}篇文档词向量耗时{}".format(len(doc_word_list_all), e-s))

if __name__ == '__main__':
    main()