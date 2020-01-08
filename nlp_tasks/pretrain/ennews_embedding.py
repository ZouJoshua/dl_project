#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 19-6-12 下午5:29
@File    : ennews_embedding.py
@Desc    : 预训英语新闻词向量
"""


import os
import time
from gensim.models import word2vec
from preprocess.tools import split_text, CleanDoc
from preprocess.tools import read_json_format_file, read_txt_file


def clean_text(doc):
    title = doc['title']
    content = doc['content']
    text = title + " " + content
    text = CleanDoc(text).text
    return text


def get_embed_from_rawfile(file):
    """
    直接从原始文件生成词向量训练语料
    :param file: 原始数据文件
    :return:
    """
    print(">>>>> 正在从原始文件获取分词list语料")
    doc_word_list = list()
    _doc_count = 0
    for doc in read_json_format_file(file):
        _doc_count += 1
        if _doc_count % 100000 == 0:
            print(">>>>> 已处理{}篇文档".format(_doc_count))
        clean_doc = clean_text(doc)
        if clean_doc:
            doc_word_list.append(clean_doc)
        else:
            continue
    return doc_word_list


def write_embed_file(corpus_data_path):
    """
    处理为embed训练文件，清洗文本
    所以采取分层抽取1/4数据写入文件
    :return:
    """
    raw_file = os.path.join(corpus_data_path, "raw_data")
    embed_file = os.path.join(corpus_data_path, "word_embed.txt")
    ef = open(embed_file, 'w')
    _doc_count = 0
    for doc in read_json_format_file(raw_file):
        _doc_count += 1
        if _doc_count % 100000 == 0:
            print(">>>>> 已处理{}篇文档".format(_doc_count))
        clean_doc = clean_text(doc)
        if clean_doc:
            ef.write(clean_doc + "\n")
        else:
            continue
    ef.close()
    print("<<<<< 【{}】embed文件已生成".format(embed_file))
    return _doc_count


def get_embed_from_embedfile(file):
    """
    从embed文件获取word2vec训练语料
    :param file: embed 文件
    :return:
    """
    print(">>>>> 正在读取embed语料")
    doc_word_list = list()
    _doc_count = 0
    for doc in read_txt_file(file):
        _doc_count += 1
        word_list = split_text(doc)
        doc_word_list.append(word_list)
    print("<<<<< 已读取{}文档".format(_doc_count))
    return doc_word_list


class Sentences(object):

    def __init__(self, filename):
        self.file = filename

    def __iter__(self):
        print(">>>>> 正在读取embed语料")
        _doc_count = 0
        for doc in read_txt_file(self.file):
            _doc_count += 1
            word_list = split_text(doc)
            yield word_list
        print("<<<<< 已读取{}文档".format(_doc_count))



def train_word2vec_embed_by_gensim(doc_word_list, save_path=None, model_file="word2vec.model", word2vec_file="word2vec.bin"):
    """
    gensim训练词向量
    :param doc_word_list: a memory-friendly iterator or list
    :param save_path:
    :param model_file:
    :param word2vec_file:
    :return:
    """
    # 引入日志配置
    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    #skip gram（预测更准确些）
    if save_path:
        model_path = os.path.join(save_path, model_file)
        vector_path = os.path.join(save_path, word2vec_file)
    else:
        model_path = model_file
        vector_path = word2vec_file
    print(">>>>> 正在使用skip gram模型训练词向量")
    model = word2vec.Word2Vec(doc_word_list, size=300, workers=8, sg=1, iter=50)  # 默认训练词向量的时候把频次小于5的单词从词汇表中剔除掉
    model.save(model_path)
    model.wv.save_word2vec_format(vector_path, binary=True)
    print("<<<<< 词向量模型已保存【{}】".format(model_path))
    print("<<<<< 词向量embedding已保存【{}】".format(vector_path))




def main():
    corpus_data_path = "/data/word2vec/en_news"
    # corpus_data_path = "/home/zoushuai/tensorflow_project/preprocess"
    file = os.path.join(corpus_data_path, "word_embed.txt")
    sentences = None
    if not os.path.exists(file):
        doc_count = write_embed_file(corpus_data_path)
        if doc_count > 100000:
            sentences = Sentences(file)
        else:
            sentences = get_embed_from_embedfile(file)
    if not sentences:
        sentences = Sentences(file)
    # random.shuffle(doc_word_list_all)
    print(">>>>> 开始训练词向量")
    s = time.time()
    train_word2vec_embed_by_gensim(sentences, corpus_data_path)
    e = time.time()
    print(">>>>> 训练文档词向量耗时{}".format(e-s))


def _test_read_data():
    corpus_data_path = "/home/zoushuai/tensorflow_project/preprocess"
    file = os.path.join(corpus_data_path, "word_embed.txt")
    sentences = Sentences(file)
    print(type(sentences))
    print(sentences)


if __name__ == '__main__':
    main()
    # _test_read_data()