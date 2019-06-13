#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 19-6-12 下午5:29
@File    : pre_train_video_word2vec.py
@Desc    : 预训英语新闻词向量
"""



import os
import time
from gensim.models import word2vec
import re
import string
import random

import sys
cwd = os.path.realpath(__file__)
print(cwd)
root_dir = os.path.dirname(os.path.dirname(cwd))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, "preprocess"))

from preprocess.preprocess_utils import split_text
from preprocess.preprocess_utils import read_json_format_file, read_txt_file


class CleanDoc(object):

    def __init__(self, text):
        self.text = self.clean_text(text)

    def clean_text(self, text):
        """
        清洗流程
        step1 -> 替换掉换行符、制表符等
        step2 -> 转小写
        step3 -> 清洗网址
        step4 -> 清洗邮箱
        step5 -> 清洗表情等非英文字符
        step6 -> 清洗标点符号、数字
        step7 -> 替换多个空格为一个空格
        :param text: 原始文本
        :return: 清洗后的文本
        """
        text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
        _text = text.lower()
        no_html = self._clean_html(_text)
        no_mail = self._clean_mail(no_html)
        no_emoji = self._remove_emoji(no_mail)
        no_symbol = self._remove_symbol(no_emoji)
        text = re.sub(r"\s+", " ", no_symbol)
        return text

    def _remove_emoji(self, text):
        cleaned_text = ""
        for c in text:
            if (ord(c) >= 65 and ord(c) <= 126) or (ord(c) >= 32 and ord(c) <= 63):
                cleaned_text += c
        return cleaned_text

    def _clean_html(self, text):
        # 去除网址
        pattern = re.compile(r'(?:https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]')
        # pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-zA-Z][0-9a-zA-Z]))+')
        url_list = re.findall(pattern, text)
        for url in url_list:
            text = text.replace(url, " ")
        return text.replace("( )", " ")

    def _clean_mail(self, text):
        # 去除邮箱
        pattern = re.compile(r"\w+[-_.]*[a-zA-Z0-9]+@[a-zA-Z0-9]+\.[a-zA-Z]{2,3}")
        mail_list = re.findall(pattern, text)
        for mail in mail_list:
            text = text.replace(mail, " ")
        return text

    def _remove_symbol(self, text):
        del_symbol = string.punctuation + string.digits  # ASCII 标点符号，数字
        remove_punctuation_map = dict((ord(char), None) for char in del_symbol)
        text = text.translate(remove_punctuation_map)  # 去掉ASCII 标点符号
        return text



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


def get_embed_from_rawfile(file):
    """
    直接从原始文件生成词向量训练语料
    :param file: 原始数据文件
    :return:
    """
    print(">>>>> 正在获取分词list语料")
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

def clean_text(doc):
    title = doc['title']
    content = doc['content']
    text = title + " " + content
    text = CleanDoc(text).text
    return text





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
        # doc_word_list.append(doc)
        yield word_list
    print("<<<<< 已读取{}文档".format(_doc_count))



def train_word2vec_embed_by_gensim(doc_word_list, save_path=None, model_file="word2vec.model", word2vec_file="word2vec.bin"):
    """
    gensim训练词向量
    :param doc_word_list:
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
    # corpus_data_path = "/data/word2vec/en_news"
    corpus_data_path = "/home/zoushuai/tensorflow_project/preprocess"
    file = os.path.join(corpus_data_path, "word_embed.txt")
    if not os.path.exists(file):
        write_embed_file(corpus_data_path)
    doc_word_list_all = get_embed_from_embedfile(file)
    # random.shuffle(doc_word_list_all)
    print(">>>>> 开始训练词向量")
    s = time.time()
    train_word2vec_embed_by_gensim(doc_word_list_all, corpus_data_path)
    e = time.time()
    print(">>>>> 训练{}篇文档词向量耗时{}".format(len(doc_word_list_all), e-s))


def _test_read_data():
    corpus_data_path = "/data/in_hi_news/raw_data/raw_data"
    file = os.path.join(corpus_data_path, "part-00000-69676dc0-8d50-4410-864d-79709f3f4960-c000.json")
    _doc_count = 0
    for doc in read_json_format_file(file):
        _doc_count += 1
        if _doc_count == 10:
            break
        print(doc)


if __name__ == '__main__':
    main()