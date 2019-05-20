#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 19-5-17 下午2:14
@File    : preprocess_data_hi.py
@Desc    : 
"""


import os
import time
import pickle
import json
import random

from sklearn.model_selection import StratifiedKFold
from gensim.models import KeyedVectors
from preprocess.preprocess_utils import read_json_format_file,split_text


class DataSet(object):

    def __init__(self, data_dir, word2vec_path, training_data_path):
        self.data_dir = data_dir
        self.word2index, self.index2word = self.create_vocabulary(word2vec_path)
        self.label2index, self.index2label = self.create_label_vocabulary(training_data_path)

    def create_vocabulary(self, word2vec_model_path, pad_list=None):
        """
        创建词汇索引映射表,将其序列化(或保存为json)，之后做inference时直接加载来处理数据
        :param word2vec_model_path: 训练好的word2vec模型存放路径
        :return: {单词：索引}表和{索引：单词}表
        """

        cache_path = os.path.join(self.data_dir + "word_vocabulary.pik")
        print("cache_path:", cache_path, "file_exists:", os.path.exists(cache_path))

        if os.path.exists(cache_path):  # 如果缓存文件存在，则直接读取
            with open(cache_path, 'rb') as data_f:
                _word2idx, _idx2word = pickle.load(data_f)
                return _word2idx, _idx2word
        else:
            print("building vocabulary.\nword2vec_model_path:", word2vec_model_path)
            _word2idx = dict()
            _idx2word = dict()
            if not pad_list:
                # 默认添加 "pad" 和 "UNK"
                pad_list = ["pad", "UNK"]
            add_index = len(pad_list)
            for i, pad_vocab in enumerate(pad_list):
                _word2idx[pad_vocab] = i
                _idx2word[i] = pad_vocab

            # model = Word2Vec.load(word2vec_model_path)
            model = KeyedVectors.load_word2vec_format(word2vec_model_path, binary=True)

            for i, vocab in enumerate(model.wv.vocab):
                _word2idx[vocab] = i + add_index     # 设置多个pad id
                _idx2word[i + add_index] = vocab

            # 如果不存在写到缓存文件中
            if not os.path.exists(cache_path):
                with open(cache_path, 'ab') as data_f:
                    pickle.dump((_word2idx, _idx2word), data_f)
        return _word2idx, _idx2word

    def create_label_vocabulary(self, training_data_path):
        """
        创建标签映射
        :param training_data_path: 带label的训练语料
        :return: label2idx和idx2label
        """
        print("building vocabulary_label_sorted. training_data:", training_data_path)
        cache_path = os.path.join(self.data_dir + "label_vocabulary.pik")
        print("cache_path:", cache_path, "file_exists:", os.path.exists(cache_path))

        if os.path.exists(cache_path):  # 如果缓存文件存在，则直接读取
            with open(cache_path, 'rb') as data_f:
                _label2idx, _idx2label = pickle.load(data_f)
                return _label2idx, _idx2label
        else:
            _label2idx = dict()
            _idx2label = dict()
            label_count_dict = dict()  # {label:count} 统计各类别的样本数
            lines = read_json_format_file(training_data_path)
            for line in lines:
                label = line['top_category']
                if label_count_dict.get(label, None) is not None:
                    label_count_dict[label] += 1
                else:
                    label_count_dict[label] = 1

            list_label = self.sort_by_value(label_count_dict)  # 按样本数降序排之后的key列表
            print("length of list_label:", len(list_label))

            countt = 0
            for i, label in enumerate(list_label):
                if i < 10:
                    count_value = label_count_dict[label]
                    print("label:", label, "count_value:", count_value)
                    countt += count_value
                _label2idx[label] = i
                _idx2label[i] = label
            print("count top10:", countt)

            # 如果不存在写到缓存文件中
            if not os.path.exists(cache_path):
                with open(cache_path, 'ab') as data_f:
                    pickle.dump((_label2idx, _idx2label), data_f)
        print("building vocabulary_label(sorted) ended.len of vocabulary_label: ", len(_idx2label))
        return _label2idx, _idx2label


    def sort_by_value(self, data_dict):
        _result_sort = sorted(data_dict.items(), key=lambda x: x[1], reverse=True)
        return [_result_sort[i][1] for i in range(0, len(_result_sort))]

    def load_data(self, word2idx, label2idx, training_data_path, valid_portion=0.2):
        """
        划分训练集、测试集、验证集
        input: a file path
        :return: train, test, valid. where train=(trainX, trainY). where
                        trainX: is a list of list.each list representation a sentence.trainY: is a list of label. each label is a number
        """
        # 1. load raw data
        print("load_data.started...")
        print("load_data.training_data_path:", training_data_path)
        lines = read_json_format_file(training_data_path)
        # 2.transform X as indices
        """
        #todo: 去掉停用词 -> 统计词频 -> 去除低频词
        """
        # 3.transform  y as scalar
        X = []
        Y = []
        Y_decoder_input = []
        for i, line in enumerate(lines):
            title = line["title"].strip().replace("\t", " ").replace("\n", " ").replace("\r", " ")
            content = line["content"].strip()
            x = title + " " + content
            y = line["top_category"]
            # 打印前几条
            if i < 5:
                print("x{}:".format(i), x)  # get raw x
            x = split_text(x)
            x = [word2idx.get(w, 0) for w in x]  # 若找不到单词，用0填充
            if i < 5:
                print("x{}-word-index:".format(i), x)  # word to index
            y = label2idx[y]
            X.append(x)
            Y.append(y)

        # 4.split to train,test and valid data(基于y标签分层)
        doc_num = len(X)
        print("number_doc:", doc_num)
        train, test = self.stratified_sampling(X, Y, valid_portion)
        print("load_data ended...")
        return train, test, test


    def stratified_sampling(self, x, y, valid_portion):

        skf = StratifiedKFold(n_splits=int(1/valid_portion))
        i = 0
        for train_index, test_index in skf.split(x, y):
            i += 1
            if i < 2:
                train_label_count = self._label_count([y[i] for i in train_index])
                test_label_count = self._label_count([y[j] for j in test_index])
                print("train_label_count: {}".format(train_label_count))
                print("test_label_count: {}".format(test_label_count))
                train = [(x[i], y[i]) for i in train_index]
                test = [(x[j], y[j]) for j in test_index]
                return train, test

    def _label_count(self, label_list):
        label_count = dict()
        for i in label_list:
            if label_count.get(i, None) is not None:
                label_count[i] += 1
            else:
                label_count[i] = 1
        return label_count



class TopcategoryCorpus(object):

    def __init__(self, in_file1, in_file2, out_file):
        self.f1 = in_file1
        self.f2 = in_file2
        self.out = out_file
        self.get_topcategory_corpus()

    def read_json_format_file(self, file):
        print(">>>>> 正在读原始取数据文件：{}".format(file))
        with open(file, 'r') as f:
            while True:
                _line = f.readline()
                if not _line:
                    break
                else:
                    line = json.loads(_line)
                    yield line

    def shuff_data(self):
        data_all = list()
        for line in self.read_json_format_file(self.f1):
            if 'tags' in line.keys():
                del line['tags']
            if line['top_category'] not in ("technology", "auto", "science"):
                data_all.append(line)
        for line_ in self.read_json_format_file(self.f2):
            data_all.append(line_)
        # print(len(data_all))
        random.shuffle(data_all)
        return data_all

    def get_topcategory_corpus(self):
        print(">>>>> 正在处理训练语料")
        o_file = open(self.out, 'w')
        category_count = dict()
        for line in self.shuff_data():
            if line['top_category'] in category_count.keys():
                if category_count[line['top_category']] < 10000:
                    category_count[line['top_category']] += 1
                    o_file.write(json.dumps(line, ensure_ascii=False) + "\n")
                else:
                    continue
            else:
                category_count[line['top_category']] = 1
                o_file.write(json.dumps(line, ensure_ascii=False) + "\n")
        print(">>>>> 各一级类类别：\n{}".format(json.dumps(category_count, indent=4)))
        print("<<<<< 训练语料已处理完成：{}".format(self.out))


def main():
    data_base_dir = r'/data/in_hi_news/train_data'
    file1 = os.path.join(data_base_dir, 'topcategory_all')
    file2 = os.path.join(data_base_dir, 'auto_science_tech')
    out_file = os.path.join(data_base_dir, 'top_category_corpus')
    TopcategoryCorpus(file1, file2, out_file)



