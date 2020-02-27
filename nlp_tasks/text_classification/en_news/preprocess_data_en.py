#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 19-7-1 上午10:14
@File    : preprocess_data_en.py
@Desc    : 处理英文新闻
"""

import json
import os
import random
import string
import re
import numpy as np
import pickle
import h5py
from preprocess.common_tools import read_json_format_file, CleanDoc, split_text
from gensim.models import KeyedVectors
from sklearn.model_selection import StratifiedKFold


def rewrite_train_cropus(input_file, output_file):
    """
    预处理训练语料
    1.修改article_id 为id
    2.抽样每个label25000样本
    3.统计文本内容分词后的平均大小
    :param input_file: 原始数据文件
    :param output_file: 处理后文件
    :return:
    """
    print(">>>>> 正在重新写入训练语料")
    out = open(output_file, 'w')
    all_corpus = list()
    new_corpus = dict()
    for line in read_json_format_file(input_file):
        line["id"] = line.pop("article_id")
        line["tokens"] = len(line["content"].split(" "))
        all_corpus.append(line)
    random.shuffle(all_corpus)

    for doc in all_corpus:
        category = doc["one_level"]
        if category in new_corpus.keys():
            if new_corpus[category] < 25000:
                new_corpus[category] += 1
                new_corpus[category + "_tokens_len"] += doc["tokens"]
                out.write(json.dumps(doc) + "\n")
            else:
                continue
        else:
            new_corpus[category] = 1
            new_corpus[category+"_tokens_len"] = doc["tokens"]
            out.write(json.dumps(doc) + "\n")
    out.close()
    # print(new_corpus)
    for k in new_corpus.keys():
        print(k)
        it = k.split("_tokens_")
        if len(it) == 2:
            _k = it[0]
            new_corpus[_k + "_avg_token_len"] = int(new_corpus.pop(_k + "_tokens_len") / new_corpus[_k])
        else:
            continue
    print(">>>>> 写入文件各类别样本量\n{}".format(json.dumps(new_corpus, indent=4)))
    print("<<<<< 已写入文件【{}】新的训练语料".format(output_file))


class SplitData2tsv(object):
    """
    处理训练文件为bert模型格式
    """
    def __init__(self, in_file1, out_dir):
        self.f1 = in_file1
        # self.ft_file = os.path.join(out_dir, 'train_corpus')
        self.train_file = os.path.join(out_dir, 'train.tsv')
        self.dev_file = os.path.join(out_dir, 'dev.tsv')
        self.test_file = os.path.join(out_dir, 'test.tsv')
        self.label2index = {"national": 4, "tech": 10,
                            "sports": 6, "science": 9,
                            "international": 3, "business": 8,
                            "entertainment": 15, "lifestyle": 12,
                            "auto": 11}
        self.get_category_corpus_file()


    def get_data(self):
        X = list()
        Y = list()
        _count = dict()
        for line in read_json_format_file(self.f1):
            if line:
                # result = self._preline(line)
                result = self._preline_v2(line)
                if result:
                    x, y = result
                    if y in _count.keys():
                        if _count[y] > 25000:
                            continue
                        else:
                            _count[y] += 1
                            X.append(x)
                            Y.append(y)
                    else:
                        _count[y] = 1
                        X.append(x)
                        Y.append(y)
        return X, Y

    def stratified_sampling(self, x, y, valid_portion):

        skf = StratifiedKFold(n_splits=int(1/valid_portion))
        i = 0
        for train_index, test_index in skf.split(x, y):
            train_label_id_count = self._label_count([y[i] for i in train_index])
            test_label_id_count = self._label_count([y[j] for j in test_index])
            # train_label_count = dict()
            # test_label_count = dict()
            # for cat, count in train_label_id_count.items():
            #     train_label_count[cat] = count
            # for cat, count in test_label_id_count.items():
            #     test_label_count[cat] = count
            print("train_label_count: {}".format(json.dumps(train_label_id_count, indent=4)))
            print("test_label_count: {}".format(json.dumps(test_label_id_count, indent=4)))
            train = [str(self.label2index[y[i]]) + "\t" + x[i] for i in train_index]
            dev = [str(self.label2index[y[j]]) + "\t" + x[j] for j in test_index]
            i += 1
            if i < 2:
                break

            return train, dev

    def _label_count(self, label_list):
        label_count = dict()
        for i in label_list:
            if label_count.get(i, None) is not None:
                label_count[i] += 1
            else:
                label_count[i] = 1
        return label_count

    def clean_title(self, text):
        """
        清洗标题
        :param text:
        :return:
        """
        text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
        text = text.lower()
        no_emoji = CleanDoc(text)._remove_emoji(text)
        del_symbol = string.punctuation  # ASCII 标点符号
        remove_punctuation_map = dict((ord(char), " ") for char in del_symbol)
        text = no_emoji.translate(remove_punctuation_map)  # 去掉ASCII 标点符号
        text = re.sub(r"\s+", " ", text)
        return text

    def _preline(self, line_json):
        """
        处理文件每一行（包括title和content）
        :param line_json:
        :return:
        """
        # line_json = json.loads(line)
        title = line_json["title"]
        content = ""
        dataY = line_json["one_level"]
        if "content" in line_json:
            content = line_json["content"]
        if title or content:
            _dataX = title + ' ' + content
            dataX = CleanDoc(_dataX).text  # 清洗数据
            return dataX, dataY
        else:
            return None


    def _preline_v2(self, line_json):
        """
        处理文件每一行（只包括title）
        :param line_json:
        :return:
        """
        title = line_json["title"]
        dataY = line_json["one_level"]
        t_str = self.clean_title(title)  # 清洗数据
        if t_str.replace(" ", ""):
            # dataX = title
            dataX = t_str
            return dataX, dataY
        else:
            return None


    def write_tvs_file(self, data, file):
        print(">>>>> 正在写入文件")
        with open(file, "w") as f:
            for line in data:
                f.write(line + "\n")
        print("<<<<< 已写入到文件：{}".format(file))


    def get_category_corpus_file(self):
        print(">>>>> 正在处理训练语料")
        X, Y = self.get_data()
        train, dev = self.stratified_sampling(X, Y, 0.2)
        self.write_tvs_file(train, self.train_file)
        self.write_tvs_file(dev, self.dev_file)
        self.write_tvs_file(dev, self.test_file)




class DataSet(object):
    """
    深度学习模型数据数据格式
    """

    def __init__(self, data_dir, word2vec_file, training_data_file, embedding_dims=300):
        """
        数据预处理
        step1 -> 创建word2vec词汇索引映射表
        step2 -> 创建label映射表
        step3 -> 划分训练集、测试集、验证集

        :param data_dir: 数据资源目录
        :param word2vec_file: 预训练词向量文件
        :param training_data_file: 原始数据文件
        :param embedding_dims: embedding词向量维度
        """
        self.data_dir = data_dir
        self.word2vec_path = word2vec_file
        self.raw_data_path = training_data_file
        self.embedding = None
        self.embed_dim = embedding_dims
        self.word2index, self.index2word, self.word2embed = self.create_vocabulary(embedding_dims=self.embed_dim)
        self.label2index, self.index2label = self.create_label_vocabulary()

    def create_vocabulary(self, embedding_dims=300, pad_list=None):
        """
        创建词汇索引映射表,将其序列化(或保存为json)，之后做inference时直接加载来处理数据
        :param word2vec_model_path: 训练好的word2vec模型存放路径
        :return: {单词：索引}表和{索引：单词}表
        """

        cache_path = os.path.join(self.data_dir, "word_vocabulary.pik")
        print("cache_path:", cache_path, "file_exists:", os.path.exists(cache_path))

        if os.path.exists(cache_path):  # 如果缓存文件存在，则直接读取
            with open(cache_path, 'rb') as data_f:
                _word2idx, _idx2word, _word2embed = pickle.load(data_f)
                return _word2idx, _idx2word, _word2embed
        else:
            print("building vocabulary.\nword2vec_model_path:", self.word2vec_path)
            _word2idx = dict()
            _idx2word = dict()
            _word2embed = dict()
            if not pad_list:
                # 默认添加 "pad" 和 "UNK"
                pad_list = ["PAD", "UNK"]
                # pad_list = ['PAD', 'UNK', 'CLS', 'SEP', 'unused1', 'unused2', 'unused3', 'unused4', 'unused5']
                _word2embed["PAD"] = np.zeros(embedding_dims)
                _word2embed["UNK"] = np.random.randn(embedding_dims)
                # bound = np.sqrt(6.0) / np.sqrt(vocab_size)  # bound for random variables
                # _word2embed["UNK"] = np.random.uniform(-bound, bound, embedding_dims)
            add_index = len(pad_list)
            for i, pad_vocab in enumerate(pad_list):
                _word2idx[pad_vocab] = i
                _idx2word[i] = pad_vocab

            # model = Word2Vec.load(word2vec_model_path)
            model = KeyedVectors.load_word2vec_format(self.word2vec_path, binary=True)

            for i, vocab in enumerate(model.wv.vocab):
                _word2idx[vocab] = i + add_index     # 设置多个pad id
                _idx2word[i + add_index] = vocab
                _word2embed[vocab] = model.wv[vocab]

            # 如果不存在写到缓存文件中
            if not os.path.exists(cache_path):
                with open(cache_path, 'ab') as data_f:
                    pickle.dump((_word2idx, _idx2word, _word2embed), data_f)
        return _word2idx, _idx2word, _word2embed

    def create_label_vocabulary(self):
        """
        创建标签映射
        :param training_data_path: 带label的训练语料
        :return: label2idx和idx2label
        """
        print("building vocabulary_label_sorted. training_data:", self.raw_data_path)
        cache_path = os.path.join(self.data_dir, "label_vocabulary.pik")
        print("cache_path:", cache_path, "file_exists:", os.path.exists(cache_path))

        if os.path.exists(cache_path):  # 如果缓存文件存在，则直接读取
            with open(cache_path, 'rb') as data_f:
                _label2idx, _idx2label = pickle.load(data_f)
                return _label2idx, _idx2label
        else:
            _label2idx = dict()
            _idx2label = dict()
            label_count_dict = dict()  # {label:count} 统计各类别的样本数
            lines = read_json_format_file(self.raw_data_path)
            for line in lines:
                label = str(line['one_level'])
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
            print("count top10 labels:", countt)

            # 如果不存在写到缓存文件中
            if not os.path.exists(cache_path):
                with open(cache_path, 'ab') as data_f:
                    pickle.dump((_label2idx, _idx2label), data_f)
        print("building vocabulary_label(sorted) ended.len of vocabulary_label: ", len(_idx2label))
        return _label2idx, _idx2label

    def load_data(self, use_embedding=True, valid_portion=0.2):
        """
        划分训练集、测试集、验证集
        :param use_embedding: 提供embedding 词向量（默认提供）
        :param valid_portion: 测试集比例（默认0.2）
        :return: train, test, valid. where train=(trainX, trainY). where
                        trainX: is a list of list.each list representation a sentence.trainY: is a list of label. each label is a number
        """
        # 1. load raw data
        print("load_data.started...")
        print("load_data.training_data_path:", self.raw_data_path)
        lines = [line for line in read_json_format_file(self.raw_data_path)]
        random.shuffle(lines)
        # 2.transform X as indices
        """
        #todo: 去掉停用词 -> 统计词频 -> 去除低频词
        """
        # 3.transform  y as scalar
        X = []
        Y = []
        count_not_exist = 0
        if use_embedding:
            self.embedding = self.get_embedding(self.word2embed)

        for i, line in enumerate(lines):
            title = line["title"].strip()
            content = line["content"].strip()
            x = title + " " + content
            # x = self.clean_title(title)
            x = CleanDoc(x).text
            y = str(line["one_level"])
            # 打印前几条
            if i < 1:
                print("x{}:".format(i), x)  # get raw x
            x = split_text(x)
            x = [self.word2index.get(w, 0) for w in x]  # 若找不到单词，用0填充
            if i < 1:
                print("x{}-word-index:".format(i), x)  # word to index
            y = self.label2index[y]
            X.append(x)
            Y.append(y)


        # 4.split to train,test and valid data(基于y标签分层)
        doc_num = len(X)
        print("number_doc:", doc_num)
        train, test = self.stratified_sampling(X, Y, valid_portion)
        print("load_data ended...")
        return train, test, test

    def load_data_sample(self, use_embedding=True, valid_portion=0.2):
        """
        划分训练集、测试集、验证集(小批量数据验证模型网络结构)
        :param use_embedding: 提供embedding 词向量（默认提供）
        :param valid_portion: 测试集比例（默认0.2）
        :return: train, test, valid. where train=(trainX, trainY). where
                        trainX: is a list of list.each list representation a sentence.trainY: is a list of label. each label is a number
        """
        # 1. load raw data
        print("load_data.started...")
        print("load_data.training_data_path:", self.raw_data_path)
        lines = read_json_format_file(self.raw_data_path)
        # 2.transform X as indices
        """
        #todo: 去掉停用词 -> 统计词频 -> 去除低频词
        """
        # 3.transform  y as scalar
        X = []
        Y = []
        count_not_exist = 0
        if use_embedding:
            self.embedding = self.get_embedding(self.word2embed)

        for i, line in enumerate(lines):
            title = line["title"].strip()
            content = line["content"].strip()
            x = title + " " + content
            x = CleanDoc(x).text
            y = str(line["one_level"])
            # 打印前几条
            # if i < 1:
            #     print("x{}:".format(i), x)  # get raw x
            x = split_text(x)
            x = [self.word2index.get(w, 0) for w in x]  # 若找不到单词，用0填充
            if i < 1:
                print("x{}-word-index:".format(i), x)  # word to index
            y = self.label2index[y]
            if i < 1000:
                X.append(x)
                Y.append(y)
            else:
                pass


        # 4.split to train,test and valid data(基于y标签分层)
        doc_num = len(X)
        print("number_doc:", doc_num)
        train, test = self.stratified_sampling(X, Y, valid_portion)
        print("load_data ended...")
        return train, test, test


    def get_embedding(self, word2embed):
        vocab_size = len(word2embed)
        unk = word2embed.get("UNK", np.random.randn(self.embed_dim))
        word_embedding_list = [[]] * vocab_size  # create an empty word_embedding list.
        for i, vocab in enumerate(word2embed):
            word_embedding_list[i] = word2embed.get(vocab, unk)
        word_embedding = np.array(word_embedding_list)
        return word_embedding

    def stratified_sampling(self, x, y, valid_portion):

        skf = StratifiedKFold(n_splits=int(1/valid_portion))
        i = 0
        for train_index, test_index in skf.split(x, y):
            i += 1
            if i < 2:
                train_label_id_count = self._label_count([y[i] for i in train_index])
                test_label_id_count = self._label_count([y[j] for j in test_index])
                train_label_count = dict()
                test_label_count = dict()
                for id, count in train_label_id_count.items():
                    train_label_count[self.index2label[id]] = count
                for id, count in test_label_id_count.items():
                    test_label_count[self.index2label[id]] = count
                print("train_label_count: {}".format(json.dumps(train_label_count, indent=4)))
                print("test_label_count: {}".format(json.dumps(test_label_count, indent=4)))
                trainx = [x[i] for i in train_index]
                trainy = [y[i] for i in train_index]
                testx = [x[j] for j in test_index]
                testy = [y[j] for j in test_index]
                return (trainx, trainy), (testx, testy)

    def load_data_from_h5py(self, cache_file_h5py, cache_file_pickle):
        """
        load data from h5py and pickle cache files, which is generate by take step by step of pre-processing.ipynb
        :param cache_file_h5py:
        :param cache_file_pickle:
        :return:
        """
        if not os.path.exists(cache_file_h5py) or not os.path.exists(cache_file_pickle):
            raise RuntimeError("############################ERROR##############################\n. "
                               "please download cache file, it include training data and vocabulary & labels. "
                               "link can be found in README.md\n download zip file, unzip it, then put cache files as FLAGS."
                               "cache_file_h5py and FLAGS.cache_file_pickle suggested location.")
        print("INFO. cache file exists. going to load cache file")
        f_data = h5py.File(cache_file_h5py, 'r')
        print("f_data.keys:", list(f_data.keys()))
        train_X = f_data['train_X']  # np.array(
        print("train_X.shape:", train_X.shape)
        train_Y = f_data['train_Y']  # np.array(
        print("train_Y.shape:", train_Y.shape)
        vaild_X = f_data['vaild_X']  # np.array(
        valid_Y = f_data['valid_Y']  # np.array(
        test_X = f_data['test_X']  # np.array(
        test_Y = f_data['test_Y']  # np.array(

        word2index, label2index = None, None
        with open(cache_file_pickle, 'rb') as data_f_pickle:
            word2index, label2index = pickle.load(data_f_pickle)
        print("INFO. cache file load successful...")
        return word2index, label2index, train_X, train_Y, vaild_X, valid_Y, test_X, test_Y

    def save_data(self, cache_file_h5py, cache_file_pickle, word2index, label2index, train_X, train_Y, vaild_X, valid_Y,
                  test_X, test_Y):
        """
        存储数据为h5py格式
        :param cache_file_h5py:
        :param cache_file_pickle:
        :param word2index:
        :param label2index:
        :param train_X:
        :param train_Y:
        :param vaild_X:
        :param valid_Y:
        :param test_X:
        :param test_Y:
        :return:
        """
        # train/valid/test data using h5py
        f = h5py.File(cache_file_h5py, 'w')
        f['train_X'] = train_X
        f['train_Y'] = train_Y
        f['vaild_X'] = vaild_X
        f['valid_Y'] = valid_Y
        f['test_X'] = test_X
        f['test_Y'] = test_Y
        f.close()
        # save word2index, label2index
        with open(cache_file_pickle, 'ab') as target_file:
            pickle.dump((word2index, label2index), target_file)

    def sort_by_value(self, data_dict):
        """
        转换dict
        :param data_dict:
        :return:
        """
        _result_sort = sorted(data_dict.items(), key=lambda x: x[1], reverse=True)
        return [_result_sort[i][0] for i in range(0, len(_result_sort))]

    def _label_count(self, label_list):
        """
        统计label
        :param label_list:
        :return:
        """
        label_count = dict()
        for i in label_list:
            if label_count.get(i, None) is not None:
                label_count[i] += 1
            else:
                label_count[i] = 1
        return label_count

    def clean_title(self, text):
        """
        清洗标题
        :param text:
        :return:
        """
        text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
        text = text.lower()
        no_emoji = CleanDoc(text)._remove_emoji(text)
        del_symbol = string.punctuation  # ASCII 标点符号
        remove_punctuation_map = dict((ord(char), " ") for char in del_symbol)
        text = no_emoji.translate(remove_punctuation_map)  # 去掉ASCII 标点符号
        text = re.sub(r"\s+", " ", text)
        return text






if __name__ == '__main__':
    data_dir = "/home/zoushuai/algoproject/tf_project/data/news_category"
    input_file = os.path.join(data_dir, "top_category_corpus")
    output_file = os.path.join(data_dir, "train_corpus")
    # rewrite_train_cropus(input_file, output_file)
    SplitData2tsv(output_file, data_dir)