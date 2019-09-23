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
import numpy as np

import h5py
from sklearn.model_selection import StratifiedKFold
from gensim.models import KeyedVectors
from preprocess.preprocess_tools import read_json_format_file, split_text, get_ngrams, write_json_format_file


class DataSet(object):

    def __init__(self, data_dir, word2vec_file, training_data_file=None, embedding_dims=300):
        """
        数据预处理
        :param data_dir: 数据资源目录
        :param word2vec_file: 预训练词向量文件
        :param training_data_file: 原始数据文件
        :param embedding_dims: embedding词向量维度
        """
        self.data_dir = data_dir
        self.word2vec_path = word2vec_file
        if training_data_file:
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
            print("building vocabulary from cache file: {}".format(cache_path))
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
        cache_path = os.path.join(self.data_dir, "label_vocabulary.pik")
        print("cache_path:", cache_path, "file_exists:", os.path.exists(cache_path))

        if os.path.exists(cache_path):  # 如果缓存文件存在，则直接读取
            print("building vocabulary_label from cache file...")
            with open(cache_path, 'rb') as data_f:
                _label2idx, _idx2label = pickle.load(data_f)
                return _label2idx, _idx2label
        else:
            print("building vocabulary_label_sorted. training_data:", self.raw_data_path)
            _label2idx = dict()
            _idx2label = dict()
            label_count_dict = dict()  # {label:count} 统计各类别的样本数
            lines = read_json_format_file(self.raw_data_path)
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
        lines = read_json_format_file(self.raw_data_path)
        # 2.transform X as indices
        """
        #todo: 去掉停用词 -> 统计词频 -> 去除低频词
        """
        # 3.transform  y as scalar
        X = []
        Y = []
        Y_decoder_input = []
        count_not_exist = 0
        if use_embedding:
            self.embedding = self.get_embedding(self.word2embed)

        for i, line in enumerate(lines):
            title = line["title"].strip().replace("\t", " ").replace("\n", " ").replace("\r", " ")
            content = line["content"].strip()
            x = title + " " + content
            y = line["top_category"]
            # 打印前几条
            if i < 2:
                print("x{}:".format(i), x)  # get raw x
            x = split_text(x)
            x = [self.word2index.get(w, 0) for w in x]  # 若找不到单词，用0填充
            if i < 2:
                print("x{}-word-index:".format(i), x)  # word to index
            y = self.label2index[y]
            X.append(x)
            Y.append(y)

        # 4.split to train,test and valid data(基于y标签分层)
        doc_num = len(X)
        print("number_doc:", doc_num)
        train, test = self.stratified_sampling(X, Y, lines, valid_portion)
        print("load_data ended...")
        return train, test, test

    def load_data_predict(self, predict_data_path, word2index, n_gram=False):
        # 1. load raw data
        print("load_data_predict.started...")
        if predict_data_path:
            predict_file = predict_data_path
        else:
            predict_file = self.corpus_predict_file
        print("load_data.predict_data_path:", predict_file)
        lines = read_json_format_file(predict_file)
        # 2.transform X as indices
        """
        #todo: 去掉停用词 -> 统计词频 -> 去除低频词
        """
        # 3.transform  y as scalar
        predict_x = []

        for i, line in enumerate(lines):
            title = line["title"].strip().replace("\t", " ").replace("\n", " ").replace("\r", " ")
            content = line["content"].strip()
            doc_id = line["id"]
            x = title + " " + content
            # 打印前几条
            if i < 2:
                print("x{}:".format(i), x)  # get raw x
            if n_gram:
                x_ = get_ngrams(x)
                x = split_text(x_)
            else:
                x = split_text(x)
            x = [word2index.get(w, 0) for w in x]  # 若找不到单词，用0填充
            if i < 2:
                print("x{}-word-index:".format(i), x)  # word to index
            predict_x.append((doc_id, x))
        number_examples = len(predict_x)
        print("number_examples:", number_examples)  #
        return predict_x


    def get_embedding(self, word2embed):
        vocab_size = len(word2embed)
        unk = word2embed.get("UNK", np.random.randn(self.embed_dim))
        word_embedding_list = [[]] * vocab_size  # create an empty word_embedding list.
        for i, vocab in enumerate(word2embed):
            word_embedding_list[i] = word2embed.get(vocab, unk)
        word_embedding = np.array(word_embedding_list)
        return word_embedding

    def stratified_sampling(self, x, y, all_data, valid_portion=0.2):
        self.corpus_predict_file = os.path.join(self.data_dir, "corpus_predict")
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
                print("train_label_count: {}".format(train_label_count))
                print("test_label_count: {}".format(test_label_count))
                trainx = [x[i] for i in train_index]
                trainy = [y[i] for i in train_index]
                testx = [x[j] for j in test_index]
                testy = [y[j] for j in test_index]
                corpus_pred = [all_data[i] for i in test_index]
                write_json_format_file(corpus_pred, self.corpus_predict_file)
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
        _result_sort = sorted(data_dict.items(), key=lambda x: x[1], reverse=True)
        return [_result_sort[i][0] for i in range(0, len(_result_sort))]

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
    # 处理语料
    # data_base_dir = r'/data/in_hi_news/train_data'
    data_base_dir = r"/home/joshua/tensorflowProject/tensorflow_project/data/hi_news"
    file1 = os.path.join(data_base_dir, )
    file2 = os.path.join(data_base_dir, 'auto_science_tech')
    out_file = os.path.join(data_base_dir, 'top_category_corpus')
    TopcategoryCorpus(file1, file2, out_file)

    # 加载数据
    # data_dir = "/home/zoushuai/algoproject/tf_project/data/hi_news"
    # training_data_file = os.path.join(data_dir, "top_category_corpus")
    # word2vec_file = os.path.join(data_dir, "word2vec.bin")
    # ds = DataSet(data_dir, word2vec_file, training_data_file)
    # train, test, _ = ds.load_data(use_embedding=False, valid_portion=0.2)
    # trainx, trainy = train
    # testx, testy = test
    # print(trainx[:1])
    # print(trainy[:1])

if __name__ == "__main__":
    main()


