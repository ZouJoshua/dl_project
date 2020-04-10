#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 3/23/20 12:18 AM
@File    : dataset_loader_for_multi_models.py
@Desc    : 

"""


import os
import json
import pickle
from collections import Counter
from sklearn.utils import shuffle
import tqdm
import gensim
import numpy as np
import logging

from model_tensorflow.basic_data import DataBase




class DatasetLoader(DataBase):

    def __init__(self, config, logger=None):
        super(DatasetLoader, self).__init__(config)

        if logger:
            self.log = logger
        else:
            self.log = logging.getLogger("train_log")
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
            logging.root.setLevel(level=logging.INFO)

        self._data_path = config.data_path
        self._label2idx_path = config.label2idx_path
        self.embedding_dim = config.embedding_dim
        self.sequence_length = config.sequence_length
        self._pretrain_embedding_path = config.pretrain_embedding
        self.vocab_size = config.vocab_size
        self.word_embedding = None
        self.word2index = None
        self.label2index = None
        self.word2idx_pkl_file = os.path.join(self._data_path, "word2index.pkl")
        self.label2idx_pkl_file = os.path.join(self._data_path, "label2index.pkl")
        self.word_embedding_path = os.path.join(self._data_path, "word_embedding.npy")
        self.init_vocab()



    def read_data(self, file, mode=""):
        """
        读取数据
        :return: 返回分词后的文本内容和标签，inputs = [[]], labels = []
        """
        self.log.info("*** Read data from file:{}".format(file))
        inputs = []
        labels = []
        with open(file, "r", encoding="utf-8") as f:
            # 将数据集全部加载到内存
            lines = [eval(line) for line in tqdm.tqdm(f, desc="Loading {} dataset".format(mode))]
            # 打乱顺序
            lines = shuffle(lines)
            # 获取数据长度(条数)
            # corpus_lines = len(lines)
            for i, line in enumerate(lines):
                try:
                    text, label = self._get_text_and_label(line)
                    inputs.append(text)
                    labels.append(label)
                except:
                    self.log.warning("Error with line {}: {}".format(i, line))
                    continue
        self.log.info("Read finished")

        return inputs, labels

    def _get_text_and_label(self, dict_line):
        # 获取文本和标记
        text = dict_line["text"]
        label = dict_line["label"]
        return text, label

    def remove_stop_word(self, inputs):
        """
        去除低频词和停用词
        :param inputs: 输入
        :return:
        """
        # all_words = [word for data in inputs for word in data]
        # word_count = Counter(all_words)  # 统计词频
        self.log.info("*** Removing low frequency words and stop words")
        word_count = dict()
        for data in inputs:
            for word in data:
                if word in word_count:
                    word_count[word] += 1
                else:
                    word_count[word] = 1

        sort_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)

        # 去除低频词
        words = [item[0] for item in sort_word_count]

        # 如果传入了停用词表，则去除停用词
        if self.config.stopwords_path:
            with open(self.config.stopwords_path, "r", encoding="utf-8") as fr:
                stop_words = [line.strip() for line in fr.readlines()]
            words = [word for word in words if word not in stop_words]
        self.log.info("Word process finished")

        return words

    def get_word_embedding(self, vocab):
        """
        加载词向量，并获得相应的词向量矩阵
        :param vocab: 训练集所含有的单词
        :return:
        """
        self.log.info("*** Load embedding from pre-training file: {}".format(self._pretrain_embedding_path))
        word_embedding = (1 / np.sqrt(len(vocab)) * (2 * np.random.rand(len(vocab), self.embedding_dim) - 1))
        if os.path.splitext(self._pretrain_embedding_path)[-1] == ".bin":
            word_vec = gensim.models.KeyedVectors.load_word2vec_format(self._pretrain_embedding_path, binary=True)
        else:
            word_vec = gensim.models.KeyedVectors.load_word2vec_format(self._pretrain_embedding_path)

        for i in range(len(vocab)):
            try:
                vector = word_vec.wv[vocab[i]]
                word_embedding[i, :] = vector
            except:
                self.log.warning(vocab[i] + "不存在于字向量中")
        self.log.info("Load finished")
        return word_embedding

    def init_vocab(self):

        if os.path.exists(self.word2idx_pkl_file) and \
                os.path.exists(self.label2idx_pkl_file):
            self.word2index, self.label2index = self.load_vocab()
        else:
            all_data_file = os.path.join(self._data_path, "thuc_news.all.txt")
            # 1，读取原始数据
            inputs, labels = self.read_data(all_data_file, mode="all")

            # 2，得到去除低频词和停用词的词汇表
            words = self.remove_stop_word(inputs)

            # 3，得到词汇表
            self.word2index, self.label2index = self.gen_vocab(words, labels)

        if os.path.exists(self.word_embedding_path):
            self.log.info("Load word embedding from file: {}".format(self.word_embedding_path))
            self.word_embedding = np.load(self.word_embedding_path)
        elif os.path.exists(self._pretrain_embedding_path):
            self.word_embedding = self.get_word_embedding()


    def gen_vocab(self, words, labels):
        """
        生成词汇，标签等映射表
        :param words: 训练集所含有的单词
        :param labels: 标签
        :return:
        """
        self.log.info("*** Generate mapping tables for vocabulary, labels, etc.")

        spec_tokens = ["<PAD>", "<UNK>", "<CLS>", "<SEP>", "<MASK>", "<NUM>"]
        words = spec_tokens + words
        vocab = words[:self.vocab_size]

        # 若vocab的长读小于设置的vocab_size，则选择vocab的长度作为真实的vocab_size
        self.vocab_size = len(vocab)

        if self._pretrain_embedding_path:
            word_embedding = self.get_word_embedding(vocab)
            # 将本项目的词向量保存起来
            np.save(self.word_embedding_path, word_embedding)

        word2index = dict(zip(vocab, list(range(len(vocab)))))

        # 将词汇-索引映射表保存为pkl数据，之后做inference时直接加载来处理数据
        with open(self.word2idx_pkl_file, "wb") as f:
            pickle.dump(word2index, f)

        # 将标签-索引映射表保存为pkl数据
        # unique_labels = list(set(labels))
        # label2index = dict(zip(unique_labels, list(range(len(unique_labels)))))
        label2index = self.get_label_to_index()
        with open(self.label2idx_pkl_file, "wb") as f:
            pickle.dump(label2index, f)
        self.log.info("Vocab process finished")

        return word2index, label2index


    def load_vocab(self):

        """
        加载词汇和标签的映射表
        :return:
        """
        # 将词汇-索引映射表加载出来
        self.log.info("Load word2index from file: {}".format(self.word2idx_pkl_file))
        with open(self.word2idx_pkl_file, "rb") as f:
            word2index = pickle.load(f)

        # 将标签-索引映射表加载出来
        self.log.info("Load label2index from file: {}".format(self.label2idx_pkl_file))
        with open(self.label2idx_pkl_file, "rb") as f:
            label2index = pickle.load(f)

        self.vocab_size = len(word2index)

        return word2index, label2index


    def get_label_to_index(self):
        if os.path.exists(self._label2idx_path):
            with open(self._label2idx_path, "r", encoding="utf-8") as fr:
                return json.load(fr)
        else:
            raise FileNotFoundError

    @staticmethod
    def trans_to_index(inputs, word_to_index):
        """
        将输入转化为索引表示
        :param inputs: 输入
        :param word_to_index: 词汇-索引映射表
        :return:
        """

        inputs_idx = [[word_to_index.get(word, word_to_index["<UNK>"]) for word in sentence] for sentence in inputs]

        return inputs_idx

    @staticmethod
    def trans_label_to_index(labels, label_to_index):
        """
        将标签也转换成数字表示
        :param labels: 标签
        :param label_to_index: 标签-索引映射表
        :return:
        """
        labels_idx = [label_to_index[label] for label in labels]
        return labels_idx

    def padding(self, inputs, sequence_length):
        """
        对序列进行截断和补全
        :param inputs: 输入
        :param sequence_length: 预定义的序列长度
        :return:
        """
        new_inputs = [sentence[:sequence_length]
                      if len(sentence) > sequence_length
                      else sentence + [0] * (sequence_length - len(sentence))
                      for sentence in inputs]

        return new_inputs



    def convert_examples_to_features(self, file_path, pkl_file, mode):

        if not os.path.exists(pkl_file):
            self.log.info("*** Loading {} dataset from original file ***".format(mode))
            # 1.读取原始数据
            inputs, labels = self.read_data(file_path, mode)

            # 2.输入转索引
            inputs_idx = self.trans_to_index(inputs, self.word2index)
            self.log.info("Index transform finished")

            # 3.对输入做padding
            inputs_idx = self.padding(inputs_idx, self.sequence_length)
            self.log.info("Padding finished")

            # 4.标签转索引
            labels_idx = self.trans_label_to_index(labels, self.label2index)
            self.log.info("Label index transform finished")

            corpus_data = dict(inputs_idx=inputs_idx, labels_idx=labels_idx)
            with open(pkl_file, "wb") as fw:
                pickle.dump(corpus_data, fw)

            return np.array(inputs_idx), np.array(labels_idx)
        else:
            self.log.info("Load existed {} data from pkl file: {}".format(mode, pkl_file))
            with open(pkl_file, "rb") as f:
                corpus_data = pickle.load(f)
            return np.array(corpus_data["inputs_idx"]), np.array(corpus_data["labels_idx"])



    def next_batch(self, x, y, batch_size):
        """
        生成batch数据集
        :param x: 输入
        :param y: 标签
        :param batch_size: 批量的大小
        :return:
        """
        perm = np.arange(len(x))
        np.random.shuffle(perm)
        x = x[perm]
        y = y[perm]

        num_batches = len(x) // batch_size

        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            batch_x = np.array(x[start: end], dtype="int64")
            batch_y = np.array(y[start: end], dtype="float32")

            yield dict(x=batch_x, y=batch_y)