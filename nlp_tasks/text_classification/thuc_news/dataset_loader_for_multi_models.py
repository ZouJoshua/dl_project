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
import pickle as pkl
from collections import Counter
from string import punctuation
from sklearn.utils import shuffle
import tqdm
import gensim
import jieba
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
        self._label2idx_path = config.label2idx_file
        self.embedding_dim = config.embedding_dim
        self.sequence_length = config.sequence_length
        self._pretrain_embedding_file = config.pretrain_embedding_file
        self._stopwords_file = config.stopwords_file
        self.vocab_size = config.vocab_size
        self.word_cut = True
        self.word_embedding = None
        self.word2index = None
        self.label2index = None
        if self.word_cut:
            self.word2idx_pkl_file = os.path.join(self._data_path, "word2index.pkl")
            self.embedding_file = os.path.join(self._data_path, "word_embedding.npy")
        else:
            self.word2idx_pkl_file = os.path.join(self._data_path, "char2index.pkl")
            self.embedding_file = os.path.join(self._data_path, "char_embedding.npy")

        self.label2idx_pkl_file = os.path.join(self._data_path, "label2index.pkl")

        self.init_vocab()

    def init_vocab(self):
        """
        构建词表,标签映射(我以新闻全量数据构建)
        :return:
        """
        self.log.info("*** Init vocab and label ***")
        if os.path.exists(self.word2idx_pkl_file) and \
                os.path.exists(self.label2idx_pkl_file):
            self.word2index, self.label2index = self.load_vocab()
        else:
            # 1，读取原始数据
            all_data_file = os.path.join(self._data_path, "thuc_news.all.txt")
            if self.word_cut:
                _clean_data_file = os.path.join(self._data_path, "thuc_news.word.all.txt")
            else:
                _clean_data_file = os.path.join(self._data_path, "thuc_news.char.all.txt")

            word_count = dict()
            labels = list()
            for word_list, label in self.build_clean_data(all_data_file, _clean_data_file, mode="all"):
                labels.append(label)
                for word in word_list:
                    if word in word_count:
                        word_count[word] += 1
                    else:
                        word_count[word] = 1

            # 2，得到去除低频词和停用词的词汇表
            words = self.remove_stop_word(word_count)

            # 3，得到词汇表
            self.word2index, self.label2index = self.gen_vocab(words, labels)

        # if os.path.exists(self.embedding_file):
        #     self.log.info("Load word embedding from file: {}".format(self.embedding_file))
        #     self.word_embedding = np.load(self.embedding_file)
        # elif os.path.exists(self._pretrain_embedding_file):
        #     self.word_embedding = self.get_embedding()
        # self.word_embedding = self.get_word_embedding(words)

        self.log.info("*** Init finished ***")

    def build_clean_data(self, data_file, clean_data_file, mode=""):
        self.log.info("*** Build {} clean dataset ***".format(mode))
        if os.path.exists(clean_data_file):
            self.log.info("Loading {} dataset from clean data file".format(mode))
            return self.read_data(clean_data_file, mode=mode)
        else:
            self.log.info("Loading {} dataset from original data file".format(mode))

            f = open(clean_data_file, "w", encoding="utf-8")
            for text, label in self.read_data(data_file, mode=mode):
                line = dict()
                if self.word_cut:
                    word_list = self.split_sentence_by_jieba(text)
                else:
                    word_list = self.split_sentence_by_char(text)
                if word_list:
                    line["text"] = word_list
                    line["label"] = label
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")
                    f.flush()
                    yield word_list, label
                else:
                    self.log.warning("Split error: {}".format(text))
            f.close()
        self.log.info("*** Build clean dataset finished ***")


    def load_vocab(self):

        """
        加载词汇和标签的映射表
        :return:
        """
        # 将词汇-索引映射表加载出来
        self.log.info("Load word2index from file: {}".format(self.word2idx_pkl_file))
        with open(self.word2idx_pkl_file, "rb") as f:
            word2index = pkl.load(f)

        # 将标签-索引映射表加载出来
        self.log.info("Load label2index from file: {}".format(self.label2idx_pkl_file))
        with open(self.label2idx_pkl_file, "rb") as f:
            label2index = pkl.load(f)

        self.vocab_size = len(word2index)

        return word2index, label2index

    def gen_vocab(self, words, labels):
        """
        生成词汇，标签等映射表
        :param words: 训练集所含有的单词
        :param labels: 标签
        :return:
        """
        self.log.info("Generate mapping tables for vocabulary, labels, etc.")

        spec_tokens = ["<PAD>", "<UNK>", "<CLS>", "<SEP>", "<MASK>", "<NUM>"]
        words = spec_tokens + words

        # 若vocab的长读小于设置的vocab_size，则选择vocab的长度作为真实的vocab_size
        self.vocab_size = len(words)
        self.log.info("Vocab size: {}".format(self.vocab_size))

        vocab = words[:self.vocab_size]

        # if self._pretrain_embedding_file:
        #     word_embedding = self.get_word_embedding(vocab)

        word2index = dict(zip(vocab, list(range(len(vocab)))))
        # word2index = {word: idx for idx, word in enumerate(vocab)}

        # 将词汇-索引映射表保存为pkl数据，之后做inference时直接加载来处理数据
        pkl.dump(word2index, open(self.word2idx_pkl_file, "wb"))

        # 将标签-索引映射表保存为pkl数据
        # unique_labels = list(set(labels))
        # label2index = dict(zip(unique_labels, list(range(len(unique_labels)))))
        label2index = self.get_label_to_index()
        pkl.dump(label2index, open(self.label2idx_pkl_file, "wb"))

        self.log.info("Vocab process finished")

        return word2index, label2index

    def remove_stop_word(self, inputs):
        """
        去除低频词和停用词
        :param inputs: 输入
        :return:
        """
        # all_words = [word for data in inputs for word in data]
        # word_count = Counter(all_words)  # 统计词频
        self.log.info("Removing low frequency words and stop words")

        sort_word_count = sorted(inputs.items(), key=lambda x: x[1], reverse=True)

        # 去除低频词
        words = [item[0] for item in sort_word_count]

        # 如果传入了停用词表，则去除停用词
        if self._stopwords_file:
            with open(self._stopwords_file, "r", encoding="utf-8") as fr:
                stop_words = [line.strip() for line in fr.readlines()]
            words = [word for word in words if word not in stop_words]
        self.log.info("Word process finished")

        return words


    def read_data(self, file, mode=""):
        """
        读取数据
        :return: 返回分词后的文本内容和标签，inputs = [[]], labels = []
        """
        self.log.info("Read data from file:{}".format(file))
        with open(file, "r", encoding="utf-8") as f:
            # 将数据集全部加载到内存
            for i, _line in enumerate(tqdm.tqdm(f, desc="Loading {} dataset".format(mode))):
                if _line:
                    line = eval(_line)
                    try:
                        text, label = self._get_text_and_label(line)
                        yield text, label
                    except:
                        self.log.warning("Error with line {}: {}".format(i, line))
                        continue
        self.log.info("Read finished")


    def _get_text_and_label(self, dict_line):
        # 获取文本和标记
        text = dict_line["text"]
        label = dict_line["label"]
        return text, label

    def split_sentence_by_char(self, text):
        """
        按字切分句子,去除非中文字符及标点
        :param text:
        :return:
        """
        # print("splitting chinese char")
        seg_list = list()
        none_chinese = ""
        for char in text:
            if self.is_chinese(char) is False:
                if char in self.punc_list:
                    continue
                none_chinese += char
            else:
                if none_chinese:
                    seg_list.append(none_chinese)
                    none_chinese = ""
                seg_list.append(char)
        if not seg_list:
            seg_list = None
        return seg_list

    def split_sentence_by_jieba(self, text):
        """
        按结巴分词.去除标点
        :param text:
        :return:
        """
        seg_list = jieba.cut(text, cut_all=False)
        words = [w for w in seg_list if w not in self.punc_list]
        if not words:
            words = None

        return words

    def is_chinese(self, uchar):
        """判断一个unicode是否是汉字"""
        if (uchar >= u'\u4e00') and (uchar <= u'\u9fa5'):
            return True
        else:
            return False

    @property
    def punc_list(self):
        add_punc = '，。、【 】 “”：；（）《》‘’{}？！⑦()、%^>℃：”“^-——=&#@￥\n「」…『』\u3000\xa0'
        return punctuation + add_punc


    def get_embedding(self):
        """
        提取预训练词向量
        :return:
        """
        self.log.info("Load embedding from pre-training file: {}".format(self._pretrain_embedding_file))
        word_embedding = (1 / np.sqrt(self.vocab_size) * (2 * np.random.rand(self.vocab_size, self.embedding_dim) - 1))

        # word_embedding = np.random.rand(self.vocab_size, self.config.embedding_dim)
        f = open(self._pretrain_embedding_file, "r", encoding='UTF-8')
        for i, line in enumerate(f.readlines()):
            if i == 0:  # 若第一行是标题，则跳过
                continue
            lin = line.strip().split(" ")
            if lin[0] in self.word2index:
                idx = self.word2index[lin[0]]
                emb = [float(x) for x in lin[1:self.config.embedding_dim+1]]
                word_embedding[idx] = np.asarray(emb, dtype='float32')
        f.close()
        # 将本项目的词向量保存起来
        np.save(self.embedding_file, word_embedding)
        # np.savez_compressed(self.word_embedding_file, embeddings=word_embedding)
        self.log.info("Load embedding finished")
        return word_embedding


    def get_word_embedding(self, vocab):
        """
        加载词向量，并获得相应的词向量矩阵
        :param vocab: 训练集所含有的单词
        :return:
        """
        self.log.info("Load embedding from pre-training file: {}".format(self._pretrain_embedding_file))
        word_embedding = (1 / np.sqrt(len(vocab)) * (2 * np.random.rand(len(vocab), self.embedding_dim) - 1))
        if os.path.splitext(self._pretrain_embedding_file)[-1] == ".bin":
            word_vec = gensim.models.KeyedVectors.load_word2vec_format(self._pretrain_embedding_file, binary=True)
        else:
            word_vec = gensim.models.KeyedVectors.load_word2vec_format(self._pretrain_embedding_file)

        for i in range(len(vocab)):
            try:
                vector = word_vec.wv[vocab[i]]
                word_embedding[i, :] = vector
            except:
                self.log.warning(vocab[i] + "不存在于字向量中")
        # 将本项目的词向量保存起来
        np.save(self.embedding_file, word_embedding)
        self.log.info("Load embedding finished")
        return word_embedding


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
        self.log.info("*** Build {} dataset ***".format(mode))
        if not os.path.exists(pkl_file):

            # 1.读取原始数据

            if self.word_cut:
                _clean_data_file = os.path.join(os.path.split(file_path)[0], "thuc_news.word.{}.txt".format(mode))
            else:
                _clean_data_file = os.path.join(os.path.split(file_path)[0], "thuc_news.char.{}.txt".format(mode))

            inputs = list()
            labels = list()

            for word_list, label in self.build_clean_data(file_path, _clean_data_file, mode=mode):
                inputs.append(word_list)
                labels.append(label)

            # 2.输入转索引
            inputs_idx = self.trans_to_index(inputs, self.word2index)
            self.log.info("Index transform finished")
            self.log.info("Input example:\n{}".format(inputs_idx[:2]))

            # 3.对输入做padding
            inputs_idx = self.padding(inputs_idx, self.sequence_length)
            self.log.info("Padding finished")
            self.log.info("Padding input example:\n{}".format(inputs_idx[:2]))

            # 4.标签转索引
            labels_idx = self.trans_label_to_index(labels, self.label2index)
            self.log.info("Label index transform finished")
            self.log.info("Label example:\n{}".format(labels_idx[:2]))

            corpus_data = dict(inputs_idx=inputs_idx, labels_idx=labels_idx)
            shuffle(corpus_data)
            with open(pkl_file, "wb") as fw:
                pkl.dump(corpus_data, fw)

        else:
            self.log.info("Load existed {} data from pkl file: {}".format(mode, pkl_file))
            with open(pkl_file, "rb") as f:
                corpus_data = pkl.load(f)
                inputs_idx = corpus_data["inputs_idx"]
                labels_idx = corpus_data["labels_idx"]

        self.log.info("*** Convert examples to features finished ***")
        return np.array(inputs_idx), np.array(labels_idx)



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