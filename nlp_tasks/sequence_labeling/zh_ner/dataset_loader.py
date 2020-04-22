#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 4/20/20 9:57 PM
@File    : dataset_loader.py
@Desc    : 

"""

import os
import numpy as np
import pickle as pkl, tqdm
from model_tensorflow.basic_data import DataBase
from sklearn.utils import shuffle
import random
import json
import logging
from nlp_tasks.sequence_labeling.zh_ner.preprocess_data import update_tag_scheme
import jieba

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
        self.vocab_size = config.vocab_size
        self.tag_scheme = config.tag_scheme
        self.word_cut = True
        self.word_embedding = None
        self.word2index = None
        self.label2index = None

        self.word2idx_pkl_file = os.path.join(self._data_path, "char2index.pkl")
        self.embedding_file = os.path.join(self._data_path, "char_embedding.npy")
        self.label2idx_pkl_file = os.path.join(self._data_path, "label2index.pkl")

        self.init_vocab()

    def init_vocab(self):
        """
        初始化词表,标签映射id
        构建词表,标签映射(我以新闻全量数据构建)
        :return:
        """
        self.log.info("*** Init vocab and label ***")
        if os.path.exists(self.word2idx_pkl_file) and \
                os.path.exists(self.label2idx_pkl_file):
            self.word2index, self.label2index = self.load_vocab()
        else:
            self.word2index, self.label2index = self.gen_vocab()


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

    def gen_vocab(self):
        self.log.info("Generate mapping tables for vocabulary, labels, etc.")
        words, labels = self._build_vocab()

        spec_tokens = [self._pad_token, self._unk_token, self._num_token]
        words = spec_tokens + words

        # 若vocab的长读小于设置的vocab_size，则选择vocab的长度作为真实的vocab_size
        self.vocab_size = len(words)
        # self.log.info("Vocab size: {}".format(self.vocab_size))

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

    def read_data(self, file, mode="origin"):
        """
        读取数据
        加载数据集,每行一个汉子和一个标记,句子和句子之间以空格分割
        :return: 返回句子集合
        """
        self.log.info("Read data from file:{}".format(file))
        data = []
        sent_, tag_ = [], []
        with open(file, "r", encoding="utf-8") as f:
            # 将数据集全部加载到内存
            for i, _line in enumerate(tqdm.tqdm(f, desc="Loading {} dataset".format(mode))):
                if _line:
                    if _line != '\n':
                        [char, label] = _line.strip().split()
                        sent_.append(char)
                        tag_.append(label)
                    else:
                        data.append((sent_, tag_))
                        sent_, tag_ = [], []
                else:
                    self.log.warning("Error with line {}: {}".format(i, _line))
                    continue
        self.log.info("Read finished")
        return data

    def remove_stop_word(self, inputs):
        """
        去除低频词和停用词
        :param inputs: 字符统计字典 ex.{"我":121,...}
        :return:
        """
        # all_words = [word for data in inputs for word in data]
        # word_count = Counter(all_words)  # 统计词频
        self.log.info("Removing low frequency words and stop words")

        sort_word_count = sorted(inputs.items(), key=lambda x: x[1], reverse=True)

        # 去除低频词
        words = [item[0] for item in sort_word_count if item[1] >= 1]

        self.log.info("Word process finished")

        return words


    def get_label_to_index(self):
        """
        从文件读取标签映射文件
        :return:
        """
        if os.path.exists(self._label2idx_path):
            with open(self._label2idx_path, "r", encoding="utf-8") as fr:
                return json.load(fr)
        else:
            raise FileNotFoundError


    def _build_vocab(self):
        """
        从原始文件构建words,labels
        :return:
        """
        # 1，读取原始数据(训练数据\测试数据)
        file_name = ["train", "test"]
        word_count = dict()
        labels = None
        for name in file_name:
            file = os.path.join(self._data_path, "{}_source.txt".format(name))
            if not os.path.exists(file):
                raise FileNotFoundError("File {} not found".format(file))

            with open(file, "r", encoding="utf-8") as f:
                for i, _line in enumerate(tqdm.tqdm(f, desc="Loading data from {}".format(file))):
                    if _line:
                        chars = _line.strip().split()
                        for char in chars:
                            # char = self._replace_char_with_special_token(char)
                            if char in word_count:
                                word_count[char] += 1
                            else:
                                word_count[char] = 1

        # 2，得到去除低频词和停用词的词汇表
        words = self.remove_stop_word(word_count)
        return words, labels

    def _replace_char_with_special_token(self, word):
        """
        replace char with special token
        :param word:
        :return:
        """
        if word.isdigit():
            # Inspecting the str whether combine by number
            word = self._num_token
        elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
            # Judging the english
            word = self._en_token
        else:
            word = word

        return word


    def sentence2id(self, sentence):
        sentence_id = []
        for word in sentence:
            word = self._replace_char_with_special_token(word)
            if word not in self.word2index:
                # Chinese
                word = self._unk_token
            sentence_id.append(self.word2index[word])

        return sentence_id

    def trans_label_to_index(self, labels):
        """
        将标签也转换成数字表示
        :param labels: 标签
        :return:
        """
        labels_idx = [self.label2index[label] for label in labels]
        # !!注意pad时索引采用"0"表示单个字成词,我的word2id中的"<PAD>"的索引为0,为方便直接用padding方法
        labels_pad_id, _ = self.padding(labels_idx, self.sequence_length)
        return labels_pad_id

    def get_seg_features(self, words):
        """
        利用jieba分词
        采用类似bioes的编码，0表示单个字成词, 1表示一个词的开始， 2表示一个词的中间，3表示一个词的结尾
        :param words:
        :return:
        """
        seg_features = []

        word_list = list(jieba.cut(words))

        for word in word_list:
            if len(word) == 1:
                seg_features.append(0)
            else:
                temp = [2] * len(word)
                temp[0] = 1
                temp[-1] = 3
                seg_features.extend(temp)
        # !!注意pad时索引采用0表示bio标注格式中"O"的索引,我的word2id中的"<PAD>"的索引为0,为方便直接用padding方法
        seg_features_id, _ = self.padding(seg_features, self.sequence_length)
        return seg_features_id


    def padding(self, inputs, sequence_length):
        """
        对序列进行截断和补全
        :param inputs: 输入
        :param sequence_length: 预定义的序列长度
        :return:
        """
        # new_inputs = [sentence[:sequence_length]
        #               if len(sentence) > sequence_length
        #               else sentence + [0] * (sequence_length - len(sentence))
        #               for sentence in inputs]
        inputs_len = len(inputs)
        if inputs_len > sequence_length:
            inputs = inputs[:sequence_length]
        else:
            inputs = inputs + [self.word2index.get(self._pad_token)] * (sequence_length - inputs_len)

        return inputs, inputs_len

    def random_embedding(self, vocab, embedding_dim):
        embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
        embedding_mat = np.float32(embedding_mat)
        return embedding_mat


    def convert_examples_to_features(self, file_path, pkl_file, mode):
        self.log.info("*** Build {} dataset ***".format(mode))
        if not os.path.exists(pkl_file):

            word_listx = list()
            inputs_idx = list()
            labels_idx = list()
            segments_idx = list()

            i = 0
            data = self.read_data(file_path, mode=mode)
            # update_tag_scheme(data, self.tag_scheme)

            for word_list, label in data:
                i += 1
                if not word_list:
                    continue

                # 2.输入转索引并做padding
                input = self.sentence2id(word_list)
                words, _ = self.padding(word_list, self.sequence_length)
                input_id, input_len = self.padding(input, self.sequence_length)

                if self.word_cut:
                    # 增加jieba分词特征并做padding
                    segments_id = self.get_seg_features("".join(word_list))
                else:
                    segments_id = []
                # 3.标注序列转索引
                label_id = self.trans_label_to_index(label)

                if i < 2:
                    self.log.info("*** {} example {} ***".format(mode, i))
                    self.log.info("Input example: {}".format(word_list))
                    self.log.info("Input index example: {}".format(input))
                    self.log.info("Padding input example: {}".format(input_id))
                    self.log.info("Word segment example: {}".format(segments_id))
                    self.log.info("Label example: {}".format(label))
                    self.log.info("Label index example: {}".format(label_id))


                word_listx.append(words)
                inputs_idx.append(input_id)
                labels_idx.append(label_id)
                segments_idx.append(segments_id)

            corpus_data = dict(word_listx=word_listx, inputs_idx=inputs_idx, labels_idx=labels_idx, segments_idx=segments_idx)
            pkl.dump(corpus_data, open(pkl_file, "wb"))

        else:
            self.log.info("Load existed {} data from pkl file: {}".format(mode, pkl_file))

            corpus_data = pkl.load(open(pkl_file, "rb"))
            word_listx = corpus_data["word_listx"]
            inputs_idx = corpus_data["inputs_idx"]
            labels_idx = corpus_data["labels_idx"]
            segments_idx = corpus_data["segments_idx"]

        self.log.info("*** Convert examples to features finished ***")
        return (np.array(word_listx), np.array(inputs_idx), np.array(labels_idx)), np.array(segments_idx)



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
        x_word_list = x[0][perm]
        x_input1 = x[1][perm]
        x_input2 = x[2][perm]
        y = y[perm]

        num_batches = len(x_input1) // batch_size

        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            batch_x = x_word_list[start: end]
            batch_x1 = x_input1[start: end]
            batch_x2 = x_input2[start: end]
            batch_y = y[start: end]
            yield dict(word_list=batch_x, char_input=batch_x1, seg_input=batch_x2, y=batch_y)

