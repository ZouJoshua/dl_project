#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 3/23/20 12:18 AM
@File    : dataset_loader_for_multi_models_pt.py
@Desc    : 

"""



import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
from sklearn.utils import shuffle
import time
from datetime import timedelta
import logging
import json




class DatasetLoader(object):

    def __init__(self, config, logger=None):
        super(DatasetLoader, self).__init__()

        if logger:
            self.log = logger
        else:
            self.log = logging.getLogger("train_log")
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
            logging.root.setLevel(level=logging.INFO)

        self.spec_tokens = ["<PAD>", "<UNK>", "<CLS>", "<SEP>", "<MASK>", "<NUM>"]  #特殊符号

        self.config = config
        self._data_path = config.data_path
        self._word2idx_file = config.word2idx_file
        self._label2idx_file = config.label2idx_file
        self.embedding_dim = config.embedding_dim
        self.sequence_length = config.sequence_length
        self._pretrain_embedding_path = config.pretrain_embedding_file
        self.vocab_size = None
        self.word_embedding = None
        self.word2index = None
        self.label2index = None
        self.word_pkl_file = os.path.join(self._data_path, "word2index.pkl")
        self.label_pkl_file = os.path.join(self._data_path, "label2index.pkl")
        self.word_embedding_path = os.path.join(self._data_path, "word_embedding.npy")
        self.init_vocab_label()

    def init_vocab_label(self, use_word=False):
        self.log.info("*** Init vocab and label ***")
        if os.path.exists(self.word_pkl_file) and \
                os.path.exists(self.label_pkl_file):
            self.word2index, self.label2index = self.load_vocab_label(self.word_pkl_file, self.label_pkl_file)
        else:
            if use_word:
                self.tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
            else:
                self.tokenizer = lambda x: [y for y in x]  # char-level
            all_data_file = os.path.join(self._data_path, "thuc_news.all.txt")
            self.word2index, self.label2index = self.build_vocab(all_data_file, self.tokenizer)

        self.vocab_size = len(self.word2index)
        if self.config.vocab_size > len(self.word2index):
            self.vocab_size = self.config.vocab_size

        if os.path.exists(self.word_embedding_path):
            self.log.info("Load word embedding from file: {}".format(self.word_embedding_path))
            self.word_embedding = np.load(self.word_embedding_path)
        self.log.info("*** Init finished ***")


    def load_vocab_label(self, vocab_file, label_file):

        """
        加载词汇和标签的映射表
        :return:
        """
        # 将词汇-索引映射表加载出来
        self.log.info("Load word2index from file: {}".format(vocab_file))
        with open(vocab_file, "rb") as f:
            word2index = pkl.load(f)

        # 将标签-索引映射表加载出来
        self.log.info("Load label2index from file: {}".format(label_file))
        with open(label_file, "rb") as f:
            label2index = pkl.load(f)

        return word2index, label2index


    def build_vocab(self, file_path, tokenizer, max_size=-1, min_freq=1):
        self.log.info("Start dump file with word to index")
        word_count = {}
        with open(file_path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                text, label = self._get_text_and_label(json.loads(lin))
                for word in tokenizer(text):
                    word_count[word] = word_count.get(word, 0) + 1
            sort_word_count = sorted([_ for _ in word_count.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
            words = [item[0] for item in sort_word_count]
            all_words = self.spec_tokens + words
            word2index = {word: idx for idx, word in enumerate(all_words)}
            # vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
            pkl.dump(word2index, open(self.word_pkl_file, 'wb'))
        self.log.info("Dump {} words to file".format(len(word2index)))
        self.log.info("Start dump file with label to index")
        label2index = self.get_label_to_index()
        pkl.dump(label2index, open(self.label_pkl_file, "wb"))
        self.log.info("Dump down")
        return word2index, label2index



    def get_label_to_index(self):
        if os.path.exists(self._label2idx_file):
            with open(self._label2idx_file, "r", encoding="utf-8") as fr:
                return json.load(fr)
        else:
            raise FileNotFoundError

    def _get_text_and_label(self, dict_line):
        # 获取文本和标记
        text = dict_line["text"]
        label = dict_line["label"]
        return text, label

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
            lines = [eval(line) for line in tqdm(f, desc="Loading {} dataset".format(mode))]
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
        self.log.info("Read finished ***")

        return inputs, labels


    def build_dataset(self, data_file, pkl_file, mode="train"):

        self.log.info("*** Build {} dataset".format(mode))
        if not os.path.exists(pkl_file):
            self.log.info("*** Loading {} dataset from original file ***".format(mode))
            # 1.读取原始数据
            inputs, labels = self.read_data(data_file, mode)

            # 2.输入转索引
            inputs_idx = self.trans_sentences_to_index(inputs, self.word2index)
            self.log.info("Index transform finished")

            # 3.对输入做padding
            inputs_idx, inputs_len = self.padding(inputs_idx, self.sequence_length)
            self.log.info("Padding finished")

            # 4.标签转索引
            labels_idx = self.trans_labels_to_index(labels, self.label2index)
            self.log.info("Label index transform finished")

            _corpus_data = zip(inputs_idx, labels_idx, inputs_len)
            corpus_data = [(data[0], data[1], data[2]) for data in _corpus_data]
            pkl.dump(corpus_data, open(pkl_file, "wb"))
        else:
            self.log.info("Load existed {} data from pkl file: {}".format(mode, pkl_file))
            corpus_data = pkl.load(open(pkl_file, "rb"))
        self.log.info("Build finished ***")
        return corpus_data


    @staticmethod
    def trans_sentences_to_index(inputs, word_to_index):
        """
        将输入转化为索引表示
        :param inputs: 输入
        :param word_to_index: 词汇-索引映射表
        :return:
        """

        inputs_idx = [[word_to_index.get(word, word_to_index["<UNK>"]) for word in sentence] for sentence in inputs]

        return inputs_idx

    @staticmethod
    def trans_labels_to_index(labels, label_to_index):
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
        sentences_padding = []
        sentences_lens = []
        for sentence in inputs:
            seq_len = len(sentence)
            if seq_len < sequence_length:
                sentence.extend([0] * (sequence_length - seq_len))
            else:
                sentence = sentence[:sequence_length]
                seq_len = sequence_length
            sentences_padding.append(sentence)
            sentences_lens.append(seq_len)

        return sentences_padding, sentences_lens

    def build_iterator(self, dataset):
        iter = DatasetIterater(dataset, self.config.batch_size, self.config.device)
        return iter


class DatasetIterater(object):
    def __init__(self, inputs, batch_size, device):
        self.batch_size = batch_size
        self.inputs = inputs
        self.n_batches = len(inputs) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(inputs) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.inputs[self.index * self.batch_size: len(self.inputs)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.inputs[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


if __name__ == "__main__":
    '''提取预训练词向量'''
    # 下面的目录、文件名按需更改。
    train_dir = "./THUCNews/data/train.txt"
    vocab_dir = "./THUCNews/data/vocab.pkl"
    pretrain_dir = "./THUCNews/data/sgns.sogou.char"
    emb_dim = 300
    filename_trimmed_dir = "./THUCNews/data/embedding_SougouNews"
    # if os.path.exists(vocab_dir):
    #     word_to_id = pkl.load(open(vocab_dir, 'rb'))
    # else:
    #     # tokenizer = lambda x: x.split(' ')  # 以词为单位构建词表(数据集中词之间以空格隔开)
    #     tokenizer = lambda x: [y for y in x]  # 以字为单位构建词表
    #     word_to_id = build_vocab(train_dir, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
    #     pkl.dump(word_to_id, open(vocab_dir, 'wb'))
    #
    # embeddings = np.random.rand(len(word_to_id), emb_dim)
    # f = open(pretrain_dir, "r", encoding='UTF-8')
    # for i, line in enumerate(f.readlines()):
    #     # if i == 0:  # 若第一行是标题，则跳过
    #     #     continue
    #     lin = line.strip().split(" ")
    #     if lin[0] in word_to_id:
    #         idx = word_to_id[lin[0]]
    #         emb = [float(x) for x in lin[1:301]]
    #         embeddings[idx] = np.asarray(emb, dtype='float32')
    # f.close()
    # np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)

