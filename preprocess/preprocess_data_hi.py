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

from gensim.models import KeyedVectors



class DataSet(object):

    def __init__(self, data_dir, word2vec_path):
        self.data_dir = data_dir
        self._word2index = dict()
        self._index2word = dict()


    def create_vocabulary(self, word2vec_model_path):
        """
        创建词汇索引表
        :param word2vec_model_path: 训练好的word2vec模型存放路径
        :return: {单词：索引}表和{索引：单词}表
        """

        cache_path = os.path.join(self.data_dir + "word_vocabulary.pik")
        print("cache_path:", cache_path, "file_exists:", os.path.exists(cache_path))

        if os.path.exists(cache_path):  # 如果缓存文件存在，则直接读取
            with open(cache_path, 'rb') as data_f:
                vocabulary_word2idx, vocabulary_idx2word = pickle.load(data_f)
                return vocabulary_word2idx, vocabulary_idx2word
        else:


            print("building vocabulary（words with frequency above 5 are included). word2vec_path:", word2vec_model_path)

            # 添加 "pad" 和 "UNK",
            vocab.append("pad")
            vocab.append("UNK")
            wordEmbedding.append(np.zeros(self._embeddingSize))
            wordEmbedding.append(np.random.randn(self._embeddingSize))
            self._word2index['PAD_ID'] = 0
            self._index2word[0] = 'PAD_ID'
            special_index = 0

            # model = Word2Vec.load(word2vec_model_path)
            model = KeyedVectors.load_word2vec_format(word2vec_model_path, binary=True)

            for i, vocab in enumerate(model.wv.vocab):
                    vocabulary_word2idx[vocab] = i + 1 + special_index  # 只设了一个special ID
                    vocabulary_idx2word[i + 1 + special_index] = vocab

            # 如果不存在写到缓存文件中
            if not os.path.exists(cache_path):
                with open(cache_path, 'ab') as data_f:
                    pickle.dump((vocabulary_word2idx, vocabulary_idx2word), data_f)
        return vocabulary_word2idx, vocabulary_idx2word

    def create_label_vocabulary(self, training_data_dir_path='/data/caifuli/news_classification/data', name_scope=''):
        """
        创建标签映射  label is sorted. 1 is high frequency, 2 is low frequency.
        :param training_data_path: 带label的训练语料
        :return: label2idx和idx2label
        """
        print("building vocabulary_label_sorted. training_data_dir__path:", training_data_dir_path)
        cache_path = '/data/caifuli/news_classification/textcnn/cache_vocabulary_label_pik/' + name_scope + "_label_vocabulary.pik"
        print("cache_path:", cache_path, "file_exists:", os.path.exists(cache_path))

        if os.path.exists(cache_path):  # 如果缓存文件存在，则直接读取
            with open(cache_path, 'rb') as data_f:
                vocabulary_word2index_label, vocabulary_index2word_label = pickle.load(data_f)
                return vocabulary_word2index_label, vocabulary_index2word_label
        else:
            label2idx = {}
            idx2label = {}
            label_count_dict = {}  # {label:count} 统计各类别的样本数
            fnames = os.listdir(training_data_dir_path)
            for fname in fnames:
                with open(os.path.join(training_data_dir_path, fname), "r") as f:
                    for line in f.readlines():
                        line = json.loads(line)
                        label = line['category']
                        if label_count_dict.get(label, None) is not None:
                            label_count_dict[label] = label_count_dict[label] + 1
                        else:
                            label_count_dict[label] = 1

            list_label = sort_by_value(label_count_dict)  # 按样本数降序排之后的key列表

            print("length of list_label:", len(list_label))

            for i, label in enumerate(list_label):
                label2idx[label] = i
                idx2label[i] = label

            # 如果不存在写到缓存文件中
            if not os.path.exists(cache_path):
                with open(cache_path, 'ab') as data_f:
                    pickle.dump((label2idx, idx2label), data_f)
        print("building vocabulary_label(sorted) ended.len of vocabulary_label: ", len(idx2label))
        return label2idx, idx2label



