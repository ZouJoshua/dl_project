#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 19-5-17 下午2:16
@File    : datasets.py
@Desc    : 
"""

import numpy as np
import json
from gensim.models import Word2Vec, KeyedVectors
from preprocess.preprocess_utils import read_json_format_file,read_txt_file


class DataSet(object):
    def __init__(self, config):
        self._dataSource = config.dataSource
        self._stopWordSource = config.stopWordSource

        self._sequenceLength = config.sequenceLength  # 每条输入的序列处理为定长
        self._embedding_dim = config.model.embeddingSize
        self._batch_size = config.batchSize
        self._rate = config.rate

        self._stopWordDict = {}

        self.trainReviews = []
        self.trainLabels = []

        self.evalReviews = []
        self.evalLabels = []

        self.wordEmbedding = None

        self._wordToIndex = {}
        self._indexToWord = {}

    def _readData(self, file):
        """
        从csv文件中读取数据集
        """

        df = pd.read_csv(filePath)
        labels = df["sentiment"].tolist()
        review = df["review"].tolist()
        reviews = [line.strip().split() for line in review]

        return reviews, labels

    def _reviewProcess(self, review, sequenceLength, wordToIndex):
        """
        将数据集中的每条评论用index表示
        wordToIndex中“pad”对应的index为0
        """

        reviewVec = np.zeros((sequenceLength))
        sequenceLen = sequenceLength

        # 判断当前的序列是否小于定义的固定序列长度
        if len(review) < sequenceLength:
            sequenceLen = len(review)

        for i in range(sequenceLen):
            if review[i] in wordToIndex:
                reviewVec[i] = wordToIndex[review[i]]
            else:
                reviewVec[i] = wordToIndex["UNK"]

        return reviewVec

    def _genTrainEvalData(self, x, y, rate):
        """
        生成训练集和验证集
        """

        reviews = []
        labels = []

        # 遍历所有的文本，将文本中的词转换成index表示
        for i in range(len(x)):
            reviewVec = self._reviewProcess(x[i], self._sequenceLength, self._wordToIndex)
            reviews.append(reviewVec)

            labels.append([y[i]])

        trainIndex = int(len(x) * rate)

        trainReviews = np.asarray(reviews[:trainIndex], dtype="int64")
        trainLabels = np.array(labels[:trainIndex], dtype="float32")

        evalReviews = np.asarray(reviews[trainIndex:], dtype="int64")
        evalLabels = np.array(labels[trainIndex:], dtype="float32")

        return trainReviews, trainLabels, evalReviews, evalLabels

    def _genVocabulary(self, reviews):
        """
        生成词向量和词汇-索引映射字典，可以用全数据集
        """

        allWords = [word for review in reviews for word in review]

        # 去掉停用词
        subWords = [word for word in allWords if word not in self.stopWordDict]

        wordCount = Counter(subWords)  # 统计词频
        sortWordCount = sorted(wordCount.items(), key=lambda x: x[1], reverse=True)

        # 去除低频词
        words = [item[0] for item in sortWordCount if item[1] >= 5]

        vocab, wordEmbedding = self._getWordEmbedding(words)
        self.wordEmbedding = wordEmbedding

        self._wordToIndex = dict(zip(vocab, list(range(len(vocab)))))
        self._indexToWord = dict(zip(list(range(len(vocab))), vocab))

        # 将词汇-索引映射表保存为json数据，之后做inference时直接加载来处理数据
        with open("../data/wordJson/wordToIndex.json", "w", encoding="utf-8") as f:
            json.dump(self._wordToIndex, f)

        with open("../data/wordJson/indexToWord.json", "w", encoding="utf-8") as f:
            json.dump(self._indexToWord, f)

    def _getWordEmbedding(self, words):
        """
        按照我们的数据集中的单词取出预训练好的word2vec中的词向量
        """

        wordVec = KeyedVectors.load_word2vec_format("../word2vec/word2Vec.bin", binary=True)
        vocab = []
        wordEmbedding = []

        # 添加 "pad" 和 "UNK",
        vocab.append("pad")
        vocab.append("UNK")
        wordEmbedding.append(np.zeros(self._embeddingSize))
        wordEmbedding.append(np.random.randn(self._embeddingSize))

        for word in words:
            try:
                vector = wordVec.wv[word]
                vocab.append(word)
                wordEmbedding.append(vector)
            except:
                print(word + "不存在于词向量中")

        return vocab, np.array(wordEmbedding)

    def _readStopWord(self, stopWordPath):
        """
        读取停用词
        """

        with open(stopWordPath, "r") as f:
            stopWords = f.read()
            stopWordList = stopWords.splitlines()
            # 将停用词用列表的形式生成，之后查找停用词时会比较快
            self.stopWordDict = dict(zip(stopWordList, list(range(len(stopWordList)))))

    def dataGen(self):
        """
        初始化训练集和验证集
        """

        # 初始化停用词
        self._readStopWord(self._stopWordSource)

        # 初始化数据集
        reviews, labels = self._readData(self._dataSource)

        # 初始化词汇-索引映射表和词向量矩阵
        self._genVocabulary(reviews)

        # 初始化训练集和测试集
        trainReviews, trainLabels, evalReviews, evalLabels = self._genTrainEvalData(reviews, labels, self._rate)
        self.trainReviews = trainReviews
        self.trainLabels = trainLabels

        self.evalReviews = evalReviews
        self.evalLabels = evalLabels

    def create_vocabulary(word2vec_model_path, name_scope=''):
        """
        创建词汇索引表
        :param word2vec_model_path: 训练好的word2vec模型存放路径
        :return: {单词：索引}表和{索引：单词}表
        """
        # TODO：这里需要加参数
        cache_path = "/data/caifuli/news_classification/textcnn/cache_vocabulary_pik/" + name_scope + "_word_vocabulary.pik"
        print("cache_path:", cache_path, "file_exists:", os.path.exists(cache_path))

        if os.path.exists(cache_path):  # 如果缓存文件存在，则直接读取
            with open(cache_path, 'rb') as data_f:
                vocabulary_word2idx, vocabulary_idx2word = pickle.load(data_f)
                return vocabulary_word2idx, vocabulary_idx2word
        else:
            vocabulary_word2idx = {}
            vocabulary_idx2word = {}

            print("building vocabulary（words with frequency above 5 are included). word2vec_path:", word2vec_model_path)
            vocabulary_word2idx['PAD_ID'] = 0
            vocabulary_idx2word[0] = 'PAD_ID'
            special_index = 0

            model = Word2Vec.load(word2vec_model_path)
            # model = word2vec.load(word2vec_model_path, kind='bin')

            for i, vocab in enumerate(model.wv.vocab):
                if vocab.isalpha():
                    vocabulary_word2idx[vocab] = i + 1 + special_index  # 只设了一个special ID
                    vocabulary_idx2word[i + 1 + special_index] = vocab

            # 如果不存在写到缓存文件中
            if not os.path.exists(cache_path):
                with open(cache_path, 'ab') as data_f:
                    pickle.dump((vocabulary_word2idx, vocabulary_idx2word), data_f)
        return vocabulary_word2idx, vocabulary_idx2word

    def create_label_vocabulary(training_data_dir_path='/data/caifuli/news_classification/data', name_scope=''):
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
            fnames.remove('.DS_Store')
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