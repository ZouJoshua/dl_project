#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 19-5-16 下午6:45
@File    : preprocess_data_imdb.py
@Desc    : 电影评论数据预处理
"""

import os
import pandas as pd
from bs4 import BeautifulSoup


import gensim
from gensim.models import word2vec

import numpy as np
from collections import Counter
import json

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)




def getRate(subject):
    splitList = subject[1:-1].split("_")
    return int(splitList[1])


def cleanReview(subject):
    beau = BeautifulSoup(subject)
    newSubject = beau.get_text()
    # newSubject = newSubject.replace("\\", "").replace("\'", "").replace('/', '').replace('"', '').replace(',', '').replace('.', '').replace('?', '').replace('(', '').replace(')', '')
    newSubject = newSubject.strip().split(" ")
    newSubject = [word.lower() for word in newSubject]
    newSubject = " ".join(newSubject)

    return newSubject

def get_clean_and_embed_file(raw_label_train, raw_unlabel_train, embed_file, train_file):
    print(">>>>> 正在处理原始数据文件")
    with open(raw_unlabel_train, "r") as f:
        unlabeledTrain = [line.strip().split("\t") for line in f.readlines() if len(line.strip().split("\t")) == 2]

    with open(raw_label_train, "r") as f:
        labeledTrain = [line.strip().split("\t") for line in f.readlines() if len(line.strip().split("\t")) == 3]

    unlabel = pd.DataFrame(unlabeledTrain[1:], columns=unlabeledTrain[0])
    label = pd.DataFrame(labeledTrain[1:], columns=labeledTrain[0])
    label["rate"] = label["id"].apply(getRate)
    unlabel["review"] = unlabel["review"].apply(cleanReview)
    label["review"] = label["review"].apply(cleanReview)
    newDf = pd.concat([unlabel["review"], label["review"]], axis=0)
    newDf.to_csv(embed_file, index=False)
    print("<<<<< 【{}】embed文件已生成".format(embed_file))
    newLabel = label[["review", "sentiment", "rate"]]
    newLabel.to_csv(train_file, index=False)
    print("<<<<< 【{}】训练文件已生成".format(train_file))

def generate_word2vec(embed_file, out_dir, vec_bin="word2Vec.bin"):
    sentences = word2vec.LineSentence(embed_file)
    a = list(sentences)
    print(">>>>> 总共{}文档".format(len(a)))
    vec_bin_file = os.path.join(out_dir, vec_bin)
    model = gensim.models.Word2Vec(sentences, size=200, sg=1, workers=4)
    model.wv.save_word2vec_format(vec_bin_file, binary=True)
    print("<<<<< 词向量【{}】已训练完成".format(vec_bin))


# 配置参数

class TrainingConfig(object):
    epoches = 10
    evaluateEvery = 100
    checkpointEvery = 100
    learningRate = 0.001


class ModelConfig(object):
    embeddingSize = 200
    numFilters = 128

    filterSizes = [2, 3, 4, 5]
    dropoutKeepProb = 0.5
    l2RegLambda = 0.0


class Config(object):
    sequenceLength = 200  # 取了所有序列长度的均值
    batchSize = 128

    dataSource = "../data/imdb/processedData/labeledTrain.csv"

    stopWordSource = "../data/imdb/imdb_stopwords.txt"

    numClasses = 2

    rate = 0.8  # 训练集的比例

    training = TrainingConfig()

    model = ModelConfig()



class Dataset(object):
    def __init__(self, config):
        self._dataSource = config.dataSource
        self._stopWordSource = config.stopWordSource

        self._sequenceLength = config.sequenceLength  # 每条输入的序列处理为定长
        self._embeddingSize = config.model.embeddingSize
        self._batchSize = config.batchSize
        self._rate = config.rate

        self._stopWordDict = {}

        self.trainReviews = []
        self.trainLabels = []

        self.evalReviews = []
        self.evalLabels = []

        self.wordEmbedding = None

        self._wordToIndex = {}
        self._indexToWord = {}

    def _readData(self, filePath):
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
        with open("../data/imdb/processedData/wordToIndex.json", "w", encoding="utf-8") as f:
            json.dump(self._wordToIndex, f)

        with open("../data/imdb/processedData/indexToWord.json", "w", encoding="utf-8") as f:
            json.dump(self._indexToWord, f)

    def _getWordEmbedding(self, words):
        """
        按照我们的数据集中的单词取出预训练好的word2vec中的词向量
        """

        wordVec = gensim.models.KeyedVectors.load_word2vec_format("../data/imdb/processedData/word2Vec.bin", binary=True)
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








def main():
    data_dir = "/home/zoushuai/algoproject/tf_project/data/imdb/rawData"
    label_file = os.path.join(data_dir, 'labeledTrainData.tsv')
    unlabel_file = os.path.join(data_dir, 'unlabeledTrainData.tsv')
    out_dir = "/home/zoushuai/algoproject/tf_project/data/imdb/processedData"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    train_file = os.path.join(out_dir, "labeledTrain.csv")
    embed_file = os.path.join(out_dir, "embedding.txt")
    # get_clean_and_embed_file(label_file, unlabel_file, embed_file, train_file)
    # generate_word2vec(embed_file, out_dir)

    config = Config()
    data = Dataset(config)
    data.dataGen()

    print("train data shape: {}".format(data.trainReviews.shape))
    print("train label shape: {}".format(data.trainLabels.shape))
    print("eval data shape: {}".format(data.evalReviews.shape))

if __name__ == '__main__':
    main()