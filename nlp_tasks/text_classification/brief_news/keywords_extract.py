#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 1/2/20 2:51 PM
@File    : keywords_extract.py
@Desc    : 关键词抽取和主题模型

"""

import os

import jieba
import jieba.analyse as analyse
import pandas as pd
import gensim
from gensim import corpora, models, similarities

from setting import DATA_PATH


def read_news_from_csv(data_path, stopwords, cut=True):
    df = pd.read_csv(data_path, encoding='utf-8')
    df = df.dropna()
    lines = df.content.values.tolist()
    sentences = []
    if cut:
        for line in lines:
            try:
                segs = jieba.lcut(line)
                segs = filter(lambda x: len(x) > 1, segs)
                segs = filter(lambda x: x not in stopwords, segs)
                # sentences.append(segs)  # python2
                sentences.append(list(segs))
            except Exception as e:
                print(line)
                continue
    else:
        sentences = "".join(lines)
    return sentences

def read_stopwords(stopwords_path):
    stopwords = pd.read_csv(stopwords_path, index_col=False, quoting=3, sep="\t", names=['stopword'], encoding='utf-8')
    stopwords = stopwords['stopword'].values
    return stopwords


def extract_keywords_by_tfidf(content, idf_path=None):
    if idf_path:
        analyse.set_idf_path(idf_path)
    keywords = "  ".join(analyse.extract_tags(content, topK=30, withWeight=False, allowPOS=()))
    return keywords

def extract_keywords_by_textrank(content):
    keywords = "  ".join(analyse.textrank(content, topK=20, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v')))
    return keywords



def get_topic_words(sentences, k=20):
    # 词袋模型
    dictionary = corpora.Dictionary(sentences)
    corpus = [dictionary.doc2bow(sentence) for sentence in sentences]
    lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=k)
    # lda.print_topic(3, num_words=5)
    for topic in lda.print_topics(num_topics=k, num_words=8):
        print(topic[1])
    # lda.get_document_topics()


if __name__ == "__main__":
    data_path = os.path.join(DATA_PATH, "brief_news")
    stopwords_path = os.path.join(data_path, "stopwords.txt")
    mil_news_path = os.path.join(data_path, "military_news.csv")

    stopwords = read_stopwords(stopwords_path)
    # sentences = read_news_from_csv(mil_news_path, stopwords_path, cut=False)
    sentences = read_news_from_csv(mil_news_path, stopwords_path, cut=True)
    print(sentences[:1])
    get_topic_words(sentences)

