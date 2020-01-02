#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 1/2/20 10:04 AM
@File    : text_word_cloud.py
@Desc    : 词云

"""

import os
import warnings
warnings.filterwarnings("ignore")
import jieba
import numpy as np
import codecs   #codecs提供的open方法来指定打开的文件的语言编码，它会在读取的时候自动转换为内部unicode
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

from wordcloud import WordCloud, ImageColorGenerator
from scipy.misc import imread

from setting import DATA_PATH



def read_words_from_csv(data_path):
    df = pd.read_csv(data_path, encoding='utf-8')
    df = df.dropna()
    content = df.content.values.tolist()
    # jieba.load_userdict(u"data/user_dic.txt")
    segment = []
    for line in content:
        try:
            segs = jieba.lcut(line)
            for seg in segs:
                if len(seg) > 1 and seg != '\r\n':
                    segment.append(seg)
        except:
            print(line)
            continue
    return segment


def read_stopwords(stopwords_path):
    stopwords = pd.read_csv(stopwords_path, index_col=False, quoting=3, sep="\t", names=['stopword'], encoding='utf-8')  # quoting=3全不引用
    # print(stopwords.head())
    return stopwords


def word_frequency_stat(words_df, stopwords_df):
    """
    统计词频
    :param words_df: 原始词
    :param stopwords_df: 停用词
    :return:
    """
    # 去停用词
    words_df = words_df[~words_df.segment.isin(stopwords_df.stopword)]
    # 统计词频
    words_stat = words_df.groupby(by=['segment'])['segment'].agg({"wordcount": np.size})
    words_stat = words_stat.reset_index().sort_values(by=["wordcount"], ascending=False)
    print(words_stat.head())
    return words_stat


def plot_word_cloud(words_stat, font_path, custom_path=None):
    # matplotlib.rcParams['figure.figsize'] = (10.0, 5.0)
    matplotlib.rcParams['figure.figsize'] = (10.0, 10.0)
    word_frequence = {x[0]: x[1] for x in words_stat.head(1000).values}

    if custom_path:
        bimg = imread(custom_path)
        wordcloud = WordCloud(font_path=font_path, mask=bimg, background_color="white", max_font_size=80)
        wordcloud = wordcloud.fit_words(word_frequence)
        bimgColors = ImageColorGenerator(bimg)
        plt.axis("off")
        plt.imshow(wordcloud.recolor(color_func=bimgColors))
    else:
        wordcloud = WordCloud(font_path=font_path, background_color="white", max_font_size=80)
        wordcloud = wordcloud.fit_words(word_frequence)
        plt.axis("off")
        plt.imshow(wordcloud)
    plt.show()

def generate_word_cloud(news_path, stopwords_path, font_path):
    stopwords_df = read_stopwords(stopwords_path)

    # 读取新闻数据，分词
    segment = read_words_from_csv(news_path)
    words_df = pd.DataFrame({'segment': segment})
    # print(stopwords)
    # print(words_df.head())
    ent_words_stat = word_frequency_stat(words_df, stopwords_df)
    plot_word_cloud(ent_words_stat, font_path)



if __name__ == "__main__":
    data_path = os.path.join(DATA_PATH, "brief_news")
    stopwords_path = os.path.join(data_path, "stopwords.txt")
    font_path = os.path.join(data_path, "simhei.ttf")

    ent_news_path = os.path.join(data_path, "entertainment_news.csv")
    sports_news_path = os.path.join(data_path, "sports_news.csv")

    # 娱乐新闻词云
    generate_word_cloud(ent_news_path, stopwords_path, font_path)
    # 体育新闻词云
    generate_word_cloud(sports_news_path, stopwords_path, font_path)




