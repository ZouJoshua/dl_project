#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 2020/7/28 10:04 上午
@File    : keywords_extractor_by_tfidf.py
@Desc    : 

"""


import os
from os.path import dirname

import math
from collections import Counter

from ba_wordseg.en_process_with_nltk import WordSegmenPreprocess
count = Counter(stemmed)


class CalculateTFIDF(object):

    def __init__(self, word, count, count_list=None):
        """Calculate word frequency
        Args:
            word:
            count:
            count_list
        Returns:
            TF-IDF of some keywords
        """
        self.word = word
        self.count = count

        if count_list:
            self.count_list = count_list
        else:
            self.count_list = count

    def _tokens_frequency(self):
        """Calculate tokens frequency"""
        return self.count[self.word] / sum(self.count.values())

    def _n_containing(self):
        return sum(1 for count in self.count_list if self.word in count)

    def _inverse_document_frequency(self):
        """Calculate the inverse document frequency"""
        return math.log(len(self.count_list)) / (1 + self._n_containing())

    def tfidf(self):
        """Calculate TF-IDF"""
        return self._tokens_frequency() * self._inverse_document_frequency()


def main():
    keyword_list = list()
    stopwordsDir = os.path.dirname(dirname(os.path.realpath(__file__)))
    stopword_file = os.path.join(stopwordsDir, 'stopwords_en.txt')
    preprocess = WordSegmenPreprocess(text, stopwordsfile=stopword_file)
    words = preprocess.get_words()
    corpus = []
    scores = {word: CalculateTFIDF(word, count, count_list=corpus).tfidf() for word in count}
    sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    # for word, score in sorted_words[:topk]:
    #     print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))
    return dict((word, score) for word, score in sorted_words[:20])

if __name__ == "__main__":
    main()