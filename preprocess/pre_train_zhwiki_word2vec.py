#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 12/10/19 12:10 PM
@File    : pre_train_zhwiki_word2vec.py
@Desc    : 预训练中文wiki词向量

"""


import logging
import os
import sys
import multiprocessing

from setting import DATA_PATH

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

if __name__ == "__main__":
    logger = logging.getLogger("train_zhwiki_word2vec")
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % " ".join(["train_zhwiki_word2vec", "wiki.zh.text.seg", "wiki.zh.text.model", "wiki.zh.text.vector"]))


    seg_file = os.path.join(DATA_PATH, "zhwiki", "wiki.zh.text.seg")
    model_file = os.path.join(DATA_PATH, "zhwiki", "wiki.zh.text.model")
    vector_file = os.path.join(DATA_PATH, "zhwiki", "wiki.zh.text.vector")


    model = Word2Vec(LineSentence(seg_file), size=300, window=5, min_count=5, workers=multiprocessing.cpu_count())

    model.wv.save(model_file)
    model.wv.save_word2vec_format(vector_file, binary=False)