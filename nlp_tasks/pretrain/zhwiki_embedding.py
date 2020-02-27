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
import multiprocessing

import jieba
from string import punctuation
from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

from setting import DATA_PATH, LOG_PATH
from utils.logger import Logger
from preprocess.common_tools import read_json_format_file



class ZhWikiPreProcess(object):

    def __init__(self, ori_data_path, output_path, is_xml=False, logger=None):
        self.data_path = ori_data_path
        self.xml_file = os.path.join(ori_data_path, "zhwiki-latest-pages-articles.xml.bz2")
        self.text_file = os.path.join(output_path, "wiki.zh.text")
        self.seg_file = os.path.join(output_path, "wiki.zh.text.seg")
        if logger:
            self.log = logger
        else:
            self.log = logging.getLogger("preprocess_zhwiki")
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
            logging.root.setLevel(level=logging.INFO)
        self.log.info("running %s" % " ".join(["preprocess_zhwiki", "zhwiki-latest-pages-articles.xml.bz2", "wiki.zh.text"]))

        if not (os.path.exists(self.text_file) and os.path.exists(self.seg_file)):
            if is_xml:
                self.process_wiki_xml()
            self.process_wiki_json()

    def remove_punc(self, text):
        add_punc = '，。、【 】 “”：；（）《》‘’{}？！⑦()、%^>℃：.”“^-——=&#@￥\n「」…『』'
        all_punc = punctuation + add_punc
        seg_list = jieba.cut(text, cut_all=False)
        no_punc = [w for w in seg_list if w not in all_punc]
        clean_text = " ".join(no_punc)
        return clean_text

    def process_wiki_xml(self):
        space = " "
        i = 0

        output = open(self.text_file, "w")
        seg = open(self.seg_file, "w", encoding="utf-8")
        if not os.path.exists(self.xml_file):
            raise Exception("XML file {} not found".format(self.xml_file))

        wiki = WikiCorpus(self.xml_file, lemmatize=False, dictionary={})
        for text in wiki.get_texts():
            output.write(space.join(text) + "\n")
            c_text = self.remove_punc(text)
            seg.write(c_text + "\n")
            i += 1
            if i % 10000 == 0:
                self.log.info("Saved " + str(i) + " articles")

        output.close()
        seg.close()
        self.log.info("Finished Saved " + str(i) + " articles")
        self.log.info("Word segment Saved")

    def process_wiki_json(self):

        file_dir = [os.path.join(self.data_path, i) for i in os.listdir(self.data_path)]

        space = " "
        i = 0

        output = open(self.text_file, "w")
        seg = open(self.seg_file, "w", encoding="utf-8")

        for file_path in file_dir:
            files = [os.path.join(file_path, filename) for filename in os.listdir(file_path)]
            for file in files:
                lines = read_json_format_file(file)
                for line in lines:
                    text = line["text"]
                    output.write(text.replace("\n", " ") + "\n")
                    c_text = self.remove_punc(text)
                    seg.write(c_text + "\n")
                    i += 1
                    if i % 10000 == 0:
                        self.log.info("Saved " + str(i) + " articles")

        # lines = read_json_format_file("/data/common/zhwiki_data/wiki_zh/AA/wiki_00")
        # for line in lines:
        #     text = line["text"]
        #     output.write(text + "\n")
        #     c_text = remove_punc(text)
        #     seg.write(c_text + "\n")
        #     i += 1
        #     if i % 10000 == 0:
        #         logger.info("Saved " + str(i) + " articles")

        output.close()
        seg.close()
        self.log.info("Finished Saved " + str(i) + " articles")
        self.log.info("Word segment Saved")


def main():
    ori_data_path = "/data/common/zhwiki_data/wiki_zh"
    output_path = os.path.join(DATA_PATH, "zhwiki")
    log_file = os.path.join(LOG_PATH, 'train_zhwiki_embedding_log')
    train_logger = Logger("fasttext_train_log", log2console=True, log2file=True, logfile=log_file).get_logger()

    ZhWikiPreProcess(ori_data_path, output_path, is_xml=False, logger=train_logger)

    train_logger.info("running %s" % " ".join(["train_zhwiki_word2vec", "wiki.zh.text.seg", "wiki.zh.text.model", "wiki.zh.text.vector"]))

    seg_file = os.path.join(output_path, "wiki.zh.text.seg")
    model_file = os.path.join(output_path, "wiki.zh.text.model")
    vector_file = os.path.join(output_path, "wiki.zh.text.vector")

    model = Word2Vec(LineSentence(seg_file), size=300, window=5, min_count=5, workers=multiprocessing.cpu_count())

    model.wv.save(model_file)
    model.wv.save_word2vec_format(vector_file, binary=False)



if __name__ == "__main__":
    main()