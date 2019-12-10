#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 12/10/19 12:12 PM
@File    : preprocess_data_zhwiki.py
@Desc    : 解析xml,将xml的wiki数据转换为text格式

"""


import logging
import os
import json

from gensim.corpora import WikiCorpus
import jieba
from string import punctuation

from setting import DATA_PATH
from preprocess.preprocess_tools import read_json_format_file


def remove_punc(text):
    add_punc = '，。、【 】 “”：；（）《》‘’{}？！⑦()、%^>℃：.”“^-——=&#@￥\n「」…『』'
    all_punc = punctuation + add_punc
    seg_list = jieba.cut(text, cut_all=False)
    no_punc = [w for w in seg_list if w not in all_punc]
    clean_text = " ".join(no_punc)
    return clean_text

def process_wiki_xml(logger):
    xml_file = os.path.join(DATA_PATH, "zhwiki", "zhwiki-latest-pages-articles.xml.bz2")
    text_file = os.path.join(DATA_PATH, "zhwiki", "wiki.zh.text")
    seg_file = os.path.join(DATA_PATH, "zhwiki", "wiki.zh.text.seg")

    space = " "
    i = 0

    output = open(text_file, "w")
    seg = open(seg_file, "w", encoding="utf-8")
    wiki = WikiCorpus(xml_file, lemmatize=False, dictionary={})
    for text in wiki.get_texts():
        output.write(space.join(text) + "\n")
        c_text = remove_punc(text)
        seg.write(c_text + "\n")
        i += 1
        if i % 10000 == 0:
            logger.info("Saved " + str(i) + " articles")

    output.close()
    seg.close()
    logger.info("Finished Saved " + str(i) + " articles")
    logger.info("Word segment Saved")

def process_wiki_json(logger):
    json_dir = "/data/common/zhwiki_data/wiki_zh"
    text_file = os.path.join(DATA_PATH, "zhwiki", "wiki.zh.text")
    seg_file = os.path.join(DATA_PATH, "zhwiki", "wiki.zh.text.seg")
    file_dir = [os.path.join(json_dir, i) for i in os.listdir(json_dir)]

    space = " "
    i = 0

    output = open(text_file, "w")
    seg = open(seg_file, "w", encoding="utf-8")

    for file_path in file_dir:
        files = [os.path.join(file_path, filename) for filename in os.listdir(file_path)]
        for file in files:
            lines = read_json_format_file(file)
            for line in lines:
                text = line["text"]
                output.write(text.replace("\n", " ") + "\n")
                c_text = remove_punc(text)
                seg.write(c_text + "\n")
                i += 1
                if i % 10000 == 0:
                    logger.info("Saved " + str(i) + " articles")

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
    logger.info("Finished Saved " + str(i) + " articles")
    logger.info("Word segment Saved")





if __name__ == "__main__":

    logger = logging.getLogger("preprocess_zhwiki")
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % " ".join(["preprocess_zhwiki", "zhwiki-latest-pages-articles.xml.bz2", "wiki.zh.text"]))
    # process_wiki_xml(logger)
    process_wiki_json(logger)


