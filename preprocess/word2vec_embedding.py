#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 19-5-15 下午6:13
@File    : word2vec_embedding.py
@Desc    : 
"""


import os
import json
from gensim.models import word2vec


def generate_sentences(dirname):
    fnames = os.listdir(dirname)
    sentences = []
    for fname in fnames:
        print(os.path.join(dirname, fname))
        with open(os.path.join(dirname, fname), "r") as f:
            for line in f.readlines():
                line = json.loads(line)
                title = line["title"].replace("\t", "").replace("\n", "").replace("\r", "")
                content = ""
                if "html" in line and line["html"].strip() != "":
                    content = line["html"].strip()
                if "content" in line and line["content"].strip() != "":
                    content = line["content"].strip()
                desc = title + content
                word_list = []
                for word in desc.split(" "):
                    if word.isalpha():
                        word_list.append(word.lower())
                # word_list = [word if word.isalpha() else pass for word in title.split(" ")]
                sentences.append(word_list)
    return sentences






# 对每篇文章生成vector
def generate_doc_word_list(text_json):
    title = text_json["title"].strip().replace("\t", "").replace("\n", "").replace("\r", "")
    html = PyQuery(text_json["html"]).text().strip()
    desc = title + ". " + html
    word_list = []
    for word in desc.split(" "):
        if word.isalpha():
            word_list.append(word.lower())
    return word_list


sentences = generate_sentences(dataDir)
model = word2vec.Word2Vec(sentences, size=300)  # 默认训练词向量的时候把频次小于5的单词从词汇表中剔除掉
model.save(modelPath)
model.wv.save_word2vec_format(wordVectorPath + ".bin", binary=True)