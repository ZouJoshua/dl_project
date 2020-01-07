#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 19-7-12 上午9:46
@File    : predict_browser_category.py
@Desc    : 预测浏览器视频分类
"""


import json
import time
import fasttext
from nlp_corpus_preprocess.preprocess_tools import read_json_format_file, CleanDoc


def _preline(line_json):
    # line_json = json.loads(line)
    title = line_json["article_title"]
    content = ""
    if "category" in line_json:
        dataY = str(line_json["category"])
    else:
        dataY = ""
    # dataX = clean_string((title + '.' + content).lower())  # 清洗数据
    dataX = CleanDoc(title.lower()).text  # 清洗数据
    if dataX:
        _data = dataX + "\t__label__" + dataY
        return _data
    else:
        return None



def predict(model_file, json_file, json_out_file):
    classifier = fasttext.load_model(model_file)
    with open(json_out_file, 'w', encoding='utf-8') as joutfile:
        s = time.time()
        for line in read_json_format_file(json_file):
            if "category" in line:
                dataY = str(line["category"])
            else:
                dataY = ""
            if _preline(line):
                _data = _preline(line).split("\t__label__")[0]
                new_line = {"_id": line["_id"], "text":line["text"], "article_title":line["article_title"], "category":dataY}
                if _data:
                    labels = classifier.predict_proba([_data])
                    new_line['predict_category'] = labels[0][0][0].replace("'", "").replace("__label__", "")
                    # print(line['predict_top_category'])
                    new_line['predict_category_proba'] = labels[0][0][1]
                    joutfile.write(json.dumps(new_line) + "\n")
                    del line
            else:
                continue
        e = time.time()
        print('预测及写入文件耗时{}'.format(e - s))


def predict_statistic(predict_file):

    raw_cat = dict()
    predict_cat = dict()
    predict_proba_limit_cat = dict()

    for line in read_json_format_file(predict_file):
        cat = line["category"]
        pre_cat = line["predict_category"]
        pre_prob = line["predict_category_proba"]
        if cat in raw_cat:
            raw_cat[cat] += 1
        else:
            raw_cat[cat] = 1

        if pre_cat in predict_cat:
            predict_cat[pre_cat] += 1
        else:
            predict_cat[pre_cat] = 1

        if pre_prob < 0.5:
            if pre_cat in predict_proba_limit_cat:
                predict_proba_limit_cat[pre_cat] += 1
            else:
                predict_proba_limit_cat[pre_cat] = 1

    print(">>>>> 原始分类：\n{}".format(json.dumps(raw_cat, indent=4)))
    print(">>>>> 预测分类：\n{}".format(json.dumps(predict_cat, indent=4)))
    print(">>>>> 预测概率低于0.5的分类：\n{}".format(json.dumps(predict_proba_limit_cat, indent=4)))



if __name__ == '__main__':
    raw_file = "/data/browser_category/youtube_videos.txt"
    model_file = "/data/browser_category/category_model_1/category_classification_model.bin"
    predict_file = "/data/browser_category/predict_youtube_videos.txt"
    # predict(model_file, raw_file,predict_file)
    predict_statistic(predict_file)