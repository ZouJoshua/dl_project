#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 19-8-6 下午4:38
@File    : browser_feature_extracted.py
@Desc    : 浏览器文本特征抽取
"""

import json
from nltk.corpus import stopwords
from nlp_corpus_preprocess.preprocess_tools import CleanDoc, read_json_format_file, dict_sort
import os
from model_normal.fasttext_model import FastTextClassifier
from evaluate.eval_calculate import EvaluateModel
from utils.logger import Logger
from setting import LOG_PATH

log_file = os.path.join(LOG_PATH, 'fasttext_train_log')
log = Logger("fasttext_train_log", log2console=True, log2file=True, logfile=log_file).get_logger()



def extracted_feature(datafile):
    category_feature = dict()
    line_count = 0
    for line in read_json_format_file(datafile):
        cat = str(line["category"])
        title = line["article_title"]
        title_tok = get_clean_tokens(title)
        content = line["text"]
        line_count += 1
        if cat not in category_feature:
            category_feature[cat] = dict()
            for tok in title_tok:
                if tok in category_feature[cat]:
                    category_feature[cat][tok] += 1
                else:
                    category_feature[cat][tok] = 1
        else:
            for tok in title_tok:
                if tok in category_feature[cat]:
                    category_feature[cat][tok] += 1
                else:
                    category_feature[cat][tok] = 1
        if line_count % 10000 == 0:
            print("已处理{}行".format(line_count))
    result_dir = os.path.join(os.path.dirname(datafile), "category_feature_words")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    for k, v in category_feature.items():
        file = os.path.join(result_dir, str(k))
        with open(file, "w") as f:
            f.writelines(json.dumps(dict_sort(v), ensure_ascii=False, indent=4))
    # print(json.dumps(category_feature, ensure_ascii=False, indent=4))

def get_clean_tokens(text):
    l_text = text.lower()
    text_tok = l_text.split(' ')
    new_tokens_list = list()

    for tok in text_tok:
        tok = tok.replace("\r", " ").replace("\n", " ").replace("\t", " ")
        new_tok = CleanDoc(tok).remove_symbol(tok)
        new_tok = CleanDoc(new_tok).remove_emoji(new_tok)
        if new_tok and not new_tok.isdigit() and new_tok not in stopwords.words("english"):
            new_tokens_list.append(new_tok)

    return new_tokens_list

def gen_test_file(datafile):
    label2feature = {
        "211": "latest",
        "212": "trending_video",
        "213": "top_picks",
        "214": "offbeat",
        "215": "lifestyle",
        "216": "auto",
        "217": "entertainment",
        "218": "sports",
        "219": "movies",
        "220": "food",
        "221": "tv_shows",
        "222": "fashion",
        "223": "viral",
        "224": "music_dance",
        "225": "tech",
        "226": "inspirational",
        "227": "spirituality",
        "228": "news",
        "229": "status",
        "230": "adult"
    }
    category_feature = dict()
    result_dir = os.path.join(os.path.dirname(datafile), "category_feature_words")
    test_file = os.path.join(result_dir, 'test.txt')
    f = open(test_file, 'w')
    # line_count = 0
    for line in read_json_format_file(datafile):
        cat = str(line["category"])
        feature = label2feature.get(cat, "")
        title = line["article_title"]
        if cat not in category_feature:
            category_feature[cat] = 1
        else:
            category_feature[cat] += 1
            if category_feature[cat] < 100:
                title_tok = get_clean_tokens(title)
                feature_words = " ".join(title_tok)
                if feature_words.strip():
                    line = feature_words + "\t__label__" + feature
                    f.write(line+"\n")
            else:
                continue
    f.close()




def gen_train_file(data_dir):
    label2feature = {
        "211": "latest",
        "212": "trending_video",
        "213": "top_picks",
        "214": "offbeat",
        "215": "lifestyle",
        "216": "auto",
        "217": "entertainment",
        "218": "sports",
        "219": "movies",
        "220": "food",
        "221": "tv_shows",
        "222": "fashion",
        "223": "viral",
        "224": "music_dance",
        "225": "tech",
        "226": "inspirational",
        "227": "spirituality",
        "228": "news",
        "229": "status",
        "230": "adult"
    }
    train_file = os.path.join(data_dir, 'train.txt')
    out_list = list()
    for fname in os.listdir(data_dir):
        feature = label2feature.get(str(fname), "")
        feature_file = os.path.join(data_dir, fname)
        print(feature_file)
        with open(feature_file, 'r') as f:
            feature_dict = json.load(f)
        feature_words_list = list()
        for tok in feature_dict.keys():
            v = feature_dict[tok]
            tok = CleanDoc(tok).remove_emoji(tok)
            tok = tok.replace("\r", "").replace("\n", "").replace("\t", "")
            if tok:
                toks = [tok]*v
                feature_words_list += toks
                # feature_words = " ".join(feature_words_list)
                for tok_ in toks:
                    if tok_:
                        line = tok_ + "\t__label__" + feature
                        out_list.append(line)
        # feature_words_list = [tok.replace("\r", "").replace("\n", "").replace("\t", "") for tok in list(feature_dict.keys()) if CleanDoc(tok).remove_emoji(tok)]
        # feature_words = " ".join(feature_words_list)
        # line = feature_words + "\t__label__" + feature
        # out_list.append(line)
    with open(train_file, "w") as wf:
        for line in out_list:
            wf.write(line + "\n")



def train_feature_model(model_path, file_path, log):
    classifier = FastTextClassifier(model_path, train=False, file_path=file_path, logger=log)
    test_file = os.path.join(file_path, 'test.txt')
    test_predict_file = os.path.join(file_path, "test_predict")
    with open(test_file, 'r') as f:
        lines = f.readlines()
    rf = open(test_predict_file, 'w')
    for line in lines:
        out = dict()
        line = line.strip()
        # print(line)
        tmp = line.split("__label__")
        text, label = tmp[0], tmp[1]
        result = classifier.predict(text, k=3)
        out["text"] = text
        out["category"] = label
        pred_prob = result[0][0][1] + result[0][1][1]
        if result[0][0][1] > 0.9:
            feature_words = result[0][0][0].replace('__label__', '')
        else:
            if pred_prob > 0.9:
                feature_words = result[0][0][0].replace('__label__', '') + " " + result[0][1][0].replace('__label__', '')
            else:
                feature_words = result[0][0][0].replace('__label__', '') + " " + result[0][1][0].replace('__label__', '') \
                                + " " + result[0][2][0].replace('__label__', '')
        # out["predict_category"] = result[0][0][0].replace('__label__', '')
        if feature_words.find(label) >= 0:
            out["predict_category"] = label
        else:
            out["predict_category"] = result[0][0][0].replace('__label__', '')
        rf.write(json.dumps(out, ensure_ascii=False)+"\n")
    rf.close()
    label_list = sorted([i.replace("__label__", "") for i in classifier.model.labels])
    em = EvaluateModel(test_predict_file, key_name="category", logger=log, label_names=label_list)

    return em.evaluate_model_v2()

if __name__ == "__main__":
    file = "/data/browser_category/train/raw_data"
    # extracted_feature(file)
    feature_dir = "/data/browser_category/train/category_feature_words"
    # gen_train_file(feature_dir)
    # gen_test_file(file)
    train_feature_model("/data/browser_category/train/category_feature_words/category_feature",feature_dir, log)