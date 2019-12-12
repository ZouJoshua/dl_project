#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 19-8-22 下午12:16
@File    : ner_with_crf.py
@Desc    : 
"""



import pycrfsuite
import json
import re
import string
import nltk
import langid

from itertools import chain
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer


crfModelPath = "/data/crfsuite_0625"

tagger = pycrfsuite.Tagger()
tagger.open(crfModelPath)


def clean_string(string):
    string = re.sub(r"[a-zA-z]+://[^\s]*", "", string)  # remove url: http://xxx
    string = re.sub(r"www(\.\w+)+", "", string)  # remove url: www.xxx
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r":", " : ", string)
    string = re.sub(r"\. ", " . ", string)
    string = re.sub(r"-", " - ", string)
    string = re.sub(r"&", " & ", string)
    #string = string.replace("...", "")  #.lower()
    return string


def word2fearures(sent, i):  # TODO：加入新的特征
    """
    1.当前词的小写格式
    2.当前词的后缀
    3.当前词是否全大写 isupper
    4.当前词的首字母大写，其他字母小写判断 istitle
    5.当前词是否为数字 isdigit
    6.当前词的词性
    7.当前词的词性前缀
    8.还有就是与之前后相关联的词的上述特征（类似于特征模板的定义）
    :param sent:
    :param i:
    :return:
    """
    word = sent[i][0]
    postag = sent[i][1]
    punc = string.punctuation

    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isupper=%s' % word.isupper(),  # 是否全大写
        'word.istitle=%s' % word.istitle(),  # 是否首字母大写
        'word.isdigit=%s' % word.isdigit(),  # 是否只由数字构成
        'word.hasnum=%s' % any(char.isdigit() for char in word),  # 是否含数字
        'word.containspunc=%s' % any(char in punc for char in word),   # 是否含标点符号
        #'word.ispunc=%s' % word in punc,  # 是否全为标点符号
        'postag=' + postag,
        'postag[:2]=' + postag[:2],
    ]

    if i > 0:
        word1 = sent[i - 1][0]   # 前一个词的特征
        postag1 = sent[i - 1][1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:postag=' + postag1,
            '-1:postag[:2]=' + postag1[:2],
            '-1:hasnum=%s' % any(char.isdigit() for char in word1),  # 是否含数字
            '-1:containspunc=%s' % any(char in punc for char in word1),  # 是否含标点符号
            #'-1:ispunc=%s' % word in punc  # 是否全为标点符号
        ])
        if i > 1:
            word2 = sent[i - 2][0]  # 前两个词的特征
            postag2 = sent[i - 2][1]
            features.extend([
                '-2:word.lower=' + word2.lower(),
                '-2:word.istitle=%s' % word2.istitle(),
                '-2:word.isupper=%s' % word2.isupper(),
                '-2:postag=' + postag2,
                '-2:postag[:2]=' + postag2[:2],
                '-2:hasnum=%s' % any(char.isdigit() for char in word2),  # 是否含数字
                '-2:containspunc=%s' % any(char in punc for char in word2),  # 是否含标点符号
                # '-1:ispunc=%s' % word in punc  # 是否全为标点符号
            ])
    else:
        features.append('BOS')

    if i < len(sent) - 1:  # 后一个词的特征
        word3 = sent[i + 1][0]
        postag3 = sent[i + 1][1]
        features.extend([
            '+1:word.lower=' + word3.lower(),
            '+1:word.istitle=%s' % word3.istitle(),
            '+1:word.isupper=%s' % word3.isupper(),
            '+1:postag=' + postag3,
            '+1:postag[:2]=' + postag3[:2],
            '+1:hasnum=%s' % any(char.isdigit() for char in word3),  # 是否含数字
            '+1:containspunc=%s' % any(char in punc for char in word3),  # 是否含标点符号
            #'+1:ispunc=%s' % word in punc  # 是否全为标点符号
        ])
        if i < len(sent) - 2:
            word4 = sent[i + 1][0]
            postag4 = sent[i + 1][1]
            features.extend([
                '+2:word.lower=' + word4.lower(),
                '+2:word.istitle=%s' % word4.istitle(),
                '+2:word.isupper=%s' % word4.isupper(),
                '+2:postag=' + postag4,
                '+2:postag[:2]=' + postag4[:2],
                '+2:hasnum=%s' % any(char.isdigit() for char in word4),  # 是否含数字
                '+2:containspunc=%s' % any(char in punc for char in word4),  # 是否含标点符号
                # '+1:ispunc=%s' % word in punc  # 是否全为标点符号
            ])

    else:
        features.append('EOS')

    return features


def sent2features(sent):
    """
    完成特征转化
    :param sent:
    :return:
    """
    return [word2fearures(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    """
    获取类别，即标签
    :param sent:
    :return:
    """
    return [label for token, postag, label in sent]


def sent2tokens(sent):
    """
    获取词
    :param sent:
    :return:
    """
    return [token for token, postag, label in sent]



def bio_classification_report(y_true, y_pred):
    """
    Classification report for a list of BIO-encoded sequences.
    It computes token-level metrics and discards "O" labels.

    Note that it requires scikit-learn 0.15+ (or a version from github master)
    to calculate averages properly!
    """
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

    tagset = set(lb.classes_) - {'O'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels=[class_indices[cls] for cls in tagset],
        target_names=tagset,
    )



def colldata_tagger_train():
    #step1. 准备nltk内建立的Coll2002语料库
    nltk.corpus.conll2002.fileids()
    train_sents = list(nltk.corpus.conll2002.iob_sents('esp.train'))
    test_sents = list(nltk.corpus.conll2002.iob_sents('esp.testb'))
    X_train = [sent2features(s) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]
    X_test = [sent2features(s) for s in test_sents]
    y_test = [sent2labels(s) for s in test_sents]


    #step2. 创建Trainer并加载训练集
    trainer = pycrfsuite.Trainer(verbose=False)

    # 加载训练特征和分类的类别（label)
    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)

    #step3. 设置训练参数，包括算法选择。 这里选择L-BFGS训练算法和Elastic Net回归模型
    trainer.set_params({
        'c1': 1.0,   # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 50,  # stop earlier

        # include transitions that are possible, but not observed
        'feature.possible_transitions': True
    })

    #step4. 训练
    # #含义是训练出的模型名为：conll2002-esp.crfsuite
    trainer.train('conll2002-esp.crfsuite')

def colldata_tagger_test():

    test_sents = list(nltk.corpus.conll2002.iob_sents('esp.testb'))
    X_test = [sent2features(s) for s in test_sents]
    y_test = [sent2labels(s) for s in test_sents]

    tagger = pycrfsuite.Tagger()
    tagger.open('conll2002-esp.crfsuite')

    example_sent = test_sents[0]
    print(' '.join(sent2tokens(example_sent)), end='\n\n')

    print("Predicted:", ' '.join(tagger.tag(sent2features(example_sent))))
    print("Correct:  ", ' '.join(sent2labels(example_sent)))

    y_pred = [tagger.tag(xseq) for xseq in X_test]
    print(bio_classification_report(y_test, y_pred))









# 进行标签预测
predict_resPath = "/data/shopping_crf"
testPath = "/data/whole_shopping"
with open(predict_resPath, "a") as predict_f, open(testPath, "r") as test_f:
    lines = test_f.readlines()
    for line in lines:
        #res = {}
        line_js = json.loads(line)
        desc = ""
        if "headline_s" in line:
            desc = desc + line_js["headline_s"] + " . "
        if "summary_s" in line:
            desc = desc + line_js["summary_s"]
        desc = desc.replace("\n", "").replace("\r", "")
        lang = langid.classify(desc)[0]
        if lang != "en":
            continue
        line_js["clean_text"] = clean_string(desc)
        t_words = line_js["clean_text"].split(" ")

        words = []
        for t_w in t_words:
            if t_w != "":
                words.append(t_w)

        pos_res = nltk.pos_tag(words)
        #res['text'] = text

        predict_res = tagger.tag(sent2features(pos_res))
        feature_list = []
        for pos, predict in zip(pos_res, predict_res):
            feature = (pos[0], pos[1], predict)
            #print(feature)
            feature_list.append(feature)
        line_js['word_tags'] = feature_list
        predict_f.write(json.dumps(line_js) + "\n")

# 生成预测结果
resPath = "/data/shopping_crf_keyword"
keywordPath = "/data/shopping_keyword"
cnt = 0
with open(predict_resPath, "r") as original_f, open(keywordPath, "r") as keyword_f, open(resPath, "a") as predict_f:
    count = 0
    brand_dict = {}
    item_dict = {}
    google_dict = {}
    for k_line in keyword_f.readlines():
        count += 1
        if count == 1:
            brand_dict = json.loads(k_line)
        elif count == 2:
            item_dict = json.loads(k_line)
        elif count == 3:
            google_dict = json.loads(k_line)

    lines = original_f.readlines()
    print(len(lines))
    for line in lines:
        brand_list = []
        item_list = []
        price_list = []
        line_js = json.loads(line)
        word_tags = line_js["word_tags"]
        brand = ""
        item = ""
        price = ""
        for word_tag in word_tags:
            t = word_tag[2]
            w = word_tag[0]
            if t == 'B-Brand':
                brand = w
            elif t == "I-Brand":
                brand = brand + " " + w
            elif t == "B-Item":
                item = w
            elif t == "I-Item":
                item = item + " " + w
            elif t == "B-Price":
                price = w
            elif t == "I-Price":
                price = price + " " + w
            else:
                if brand != "":
                    brand_list.append(brand.replace(".", ""))
                if item != "":
                    item_list.append(item.lower().replace("..", ""))
                if price != "":
                    price_list.append(price.replace(u'\xa0', u' ').replace("/", "").replace(" ", ""))

        line_js['item'] = list(set(item_list))
        line_js['brand'] = list(set(brand_list))
        line_js['price'] = list(set(price_list))

        category = "unknown"
        # 使用商品名进行关键词匹配
        if len(list(set(item_list))):
            c_list = []
            for i in list(set(item_list)):
                if i.lower() in item_dict and isinstance(item_dict[i.lower()], str):
                    c_list.append(item_dict[i.lower()])
                elif i.lower() in google_dict:
                    c_list.append(google_dict[i.lower()])
            if len(list(set(c_list))) == 1:
                category = list(set(c_list))[0]
            elif len(list(set(c_list))) >= 2:
                category = "Shopping"
        if category == "unknown":
            c1_list = []
            # todo：看一下如果没在物品名词库里找到对应的物品的话，把物品名拆开来看一下结果
            item_words = [i.split(" ") for i in line_js['item']]
            for item_word in item_words:
                item_word_candidate = []
                i_w_len = len(item_word)
                for i in range(i_w_len):
                    candidate = ' '.join(item_word[-i:]).strip()
                    item_word_candidate.append(candidate)
                for candidate in item_word_candidate:
                    if candidate.lower() in item_dict:
                        c1_list.append(item_dict[candidate.lower()])
                    elif candidate.lower() in google_dict:
                        c1_list.append(google_dict[candidate.lower()])
            # 合并列表，铺开
            flatten = lambda x: [subitem for item in x for subitem in flatten(item)] if type(x) is list else [x]
            print(line_js['item'])
            # print(c1_list)
            c1_list = flatten(c1_list)
            print(c1_list)
            if len(list(set(c1_list))) == 1:
                category = list(set(c1_list))[0]

        # 使用品牌名进行关键词匹配
        if category == "unknown":
            if len(list(set(brand_list))) == 1:
                b = list(set(brand_list))[0]
                if b.lower() in brand_dict and isinstance(brand_dict[b.lower()], str):
                    category = brand_dict[b.lower()]
            elif len(list(set(brand_list))) >= 2:
                b_list = []
                for b in list(set(brand_list)):
                    if b.lower() in brand_dict and isinstance(brand_dict[b.lower()], str):
                        b_list.append(brand_dict[b.lower()])
                    if len(list(set(b_list))) == 1:
                        category = list(set(b_list))[0]
                    elif len(list(set(b_list))) >= 2:
                        category = "Shopping"

        line_js['category'] = category
        if category == "unknown":
            cnt += 1
        predict_f.write(json.dumps(line_js) + "\n")
print(cnt)

