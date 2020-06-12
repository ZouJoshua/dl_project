#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 2020/6/11 6:01 下午
@File    : data_preprocessor.py
@Desc    : 数据预处理

"""

import os
import re
import string
import json
from collections import OrderedDict

import numpy as np
import jieba
import emoji
from sklearn.model_selection import StratifiedKFold


class PreCorpus(object):
    """
    处理为模型训练数据，包括数据清洗和数据分层
    """
    def __init__(self, corpus_file, out_dir):
        if not os.path.exists(corpus_file):
            raise FileNotFoundError
        self.corpus_file = corpus_file
        self.train_file = os.path.join(out_dir, "train.txt")
        self.dev_file = os.path.join(out_dir, "validate.txt")
        self.test_file = os.path.join(out_dir, "test.txt")
        # self.analysis_label_dist()
        self.get_category_corpus_file()

    def analysis_label_dist(self):
        """
        统计label分布
        """
        category_count = dict()
        for line in self.read_json_format_file(self.corpus_file):
            _, (top_category, _, _) = self._preline(line)
            self._label_dist_count(level1=top_category, dist_count=category_count)

        sorted_category_count = self.dict_sort(category_count)
        print(json.dumps(sorted_category_count, ensure_ascii=False, indent=4))

    def get_category_corpus_file(self):
        print(">>>>> preprocess corpus")
        X, Y = self.get_clean_data(category_level=1)
        train, dev = self.stratified_sampling(X, Y, 0.2)
        self.write_txt_file(train, self.train_file)
        self.write_txt_file(dev, self.dev_file)
        self.write_txt_file(dev, self.test_file)

    def get_clean_data(self, category_level=1):
        X = list()
        Y = list()
        _count = dict()
        for line in self.read_json_format_file(self.corpus_file):
            if line:
                (_id, channel, title, content), y = self._preline(line)
                label = y[category_level - 1]
                clean_title_obj = CleanDoc(title)
                clean_content_obj = CleanDoc(content)
                # 获取title和content的char特征
                # char_feature = " ".join(list(clean_title_obj.char_feature)) + " ### " + " ".join(list(clean_content_obj.char_feature))
                # 获取title和content的token特征
                # token_feature = " ".join(list(clean_title_obj.token_feature)) + " ### " + " ".join(list(clean_content_obj.token_feature))
                token_feature = " ".join(list(clean_title_obj.token_feature)) + " " + " ".join(list(clean_content_obj.token_feature))

                if len(content) > 30:
                    if label in _count.keys():
                        if _count[label] > 30000:
                            continue
                        else:
                            _count[label] += 1
                            # X.append(token_feature + "\t" + char_feature + "\t" + channel)
                            X.append(token_feature)
                            Y.append(label)
                    else:
                        _count[label] = 1
                        # X.append(token_feature + "\t" + char_feature + "\t" + channel)
                        X.append(token_feature)
                        Y.append(label)
                else:
                    continue
        return X, Y

    def stratified_sampling(self, x, y, valid_portion):
        """
        按标签类别个数分层切分训练集和验证集
        :param x:
        :param y:
        :param valid_portion:
        :return:
        """
        skf = StratifiedKFold(n_splits=int(1 / valid_portion))
        train = None
        dev = None

        index = [(train_index, test_index) for train_index, test_index in skf.split(x, y)]

        train_label_count = self._label_count([y[i] for i in index[0][0]])
        test_label_count = self._label_count([y[j] for j in index[0][1]])
        print("train_label_count: {}".format(json.dumps(train_label_count, indent=4, ensure_ascii=False)))
        print("test_label_count: {}".format(json.dumps(test_label_count, indent=4, ensure_ascii=False)))
        # train = [y[i] + "\t" + x[i] for i in index[0][0]]
        # dev = [y[j] + "\t" + x[j] for j in index[0][1]]
        train = [y[i] + "\t__label__" + x[i] for i in index[0][0]]
        dev = [y[j] + "\t__label__" + x[j] for j in index[0][1]]

        return train, dev

    def _label_dist_count(self, level1=None, level2=None, level3=None, dist_count=None):
        """
        统计标签分布计算
        :param level1:一级标签
        :param level2:二级标签
        :param level3:三级标签
        :param dist_count:标签分布字典
        :return:
        """
        if level1:
            if level1 in dist_count:
                dist_count[level1]["count"] += 1
            else:
                dist_count[level1] = dict()
                dist_count[level1]["count"] = 1
            if level2:
                if level2 in dist_count[level1]:
                    dist_count[level1][level2]["count"] += 1
                else:
                    dist_count[level1][level2] = dict()
                    dist_count[level1][level2]["count"] = 1
                if level3:
                    if level3 in dist_count[level1][level2]:
                        dist_count[level1][level2][level3] += 1
                    else:
                        dist_count[level1][level2][level3] = 1

    @staticmethod
    def dict_sort(result, limit_num=None):
        """
        字典排序, 返回有序字典
        :param result:
        :param limit_num:
        :return:
        """
        _result_sort = sorted(result.items(), key=lambda x: x[1]["count"], reverse=True)
        result_sort = OrderedDict()

        count_limit = 0
        domain_count = 0
        for i in _result_sort:
            if limit_num:
                if i[1] > limit_num:
                    result_sort[i[0]] = i[1]
                    domain_count += 1
                    count_limit += i[1]
            else:
                result_sort[i[0]] = i[1]
        return result_sort


    @staticmethod
    def _preline(line):
        """
        处理文件行（dict格式）
        """
        article_id = line.get("articleid", "")
        channel = line.get("channel", "")
        title = line.get("title", "")
        content = line.get("content", "")
        tagging_data = eval(line.get("tagscoredata", ""))
        category_data = tagging_data["categoryList"][0]["data"]
        top_category = ""
        sub_category = ""
        third_category = ""
        for item in category_data:
            if item["level"] == 1:
                top_category = str(item["name"])
            if item["level"] == 2:
                if "name" in item:
                    sub_category = str(item["name"].split("_")[-1])
            if item["level"] == 3:
                if "name" in item:
                    third_category = str(item["name"].split("_")[-1])
        return (article_id, channel, title, content), (top_category, sub_category, third_category)

    def _label_count(self, label_list):
        label_count = dict()
        for i in label_list:
            if label_count.get(i, None) is not None:
                label_count[i] += 1
            else:
                label_count[i] = 1
        return label_count

    @staticmethod
    def read_json_format_file(file):
        """
        读取每行为json格式的文本
        :param file: 文件名
        :return: 每行文本
        """
        if not os.path.exists(file):
            raise FileNotFoundError("file {} not found.".format(file))
        print(">>>>> reading file：{}".format(file))
        line_count = 0
        with open(file, 'r') as f:
            while True:
                _line = f.readline()
                line_count += 1
                if not _line:
                    break
                else:
                    line = json.loads(_line.strip())
                    # line = eval(_line.strip())
                    if line_count % 100000 == 0:
                        print(">>>>> read {} lines.".format(line_count))
                    yield line

    def write_txt_file(self, data, file):
        """
        写数据到文件
        """
        print(">>>>> start writing file")
        with open(file, "w") as f:
            for line in data:
                f.write(line + "\n")
        print("<<<<< write down：{}".format(file))


class CleanDoc(object):
    """
    文本清洗并进行特征抽取
    """
    def __init__(self, sentence, language="cn"):
        if language == "cn":
            # text = self.clean_cn_text(sentence)
            clean_text = self.clean_cn_text_by_third_party(sentence)
            self.char_feature = self.get_cn_char_feature(clean_text)
            self.token_feature = self.get_cn_token_feature(clean_text)
        else:
            self.text = self.clean_en_text(sentence)

    def get_cn_char_feature(self, text):
        """
        按字切分句子,去除非中文字符及标点，获取char级别特征
        todo:针对中文字级别处理（针对英文、数字等符号特殊处理）
        :param text:
        :return:
        """
        # print("splitting chinese char")
        seg_list = list()
        none_chinese = ""
        for char in text:
            if self.is_chinese(char) is False:
                if char in self.punc_list:
                    continue
                none_chinese += char
            else:
                if none_chinese:
                    seg_list.append(none_chinese)
                    none_chinese = ""
                seg_list.append(char)
        if not seg_list:
            seg_list = list()
        return seg_list

    def get_cn_token_feature(self, text):
        """
        按结巴分词.去除标点，获取文本token级别特征
        :param text:
        :return:
        """
        # 精确模式
        seg_list = jieba.cut(text, cut_all=False)
        words = [w for w in seg_list if w not in self.punc_list]
        if not words:
            words = list()

        return words

    def clean_cn_text(self, sentence):
        """
        中文文本清洗流程
        step1 -> 替换掉换行符、制表符等
        step2 -> 清洗网址
        step3 -> 清洗邮箱
        step4 -> 清洗表情等非英文字符
        step5 -> 替换多个空格为一个空格
        :param sentence: 原始文本
        :return: 清洗后的文本
        """

        _text = sentence.replace('\u2028', '').replace('\n', '').replace('\t', '')
        re_h = re.compile('<(/?\w+|!--|!DOCTYPE|\?xml)[^>]*>')
        _text = re_h.sub('', _text)  # html处理
        no_html = self.clean_url(_text)
        no_mail = self.clean_mail(no_html)
        no_emoji = self.remove_emoji(no_mail)
        _text = re.sub(r"\s+", " ", no_emoji)
        return sentence


    def clean_cn_text_by_third_party(self, sentence):
        """
        用第三方库清洗中文文本
        """
        from harvesttext import HarvestText
        ht_obj = HarvestText()
        # 去掉微博的@，表情符；网址；email；html代码中的一类的特殊字符等
        _text = sentence.replace('\u2028', '').replace('\n', '').replace('\t', '')
        re_h = re.compile('<(/?\w+|!--|!DOCTYPE|\?xml)[^>]*>')
        _text = re_h.sub('', _text)  # html处理
        clean_text = ht_obj.clean_text(_text)
        return clean_text

    def clean_en_text(self, sentence):
        """
        英文文本清洗流程
        step1 -> 替换掉换行符、制表符等
        step2 -> 转小写
        step3 -> 清洗网址
        step4 -> 清洗邮箱
        step5 -> 清洗表情等非英文字符
        step6 -> 清洗标点符号、数字
        step7 -> 替换多个空格为一个空格
        :param sentence: 原始文本
        :return: 清洗后的文本
        """
        text = sentence.replace("\r", " ").replace("\n", " ").replace("\t", " ")
        _text = text.lower()
        no_html = self.clean_url(_text)
        no_mail = self.clean_mail(no_html)
        no_emoji = self.remove_emoji(no_mail)
        no_symbol = self.remove_symbol(no_emoji)
        text = re.sub(r"\s+", " ", no_symbol)
        return text

    @property
    def punc_list(self):
        add_punc = '，。、【 】 “”：；（）《》‘’{}？！⑦()、%^>℃：”“^-——=&#@￥\n「」…『』\u3000\xa0'
        return string.punctuation + add_punc

    @staticmethod
    def remove_en_emoji(text):
        """
        去除英文表情符号
        :param text:
        :return:
        """
        cleaned_text = ""
        for c in text:
            if (ord(c) >= 65 and ord(c) <= 126) or (ord(c) >= 32 and ord(c) <= 63):
                cleaned_text += c
        return cleaned_text

    @staticmethod
    def remove_emoji(text):
        """
        去除表情符号
        :param text:
        :return:
        """
        token_list = text.replace("¡", "").replace("¿", "").split(" ")
        em_str = r":.*?:"
        em_p = re.compile(em_str, flags=0)
        clean_token = list()
        for token in token_list:
            em = emoji.demojize(token)
            emj = em_p.search(em)
            if emj:
                _e = emj.group(0)
                # print(_e)
            else:
                clean_token.append(token)
        cleaned_text = " ".join(clean_token)
        return cleaned_text.strip()

    @staticmethod
    def is_chinese(uchar):
        """
        判断一个unicode是否是汉字
        """
        if (uchar >= u'\u4e00') and (uchar <= u'\u9fa5'):
            return True
        else:
            return False

    @staticmethod
    def clean_url(text):
        """
        去除网址
        1.完整网址https开头的
        2.没有协议头的网址，www开头的
        :param text:
        :return:
        """

        pattern = re.compile(
            r'(?:(?:https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|])|(?:www\.[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|])')
        # pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-zA-Z][0-9a-zA-Z]))+')
        # url_list = re.findall(pattern, text)
        # for url in url_list:
        #     text = text.replace(url, " ")
        text = pattern.sub("", text)
        return text.replace("( )", " ")

    @staticmethod
    def clean_mail(text):
        """
        去除邮箱
        :param text:
        :return:
        """
        pattern = re.compile(r"\w+[-_.]*[a-zA-Z0-9]+@[a-zA-Z0-9]+\.[a-zA-Z]{2,3}")
        text = pattern.sub(" ", text)
        # mail_list = re.findall(pattern, text)
        # for mail in mail_list:
        #     text = text.replace(mail, " ")
        return text

    @staticmethod
    def remove_symbol_and_digits(text):
        """
        去除标点符号和数字
        :param text:
        :return:
        """
        del_symbol = string.punctuation + string.digits  # ASCII 标点符号，数字
        remove_punctuation_map = dict((ord(char), " ") for char in del_symbol)
        text = text.translate(remove_punctuation_map)  # 去掉ASCII 标点符号
        return text

    @staticmethod
    def remove_symbol(text):
        """
        去除标点符号
        :param text:
        :return:
        """
        del_symbol = string.punctuation  # ASCII 标点符号
        remove_punctuation_map = dict((ord(char), " ") for char in del_symbol)
        text = text.translate(remove_punctuation_map)  # 去掉ASCII 标点符号
        return text

def pre_process():
    data_file = "/Users/vivo/work/data/corpus_data"
    from setting import DATA_PATH
    out_dir = os.path.join(DATA_PATH, "corpus", "vivo_news")
    PreCorpus(data_file, out_dir)

def clean_text_demo():
    title = "搞笑GIF：聪明人一看就知道怎么做到的！"
    content = "媳妇我不练了，不练了！ 哎呀，悲催了 大哥，看着点路 一个月了原来我的篮球在这里？ 不懂艺术的人们 怪我喽？ 白猫：你干嘛，干嘛，干嘛…… 真是人间美味啊！ 妹子，还以为你衣服穿反了 你会手指打结吗？ 美女，你这喉结过分了啊 聪明人一看就知道怎么做到的！ 面这两人也是躺着中枪 辣条都让你们玩涨价了！ 妹子砸车的威武霸气，真是惹不起 这手势太吓人了，我要下车！ 妹子都进电梯了，你还拉她出来干啥 妹子这么勤快呢，桌子很干净了，休息会吧 看得出，教练确实很受打击 去车展的有几个是卖车的，我感觉更多的人是看美女的啊！ 讲道理，现在什么都是智能，连沙袋都变成智能的了 都捏不碎，老丈人表示不服，非要亲自示范一下 跟老师们一起吃饭，感觉很开心！ 爱她就背起她走回家，妥妥的真爱呐 对女孩子来说，头发就是的她的命 长不大的妹子，连鬼你都戏弄 好玩 大哥你配合得太专业了啊 知道篮球场汤为什么总是会有这么多大神呢？你这里比较有市场需求 兄弟，现在的日子过的是越来越好了啊 啊！我的眼睛"
    c_title = CleanDoc(title).char_feature
    print(c_title)
    c_content = CleanDoc(content).token_feature
    print(c_content)


if __name__ == "__main__":
    pre_process()
    # clean_text_demo()