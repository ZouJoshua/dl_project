#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 19-5-15 下午6:37
@File    : common_tools.py
@Desc    : 通用预处理方法
"""
import csv
import json
import os
from pyquery import PyQuery
from collections import OrderedDict
import re
import string
import emoji
import time


def read_data_from_csv_file(file):
    """
    读取csv文件,返回迭代器
    :param file:
    :return:
    """
    # data = list()
    if not os.path.exists(file):
        raise FileNotFoundError("【{}】文件未找到，请检查".format(file))
    print(">>>>> 正在读原始取数据文件：{}".format(file))
    with open(file, "r", encoding="utf-8-sig") as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        header = next(csv_reader)  # 读取第一行每一列的标题
        print(">>>>> 数据表头 Header: \n{}".format(header))
        for i, row in enumerate(csv_reader):   # 将 csv 文件中的数据保存到data中
            # data.append(row)
            # if i < 5:
            #     print(row)
            yield row


def read_json_format_file(file):
    """
    读取每行为json格式的文本
    :param file: 文件名
    :return: 每行文本
    """
    if not os.path.exists(file):
        raise FileNotFoundError("【{}】文件未找到，请检查".format(file))
    print(">>>>> 正在读原始取数据文件：{}".format(file))
    line_count = 0
    with open(file, 'r') as f:
        while True:
            _line = f.readline()
            line_count += 1
            if not _line:
                break
            else:
                line = json.loads(_line.strip())
                if line_count % 100000 == 0:
                    print(">>>>> 已读取{}行".format(line_count))
                yield line

def write_file(file, data, file_format='txt'):
    """
    写入文件，分普通文本格式、每行为json格式
    :param file: 文件名
    :param data: 写入数据
    :param file_format: 格式类型（txt、json）
    :return:
    """
    print(">>>>> 正在写入文件：{}".format(file))
    s = time.time()
    with open(file, 'w', encoding='utf-8') as f:
        if file_format == 'txt':
            for line in data:
                f.write(line)
                f.write('\n')
        elif file_format == 'json':
            for line in data:
                line_json = json.dumps(line, ensure_ascii=False)
                f.write(line_json)
                f.write('\n')
        else:
            raise Exception("检查数据格式")
    e = time.time()
    print('<<<<< 写文件耗时{}'.format(e -s))
    return



def write_json_format_file(source_data, file):
    """
    写每行为json格式的文件
    :param source_data:
    :param file:
    :return:
    """
    print(">>>>> 正在写入目标数据文件：{}".format(file))
    f = open(file, "w")
    _count = 0
    for _line in source_data:
        _count += 1
        if _count % 100000 == 0:
            print("<<<<< 已写入{}行".format(_count))
        if isinstance(_line, dict):
            line = json.dumps(_line, ensure_ascii=False)
            f.write(line + "\n")
        elif isinstance(_line, str):
            f.write(_line + "\n")
    f.close()


def read_txt_file(file):
    """
    读取txt格式的文本
    :param file:
    :return:
    """
    if not os.path.exists(file):
        raise FileNotFoundError("【{}】文件未找到，请检查".format(file))
    print(">>>>> 正在读原始数据文件：{}".format(file))
    with open(file, 'r') as f:
        while True:
            _line = f.readline()
            if not _line:
                break
            else:
                yield _line.strip()


def get_text_from_html(html):
    """
    从html中解析出文本内容
    :param html: html（str）
    :return: text（str）
    """
    return PyQuery(html).text().strip()


def split_text(text, lower=True, stop=None):
    """
    切分字符串（默认按空格切）
    :param text: 文本
    :param lower: 大小写（默认小写）
    :param stop: 停用词
    :return:
    """
    _text = text
    if lower:
        _text = text.lower()
    if stop:
        for i in stop:
            _text = _text.replace(i, "")
    word_list = _text.split(" ")
    return word_list

def dict_sort(result, limit_num=None):
    """
    字典排序, 返回有序字典
    :param result:
    :param limit_num:
    :return:
    """
    _result_sort = sorted(result.items(), key=lambda x: x[1], reverse=True)
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



def sort_by_value(d):
    items = d.items()
    backitems = [[v[1], v[0]] for v in items]
    backitems.sort(reverse=True)
    return [backitems[i][1] for i in range(0, len(backitems))]


def get_ngrams(sentence, n_gram=3):
    """
     # 将一句话转化为(uigram,bigram,trigram)后的字符串
    :param sentence: string. example:'w17314 w5521 w7729 w767 w10147 w111'
    :param n_gram:
    :return:string. example:'w17314 w17314w5521 w17314w5521w7729 w5521 w5521w7729 w5521w7729w767 w7729 w7729w767 w7729w767w10147 w767 w767w10147 w767w10147w111 w10147 w10147w111 w111'
    """
    result = list()
    word_list = sentence.split(" ")  # [sentence[i] for i in range(len(sentence))]
    unigram = ''
    bigram = ''
    trigram = ''
    fourgram = ''
    length_sentence = len(word_list)
    for i, word in enumerate(word_list):
        unigram = word                           # ui-gram
        word_i = unigram
        if n_gram >= 2 and i+2 <= length_sentence:  # bi-gram
            bigram = "".join(word_list[i:i+2])
            word_i = word_i + ' ' + bigram
        if n_gram >= 3 and i+3 <= length_sentence:  # tri-gram
            trigram = "".join(word_list[i:i+3])
            word_i = word_i + ' ' + trigram
        if n_gram >= 4 and i+4 <= length_sentence:  # four-gram
            fourgram = "".join(word_list[i:i+4])
            word_i = word_i + ' ' + fourgram
        if n_gram >= 5 and i+5 <= length_sentence:  # five-gram
            fivegram = "".join(word_list[i:i+5])
            word_i = word_i + ' ' + fivegram
        result.append(word_i)
    result = " ".join(result)
    return result


class CleanDoc(object):

    def __init__(self, text):
        self.text = self.clean_text(text)

    def clean_text(self, text):
        """
        清洗流程
        step1 -> 替换掉换行符、制表符等
        step2 -> 转小写
        step3 -> 清洗网址
        step4 -> 清洗邮箱
        step5 -> 清洗表情等非英文字符
        step6 -> 清洗标点符号、数字
        step7 -> 替换多个空格为一个空格
        :param text: 原始文本
        :return: 清洗后的文本
        """
        text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
        _text = text.lower()
        no_html = self.clean_html(_text)
        no_mail = self.clean_mail(no_html)
        no_emoji = self.remove_emoji(no_mail)
        no_symbol = self.remove_symbol(no_emoji)
        text = re.sub(r"\s+", " ", no_symbol)
        return text

    def remove_en_emoji(self, text):
        cleaned_text = ""
        for c in text:
            if (ord(c) >= 65 and ord(c) <= 126) or (ord(c) >= 32 and ord(c) <= 63):
                cleaned_text += c
        return cleaned_text

    def remove_emoji(self, text):
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

    def clean_html(self, text):
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

    def clean_mail(self, text):
        # 去除邮箱
        pattern = re.compile(r"\w+[-_.]*[a-zA-Z0-9]+@[a-zA-Z0-9]+\.[a-zA-Z]{2,3}")
        text = pattern.sub(" ", text)
        # mail_list = re.findall(pattern, text)
        # for mail in mail_list:
        #     text = text.replace(mail, " ")
        return text

    def remove_symbol_and_digits(self, text):
        del_symbol = string.punctuation + string.digits  # ASCII 标点符号，数字
        remove_punctuation_map = dict((ord(char), " ") for char in del_symbol)
        text = text.translate(remove_punctuation_map)  # 去掉ASCII 标点符号
        return text

    def remove_symbol(self, text):
        del_symbol = string.punctuation  # ASCII 标点符号
        remove_punctuation_map = dict((ord(char), " ") for char in del_symbol)
        text = text.translate(remove_punctuation_map)  # 去掉ASCII 标点符号
        return text


def clean_string(text):
    # 去除网址和邮箱
    text = text.replace("\n", " ").replace("\r", " ").replace("&#13;", " ").lower().strip()
    url_list = re.findall(r'http://[a-zA-Z0-9.?/&=:]*', text)
    for url in url_list:
        text = text.replace(url, " ")
    email_list = re.findall(r"[\w\d\.-_]+(?=\@)", text)
    for email in email_list:
        text = text.replace(email, " ")
    # 去除诡异的标点符号
    cleaned_text = ""
    for c in text:
        if (ord(c) >= 32 and ord(c) <= 126):
            cleaned_text += c
        else:
            cleaned_text += " "
    return cleaned_text



def clean_en_text(text):
    """
    清理英文数据,正则方式,去除标点符号等
    :param text:
    :return:
    """
    text = re.sub(r"[^A-Za-z0-9(),!?\'`]", " ", text)
    text = re.sub(r"\'s", " \'s", text)
    text = re.sub(r"\'ve", " \'ve", text)
    text = re.sub(r"n\'t", " n\'t", text)
    text = re.sub(r"\'re", " \'re", text)
    text = re.sub(r"\'d", " \'d", text)
    text = re.sub(r"\'ll", " \'ll", text)
    text = re.sub(r",", " , ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\(", " \( ", text)
    text = re.sub(r"\)", " \) ", text)
    text = re.sub(r"\?", " \? ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip().lower()


def clean_zh_text(text):
    """
    清理中文文本，正则方式
    :param text:
    :return:
    """
    text = re.sub(r'["\'` ?!【】\[\]./%：:&()=，,<>+_；;\-*]+', " ", text)
    return text

def clean(file_path):
    """
    清理文本, 然后利用清理后的文本进行训练
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines_clean = []
        for line in lines:
            line_list = line.split('__label__')
            lines_clean.append(clean_en_text(line_list[0]) + ' __label__' + line_list[1])

    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines_clean)



def clean_to_list(text):
    """
    清理英文数据，正则方式，切词后的列表
    :param text:
    :return:
    """
    text = str(text)
    text = text.lower()
    # 清理数据
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"-", " - ", text)
    text = re.sub(r"=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.split()
    return text



