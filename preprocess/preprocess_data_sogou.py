#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 12/17/19 12:54 PM
@File    : preprocess_data_sogou.py
@Desc    : 搜狗新闻数据处理

"""


import os
import re
import json
from urllib.parse import urlparse
from sklearn.model_selection import StratifiedKFold
import jieba
from string import punctuation
from setting import DATA_PATH
from preprocess.preprocess_tools import read_json_format_file


class PreCorpus(object):

    def __init__(self, ori_file_dir, output_path, is_xml_file=None):
        self.corpus_file = os.path.join(output_path, "sogou_corpus")
        self.corpus_file_with_label = os.path.join(output_path, "sogou_corpus_with_label")
        self.url_file_without_label = os.path.join(output_path, "url_without_label")
        self.domain_count_file = os.path.join(output_path, "url_domain_count.txt")
        self.label_count_file = os.path.join(output_path, "url_label_count.txt")
        if not os.path.exists(self.corpus_file):
            if is_xml_file:
                ori_file = os.path.join(ori_file_dir, "news_sohusite_xml.dat")
                if os.path.exists(ori_file):
                    self.extract_docs_from_xml_file(ori_file, self.corpus_file)
                else:
                    raise Exception("Ori_file {} not found".format("news_sohusite_xml.dat"))
            else:
                self.extract_docs(ori_file_dir, self.corpus_file)
        else:
            print("Find corpus file in {}".format(self.corpus_file))

        if not os.path.exists(self.corpus_file_with_label):
            self.analysis_url(self.corpus_file, self.domain_count_file, self.label_count_file)



    def listdir(self, path, list_name):
        """
        生成原始语料文件夹下文件列表
        :param path:
        :param list_name:
        :return:
        """
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.isdir(file_path):
                    self.listdir(file_path, list_name)
            else:
                    list_name.append(file_path)

    def extract_docs(self, file_dir, outfile):
        """
        抽取文本内容到文件
        :return:
        """
        # 获取所有语料
        list_name = list()
        self.listdir(file_dir, list_name)
        i = 0
        for path in list_name:
            print(path)
            self.extract_docs_from_xml_file(path, outfile)

            # i += 1
            # if i == 1:
            #     break


    def extract_docs_from_xml_file(self, xml_file, outfile):
        # 字符数小于这个数目的content将不被保存
        threh = 30
        with open(xml_file, 'rb') as f, open(outfile, "a+", encoding="utf-8") as o_f:
            text = f.read().decode("gb18030", "ignore")
            # print(text)
            pattern_doc = re.compile(r'<doc>(.*?)</doc>', re.S)

            # 正则匹配出url和content
            pattern_url = re.compile(r'<url>(.*?)</url>', re.S)
            pattern_title = re.compile(r'<contenttitle>(.*?)</contenttitle>', re.S)
            pattern_content = re.compile(r'<content>(.*?)</content>', re.S)
            docs = pattern_doc.findall(text)
            for doc_string in docs:
                out = dict()
                url = pattern_url.search(doc_string)
                title = pattern_title.search(doc_string)
                content = pattern_content.search(doc_string)
                out["title"] = title.group(1) if title else ""
                out["content"] = content.group(1) if content else ""
                out["url"] = url.group(1) if url else ""
                # 把所有内容小于30字符的文本全部过滤掉
                if len(content[0]) >= threh:
                    o_f.write(json.dumps(out, ensure_ascii=False) + "\n")
                else:
                    continue

    def extract_label(self, netloc, url2label_dict):
        """
        从url链接抽取label
        :param netloc:
        :param url2label_dict:
        :return:
        """
        labels = netloc.split(".")
        label = ""
        if netloc == "news.china.com":
            label = "社会"
        for la in labels:
            if la in url2label_dict:
                label = url2label_dict[la]

        return label


    def analysis_url(self, file, outfile1, outfile2):
        """
        分析url,查看可提供的label
        :param file:
        :param outfile1:
        :param outfile2:
        :return:
        """
        lines = read_json_format_file(file)
        scheme_dict = dict()
        domain_dict = dict()
        label_dict = dict()
        for line in lines:
            url = line["url"]
            url_parse = urlparse(url)
            scheme = url_parse.scheme
            if scheme not in scheme_dict:
                scheme_dict[scheme] = 1
            else:
                scheme_dict[scheme] += 1
            netloc = url_parse.netloc
            if netloc not in domain_dict:
                domain_dict[netloc] = 1
            else:
                domain_dict[netloc] += 1
            labels = netloc.split(".")
            for label in labels:
                if label not in label_dict:
                    label_dict[label] = 1
                else:
                    label_dict[label] += 1

        print(scheme_dict)
        print(domain_dict)
        print(label_dict)
        with open(outfile1, "w") as f:
            f.writelines(json.dumps(domain_dict, indent=4))
        with open(outfile2, "w") as f:
            f.writelines(json.dumps(label_dict, indent=4))



    def pre_corpus(self, ori_file, label_file, corpus_file, url_file):
        """
        将语料处理为带label
        :param ori_file:
        :param label_file:
        :param corpus_file:
        :param url_file:
        :return:
        """
        lines = read_json_format_file(ori_file)
        with open(label_file, "r", encoding="utf-8") as f:
            url2label_dict = json.load(f)
        labels = dict()
        with open(corpus_file, "w", encoding="utf-8") as cf, open(url_file, "w", encoding="utf-8") as uf:
            for line in lines:
                url = line["url"]
                url_parse = urlparse(url)
                netloc = url_parse.netloc
                label = self.extract_label(netloc, url2label_dict)
                line["category"] = label
                if label != "":
                    cf.write(json.dumps(line, ensure_ascii=False) + "\n")
                    if label not in labels:
                        labels[label] = 1
                    else:
                        labels[label] += 1
                else:
                    uf.write(url+"\n")
                del url
            print(labels)



class SplitData2tsv(object):
    """
    切分训练集\测试集\验证集
    """
    def __init__(self, corpus_file, out_dir):
        self.f1 = corpus_file
        self.train_file = os.path.join(out_dir, 'content_train.tsv')
        self.dev_file = os.path.join(out_dir, 'content_dev.tsv')
        self.test_file = os.path.join(out_dir, 'content_test.tsv')
        self.get_category_corpus_file()

    def get_data(self):
        X = list()
        Y = list()
        _count = dict()
        for line in read_json_format_file(self.f1):
            if line:
                # result = self._preline(line)
                x, y = self.preline(line)
                if len(x) > 30:
                    if y in _count.keys():
                        if _count[y] > 50000:
                            continue
                        else:
                            _count[y] += 1
                            X.append(x)
                            Y.append(y)
                    else:
                        _count[y] = 1
                        X.append(x)
                        Y.append(y)
                else:
                    continue
        return X, Y

    def clean_zh_text(self, text):
        add_punc = '，。、【 】 “”：；（）《》‘’{}？！⑦()、%^>℃：.”“^-——=&#@￥\n「」…『』'
        all_punc = punctuation + add_punc
        seg_list = jieba.cut(text, cut_all=False)
        no_punc = [w for w in seg_list if w not in all_punc]
        clean_text = " ".join(no_punc)
        return clean_text


    def preline(self, line_json):
        title = line_json["title"]
        content = line_json["content"]
        dataY = line_json["category"]
        text = self.clean_zh_text(content)
        return text, dataY


    def _label_count(self, label_list):
        label_count = dict()
        for i in label_list:
            if label_count.get(i, None) is not None:
                label_count[i] += 1
            else:
                label_count[i] = 1
        return label_count

    def stratified_sampling(self, x, y, valid_portion):
        """
        按标签类别个数分层切分训练集和验证集
        :param self:
        :param x:
        :param y:
        :param valid_portion:
        :return:
        """
        skf = StratifiedKFold(n_splits=int(1 / valid_portion))
        i = 0
        train = None
        dev = None
        for train_index, test_index in skf.split(x, y):
            train_label_count = self._label_count([y[i] for i in train_index])
            test_label_count = self._label_count([y[j] for j in test_index])
            print("train_label_count: {}".format(json.dumps(train_label_count, indent=4, ensure_ascii=False)))
            print("test_label_count: {}".format(json.dumps(test_label_count, indent=4, ensure_ascii=False)))
            train = [x[i] + "\t__label__" + y[i] for i in train_index]
            dev = [x[j] + "\t__label__" + y[j] for j in test_index]
            i += 1
            if i < 2:
                break

        return train, dev

    def write_tvs_file(self, data, file):
        print(">>>>> 正在写入文件")
        with open(file, "w") as f:
            for line in data:
                f.write(line + "\n")
        print("<<<<< 已写入到文件：{}".format(file))


    def get_category_corpus_file(self):
        print(">>>>> 正在处理训练语料")
        X, Y = self.get_data()
        train, dev = self.stratified_sampling(X, Y, 0.2)
        self.write_tvs_file(train, self.train_file)
        self.write_tvs_file(dev, self.dev_file)
        self.write_tvs_file(dev, self.test_file)




def main():
    ori_file_dir = "/data/common/sogou_data"
    data_path = os.path.join(DATA_PATH, "sogou")
    PreCorpus(ori_file_dir, data_path, is_xml_file=True)
    label_file = os.path.join(data_path, "url2label.txt")
    # extract_docs(ori_file_dir, outfile)
    # extract_docs_from_xml_file(xml_file, outfile)
    # extract_label(outfile)
    # analysis_url(outfile, domain_count_file, label_count_file)
    # pre_corpus(outfile, label_file, outfile1, outfile2)
    # SplitData2tsv(outfile1, data_path)





if __name__ == "__main__":
    main()

