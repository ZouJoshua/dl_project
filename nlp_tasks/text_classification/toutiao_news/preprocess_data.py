#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 4/30/20 2:13 PM
@File    : preprocess_data.py
@Desc    : 头条数据预处理

"""

import os
import json
import tqdm
import random
import matplotlib.pyplot as plt
from preprocess.common_tools import read_json_format_file
from setting import DATA_PATH


def trans_text_to_json_line(ori_file, out_file):
    """
    将原始文本格式转化为每行为json格式文本
    {"item_id": xxx, "c_id":xxx,"c_name":xxx, "label":xxx, "title":xxx, "keywords":xxx}
    :param ori_file: 原始文件
    :param out_file: 输出文件
    :return:
    """

    cid2name = {
        "100": "民生故事",
        "101": "文化",
        "102": "娱乐",
        "103": "体育",
        "104": "财经",
        "105": "时政",
        "106": "房产",
        "107": "汽车",
        "108": "教育",
        "109": "科技",
        "110": "军事",
        "111": "宗教",
        "112": "旅游",
        "113": "国际",
        "114": "证券",
        "115": "农业",
        "116": "电竞游戏"
    }



    with open(ori_file, "r", encoding="utf-8") as f1, open(out_file, "w", encoding="utf-8") as f2:
        for _line in tqdm.tqdm(f1.readlines()):
            out = dict()
            line = _line.strip().split("_!_")
            if len(line) == 5:
                # 6552368441838272771_!_101_!_news_culture_!_发酵床的垫料种类有哪些？哪种更好？_!_
                out["item_id"], out["c_id"], out["c_name"], out["label"], out["title"], out["keywords"] = line[0], int(line[1]), line[2], cid2name[line[1]], line[3], line[4]
                f2.write(json.dumps(out, ensure_ascii=False)+"\n")
            else:
                print(line)


class AnalysisData(object):

    def analysis_data(self, corpus_file):
        """
        分析数据分布情况,
        :param corpus_file:
        :return:
        """
        label_count = dict()
        lines = read_json_format_file(corpus_file)
        data_len = list()
        for line in lines:
            label = line["label"]
            title = line["title"]
            data_len.append(len(title))
            if label in label_count:
                label_count[label] += 1
            else:
                label_count[label] = 1

        print("数据 label 分布情况:")
        print(json.dumps(label_count, ensure_ascii=False, indent=4))
        self.plot_text_length(data_len)



    def plot_text_length(self, all_text_len, save_path=None):
        """
        绘制文本长度分布
        :param all_text_len: [22,33,34,...]
        :param save_path: 图片保存路径
        :return:
        """
        plt.figure()
        plt.title("Length dist of text")
        plt.xlabel("Length")
        plt.ylabel("Count")
        _, _, all_data = plt.hist(all_text_len, bins=100, normed=1, alpha=.2, color="b")
        plt.show()
        if save_path:
            plt.savefig(save_path)

    def get_data_example_to_file(self, file, out_file):
        """
        从原始数据中每个样本抽取1000到文件
        :param file: 原始文件
        :param out_file: example文件
        :return:
        """
        lines = read_json_format_file(file)
        all_data = dict()

        for line in lines:
            if not line:
                continue
            title = line["title"]
            label = line["label"]
            if label in all_data:
                all_data[label].append(title)
            else:
                all_data[label] = list()
                all_data[label].append(title)

        with open(out_file, "w", encoding="utf-8") as f:
            for k, v in all_data.items():
                random.shuffle(v)
                i = 0
                for text in v:
                    out = dict()
                    if len(text) > 5 and len(text) < 50:
                        out["label"] = k
                        out["text"] = text
                        f.write(json.dumps(out, ensure_ascii=False) + "\n")
                        i += 1
                        if i >= 1000:
                            break


def main():
    toutiao_data_path = os.path.join(DATA_PATH, "corpus", "toutiao_news")
    file = os.path.join(toutiao_data_path, "toutiao_cat_data.txt")
    outfile = os.path.join(toutiao_data_path, "toutiao_news_corpus")
    sample_file = os.path.join(toutiao_data_path, "toutiao_news_sample")
    # trans_text_to_json_line(file, outfile)
    # AnalysisData.analysis_data(outfile)
    AnalysisData().get_data_example_to_file(outfile, sample_file)


if __name__ == "__main__":
    main()