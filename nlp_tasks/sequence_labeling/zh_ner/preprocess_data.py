#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 4/20/20 9:06 PM
@File    : preprocess_data.py
@Desc    : 数据处理

"""



def merge_text_and_label(source_text_file, source_label_file, data_file):
    """
    合并文本和标签数据到一个文件
    :param source_data_file:
    :param source_label_file:
    :param data_file:
    :return:
    """
    with open(data_file, 'w', encoding='utf-8') as f0:
        with open(source_text_file, encoding='utf-8') as f1, open(source_label_file, encoding='utf-8') as f2:
            lines_1 = f1.readlines()
            lines_2 = f2.readlines()
            for i, j in zip(lines_1, lines_2):
                for m, n in zip(i.strip().split(), j.strip().split()):
                    f0.write(m + ' ' + n + '\n')
                f0.write('\n')

import tqdm

def read_data(file, mode="origin"):
    """
    读取数据
    加载数据集,每行一个汉子和一个标记,句子和句子之间以空格分割
    :return: 返回句子集合
    """
    data = []
    sent_, tag_ = [], []
    with open(file, "r", encoding="utf-8") as f:
        # 将数据集全部加载到内存
        for i, _line in enumerate(tqdm.tqdm(f, desc="Loading {} dataset".format(mode))):
            if _line:
                if _line != '\n':
                    [char, label] = _line.strip().split()
                    sent_.append(char)
                    tag_.append(label)
                else:
                    data.append([sent_, tag_])
                    sent_, tag_ = [], []
            else:
                print("error with line{}:{}".format(i, _line))
                continue
    return data


def update_tag_scheme(source_data, tag_scheme):
    """
    更改为指定编码
    :param source_data:[([w1,w2...], [bio1,bio2]),([],[])]
    :param tag_scheme: "bio" / "bioes"
    :return:
    """

    for i, data in enumerate(source_data):
        tags = data[-1]
        if not check_bio(tags):
            s_str = "\n".join(" ".join(w) for w in data)
            raise Exception("输入的句子应为BIO编码标注格式,请检查第{}句子：\n{}".format(i, s_str))

        if tag_scheme == "BIO":
            continue

        if tag_scheme == "BIOES":
            new_tags = bio_to_bioes(tags)
            data[-1] = new_tags
            # if i < 3:
            #     print(data)
        else:
            raise Exception("非法目标编码格式")



def check_bio(tags):
    """
    检测输入的编码是否为bio编码格式
    1)编码不是bio编码
    2)第一个编码是I
    3)当前编码不是B,前一个编码不是O
    :param tags:
    :return:
    """
    for i, tag in enumerate(tags):
        if tag == "O":
            continue
        tag_list = tag.split("-")
        if len(tag_list) != 2 or tag_list[0] not in set(["B", "I"]):
            # 非法编码
            return False
        if tag_list[0] == "B":
            continue
        elif i == 0 or tags[i-1] == "O":
            #如果第一个位置不是B或者当前编码不是B并且前一个编码O.则全部转换为B
            tags[i] = "B" + tag[1:]
        elif tags[i-1][1:] == tag[1:]:
            # 如果当前编码的后面类型斌吗域tags中的前一个编码中后面类型编码相同则跳过
            continue
        else:
            # 如果类型不一致,则从新冲B开始编码
            tags[i] = "B" + tag[1:]
    return True


def bio_to_bioes(tags):
    """
    把bio编码标注格式转换为bioes编码
    :param tags: 新的tags
    :return:
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == "O":
            new_tags.append(tag)
        elif tag.split("-")[0] == "B":
            # 如果tag是以B开头
            # 首先，如果当前tag不是最后一个，并且紧跟着的后一个是I
            if (i+1) < len(tags) and tags[i+1].split("-")[0] == "I":
                new_tags.append(tag)
            else:
                # 如果最优一个或者紧跟着的后一个不是Ｉ，那么表示单字，需要把Ｂ换成Ｓ表示单字
                new_tags.append(tag.replace("B-", "S-"))
        elif tag.split("-")[0] == "I":
            # 如果tag是以Ｉ开头，那么需要进行判断
            # 首先，如果当前tag不是最后一个，并且紧跟着的一个是Ｉ
            if (i+1) < len(tags) and tags[i+1].split("-")[0] == "I":
                new_tags.append(tag)
            else:
                # 如果是最后一个，或者后一个不是I开头的，那么就表示一个词的结尾，就把I换成E表示一个词结尾
                new_tags.append(tag.replace("I-", "E-"))
        else:
            raise Exception("非法编码标注格式")
    return new_tags

def bioes_to_bio(tags):
    """
    BIOES->BIO
    :param tags:
    :return:
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.split('-')[0] == "B":
            new_tags.append(tag)
        elif tag.split('-')[0] == "I":
            new_tags.append(tag)
        elif tag.split('-')[0] == "S":
            new_tags.append(tag.replace('S-', 'B-'))
        elif tag.split('-')[0] == "E":
            new_tags.append(tag.replace('E-', 'I-'))
        elif tag.split('-')[0] == "O":
            new_tags.append(tag)
        else:
            raise Exception('非法编码格式')
    return new_tags



def merge_main():
    file_type = ["train", "test"]
    for i in file_type:
        data_file = "/data/work/dl_project/data/corpus/zh_ner/{}.txt".format(i)
        text_file = "/data/work/dl_project/data/corpus/zh_ner/{}_source.txt".format(i)
        label_file = "/data/work/dl_project/data/corpus/zh_ner/{}_label.txt".format(i)
        merge_text_and_label(text_file, label_file, data_file)



if __name__ == "__main__":
    # merge_main()

    file = "/data/work/dl_project/data/corpus/zh_ner/test.txt"
    data = read_data(file)
    # data = load_sentences(file)
    print(data[:2])
    update_tag_scheme(data, tag_scheme="BIOES")

    print(data[:2])



