#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 5/15/20 5:40 PM
@File    : data_analysis.py
@Desc    : 数据分析

"""


import csv
import json
import os
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict


def read_data_from_csv_file(file):
    # data = list()
    with open(file, "r", encoding="utf-8-sig") as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        header = next(csv_reader)  # 读取第一行每一列的标题
        print("data header: {}".format(header))
        for i, row in enumerate(tqdm(csv_reader, desc="Reading Files of {}".format(file))):   # 将 csv 文件中的数据保存到data中
            # data.append(row)
            # if i < 5:
            #     print(row)
            yield row
    # count = len(data)
    # print("data with {} lines".format(count))

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


def write_ad_analysis_details_to_file(file, outfile):
    print("start writing ad analysis details to file: {}".format(outfile))
    ad_info = dict()
    ad_info["product_info"] = dict()
    ad_info["advertiser_info"] = dict()
    _count = 0

    for line in read_data_from_csv_file(file):
        _count += 1
        if _count < 6:
            print(line)
        # else:
        #     break
        # 产品信息
        if line[3] in ad_info["product_info"]:
            ad_info["product_info"][line[3]]["_count"] += 1
            if line[2] in ad_info["product_info"][line[3]]:
                ad_info["product_info"][line[3]][line[2]] += 1
            else:
                ad_info["product_info"][line[3]][line[2]] = 1
        else:
            ad_info["product_info"][line[3]] = dict()
            ad_info["product_info"][line[3]]["_count"] = 1
            ad_info["product_info"][line[3]][line[2]] = 1

        # 广告主信息
        if line[5] in ad_info["advertiser_info"]:
            ad_info["advertiser_info"][line[5]]["_count"] += 1
            if line[4] in ad_info["advertiser_info"][line[5]]:
                ad_info["advertiser_info"][line[5]][line[4]] += 1
            else:
                ad_info["advertiser_info"][line[5]][line[4]] = 1
        else:
            ad_info["advertiser_info"][line[5]] = dict()
            ad_info["advertiser_info"][line[5]]["_count"] = 1
            ad_info["advertiser_info"][line[5]][line[4]] = 1

    print("data with {} lines".format(_count))
    with open(outfile, "w", encoding="utf-8") as f:
        f.write(json.dumps(ad_info, ensure_ascii=False, indent=4))
    print("write down")
    return ad_info

def read_data_from_json_file(file):
    print("load data from file: {}".format(file))
    with open(file, "r", encoding="utf-8") as jsonfile:
        info = json.load(jsonfile)
    print("read down")
    return info

def analysis_ad(file, outfile):

    if not os.path.exists(outfile):
        ad_info = write_ad_analysis_details_to_file(file, outfile)
    else:
        ad_info = read_data_from_json_file(outfile)

    # 产品信息统计
    product_count = len(ad_info["product_info"].keys())
    print("\nall products category: {}".format(product_count))
    new_product_info = dict()
    for k, v in ad_info["product_info"].items():
        new_product_info[k] = v["_count"]
    sort_category = dict_sort(new_product_info)
    print("--- details of product category info ---")
    print("{}".format(json.dumps(sort_category, indent=4)))

    # 广告主信息统计
    advertiser_count = len(ad_info["advertiser_info"].keys())
    print("\nall advertisers industry: {}".format(advertiser_count))
    new_advertiser_info = dict()
    for k, v in ad_info["advertiser_info"].items():
        new_advertiser_info[k] = v["_count"]
    sort_industry = dict_sort(new_advertiser_info)
    print("--- details of advertiser industry info ---")
    print("{}".format(json.dumps(sort_industry, indent=4)))

def analysis_user(file, outfile):
    if not os.path.exists(outfile):
        user_info = write_user_analysis_details_to_file(file, outfile)
    else:
        user_info = read_data_from_json_file(outfile)

    # 用户信息统计
    print("\nall age: {}".format(user_info["age_info"].keys()))
    sort_age = dict_sort(user_info["age_info"])
    print("--- details of age info ---")
    print("{}".format(json.dumps(sort_age, indent=4)))
    print("\nall gender: {}".format(user_info["gender_info"].keys()))
    sort_gender = dict_sort(user_info["gender_info"])
    print("--- details of gender info ---")
    print("{}".format(json.dumps(sort_gender, indent=4)))
    print("\nall age with gender: {}".format(user_info["age_gender_info"].keys()))
    sort_age_gender = dict_sort(user_info["age_gender_info"])
    print("--- details of age with gender info ---")
    print("{}".format(json.dumps(sort_age_gender, indent=4)))



def write_user_analysis_details_to_file(file, outfile):
    print("start writing user analysis details to file: {}".format(outfile))
    user_info = dict()
    user_info["age_info"] = dict()
    user_info["gender_info"] = dict()
    user_info["age_gender_info"] = dict()
    _count = 0

    for line in read_data_from_csv_file(file):
        _count += 1
        if _count < 6:
            print(line)
        # else:
        #     break
        # 用户信息
        age_gender = "{}_{}".format(line[1], line[2])
        if age_gender in user_info["age_gender_info"]:
            user_info["age_gender_info"][age_gender] += 1
            if line[1] in user_info["age_info"]:
                user_info["age_info"][line[1]] += 1
            else:
                user_info["age_info"][line[1]] = 1
            if line[2] in user_info["gender_info"]:
                user_info["gender_info"][line[2]] += 1
            else:
                user_info["gender_info"][line[2]] = 1

        else:
            user_info["age_gender_info"][age_gender] = dict()
            user_info["age_gender_info"][age_gender] = 1

    print("data with {} lines".format(_count))
    with open(outfile, "w", encoding="utf-8") as f:
        f.write(json.dumps(user_info, ensure_ascii=False, indent=4))
    print("write down")
    return user_info


def plot_user_click_dist(all_text_len, desc="data", save_path=None):
    """
    绘制用户点击次数分布
    :param all_text_len: [22,33,34,...]
    :param save_path: 图片保存路径
    :return:
    """
    plt.figure()
    plt.title("Length dist of {}".format(desc))
    plt.xlabel("Length")
    plt.ylabel("Count")
    _, _, all_data = plt.hist(all_text_len, bins=100, density=True, alpha=.2, color="b")
    plt.show()
    if save_path:
        plt.savefig(save_path)


def analysis_click(file, outfile):
    if not os.path.exists(outfile):
        click_info = write_click_analysis_details_to_file(file, outfile)
    else:
        click_info = read_data_from_json_file(outfile)
    # 用户点击信息统计绘图
    user_id = list()
    user_total_click_times = list()
    user_total_click_days = list()
    for k in click_info.keys():
        user_id.append(k)
        user_total_click_times.append(int(click_info[k]["total_click_times_info"]))
        user_total_click_days.append(int(click_info[k]["total_click_days_info"]["_count"]))
    user_click_df = pd.DataFrame(zip(user_id, user_total_click_times, user_total_click_days), columns=["user_id", "total_click_times", "total_click_days"])
    columns_name = ["total_click_times", "total_click_days"]
    total_count_user = len(click_info.keys())
    print("\ntotal user: {}".format(total_count_user))

    print("--- details of click info ---")
    print(user_click_df[columns_name].describe())
    print("--- details of total click times(<=300) info ---")
    print(user_click_df[columns_name][user_click_df.total_click_times<=300].describe())
    print("--- details of total click times(>300) info ---")
    print(user_click_df[columns_name][user_click_df.total_click_times>300].describe())
    print("--- plot of click times---")
    plot_user_click_dist(user_total_click_times, desc="total click times")
    plot_user_click_dist([ct for ct in user_total_click_times if ct <= 300], desc="total click times(<=300)")
    plot_user_click_dist([ct for ct in user_total_click_times if ct > 300], desc="total click times(>300)")
    print("--- details of total click days(>45) info ---")
    print(user_click_df[columns_name][user_click_df.total_click_days>45].describe())
    # print("total user click more than 45 days: {}".format(user_click_days_df[user_click_days_df.total_click_days>45].__len__()))
    print("--- plot of click days---")
    plot_user_click_dist(user_total_click_days, desc="total click days")
    print("--- details of total click times(>100) and days(>60) info ---")
    print(user_click_df[columns_name][(user_click_df.total_click_times>100) & (user_click_df.total_click_days>60)].describe())



def write_click_analysis_details_to_file(file, outfile):
    print("start writing click analysis details to file: {}".format(outfile))
    click_info = dict()
    _count = 0

    for line in read_data_from_csv_file(file):
        _count += 1
        if _count < 6:
            print(line)
        # else:
        #     break
        # 用户点击信息
        if line[1] in click_info:
            click_info[line[1]]["total_click_times_info"] += int(line[3])
            if line[0] not in click_info[line[1]]["total_click_days_info"]:
                click_info[line[1]]["total_click_days_info"][line[0]] = 1
                click_info[line[1]]["total_click_days_info"]["_count"] += 1
            else:
                click_info[line[1]]["total_click_days_info"][line[0]] += 1
        else:
            click_info[line[1]] = dict()
            click_info[line[1]]["total_click_times_info"] = int(line[3])
            click_info[line[1]]["total_click_days_info"] = dict()
            click_info[line[1]]["total_click_days_info"][line[0]] = 1
            click_info[line[1]]["total_click_days_info"]["_count"] = 1


    print("data with {} lines".format(_count))
    with open(outfile, "w", encoding="utf-8") as f:
        f.write(json.dumps(click_info, ensure_ascii=False, indent=4))
    print("write down")
    return click_info


def load_user_info(file):
    print("load user info from file: {}".format(file))
    user_info = dict()
    for line in read_data_from_csv_file(file):
        if line[0] not in user_info:
            user_info[line[0]] = dict()
            user_info[line[0]]["age"] = int(line[1])
            user_info[line[0]]["gender"] = int(line[2])
        else:
            print("exist user id {}".format(line[0]))
    return user_info

def load_ad_info(file):
    print("load ad info from file: {}".format(file))
    ad_info = dict()
    for line in read_data_from_csv_file(file):
        if line[0] not in ad_info:
            ad_info[line[0]] = []
            ad_info[line[0]].append(int(line[1]))
            ad_info[line[0]].append(int(line[2]) if line[2] != "\\N" else None)
            ad_info[line[0]].append(int(line[3]))
            ad_info[line[0]].append(int(line[4]))
            ad_info[line[0]].append(int(line[5]) if line[5] != "\\N" else None)
        else:
            print("exist creative id {}".format(line[0]))
    return ad_info


def load_click_info(file):
    print("load click info from file: {}".format(file))
    _count = 0
    seq_data = dict()
    for line in read_data_from_csv_file(file):
        _count += 1
        if _count < 6:
            print(line)
        # else:
        #     break
        # 用户点击信息
        if line[1] in seq_data:
            seq_data[line[1]]["click_seq"][int(line[0]) - 1].append((line[2], int(line[3])))
        else:
            seq_data[line[1]] = dict()
            seq_list = [[] for _ in range(91)]
            seq_data[line[1]]["click_seq"] = seq_list
            seq_data[line[1]]["click_seq"][int(line[0])-1].append((line[2], int(line[3])))

    return seq_data


def gen_data(click_file, ad_file, user_file, train_file):
    ad_info = load_ad_info(ad_file)
    user_info = load_user_info(user_file)
    click_info = load_click_info(click_file)
    with open(train_file, "w", encoding="utf-8") as f:
        for user_id, v in click_info.items():
            line = dict()
            line["user_id"] = user_id
            click_seq = v["click_seq"]
            line["ad_id_seq"] = list()
            line["product_id_seq"] = list()
            line["product_category_seq"] = list()
            line["advertiser_id_seq"] = list()
            line["industry_seq"] = list()
            line["click_times_seq"] = list()
            if user_id in user_info:
                line["age"] = user_info[user_id]["age"]
                line["gender"] = user_info[user_id]["gender"]
            else:
                line["age"] = None
                line["gender"] = None
            for click in click_seq:
                ad_click = [[], [], [], [], []]
                click_times = []
                if click:
                    creative_id = click[0]
                    click_times = click[1]
                    if creative_id in ad_info:
                        ad_click = ad_info[creative_id]

                line["ad_id_seq"].append(ad_click[0])
                line["product_id_seq"].append(ad_click[1])
                line["product_category_seq"].append(ad_click[2])
                line["advertiser_id_seq"].append(ad_click[3])
                line["industry_seq"].append(ad_click[4])
                line["click_times_seq"].apppend(click_times)

            f.write(json.dumps(line) + "\n")



def main():
    ad_data_file = "/data/work/dl_project/data/corpus/tencent_ad_2020/train_preliminary/ad.csv"
    ad_data_analysis_file = "/data/work/dl_project/data/corpus/tencent_ad_2020/train_preliminary/ad_analysis.json"

    user_data_file = "/data/work/dl_project/data/corpus/tencent_ad_2020/train_preliminary/user.csv"
    user_data_analysis_file = "/data/work/dl_project/data/corpus/tencent_ad_2020/train_preliminary/user_analysis.json"
    click_log_data_file = "/data/work/dl_project/data/corpus/tencent_ad_2020/train_preliminary/click_log.csv"
    click_log_data_analysis_file = "/data/work/dl_project/data/corpus/tencent_ad_2020/train_preliminary/click_log_analysis.json"

    train_file = "/data/work/dl_project/data/corpus/tencent_ad_2020/train_preliminary/train.txt"

    # analysis_ad(ad_data_file, ad_data_analysis_file)
    # analysis_user(user_data_file, user_data_analysis_file)
    # analysis_click(click_log_data_file, click_log_data_analysis_file)

    # user_data = read_data_from_csv(user_data_file)
    #
    # click_log_data = read_data_from_csv(click_log_data_file)
    gen_data(click_log_data_file, ad_data_file, user_data_file, train_file)


if __name__ == "__main__":
    main()