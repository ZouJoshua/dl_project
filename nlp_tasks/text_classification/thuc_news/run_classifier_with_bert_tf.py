#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 3/13/20 5:55 PM
@File    : run_classifier_with_bert_tf.py
@Desc    : 

"""

import json
import argparse
import configparser
import time
import logging
import tensorflow as tf
from model_tensorflow.bert_model import modeling
from nlp_tasks.text_classification.thuc_news.bert_tf_model import BertClassifier
from nlp_tasks.text_classification.thuc_news.dataset_loader_for_bert_tf import DatasetLoader
from evaluate.metrics import mean, get_multi_metrics
from setting import DATA_PATH, CONFIG_PATH
from utils.logger import Logger

import time

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "100"


class Config(object):
    def __init__(self, config_file, section="THUC_NEWS"):
        config_ = configparser.ConfigParser()
        config_.read(config_file)
        all_items = config_.items(section)
        self.all_params = {}
        for i in all_items:
            self.all_params[i[0]] = i[1]
        # self.log.info("*** Init all params ***")
        # self.log.info(json.dumps(params, indent=4))

        config = config_[section]
        if not config:
            raise Exception("Config file error.")

        # path
        self.data_dir = config.get("data_dir")
        self.output_dir = config.get("output_dir")
        self.bert_init_checkpoint = config.get("init_checkpoint")
        self.bert_config_path = config.get("bert_config_file")
        self.vocab_file = config.get("vocab_file")
        self.label2idx_path = config.get("label2idx_path")
        # model params
        self.sequence_length = config.getint("sequence_length")
        self.num_classes = config.getint("num_classes")
        self.learning_rate = config.getfloat("learning_rate")
        self.num_train_epochs = config.getint("num_train_epochs")
        self.train_batch_size = config.getint("train_batch_size")
        self.warmup_proportion = config.getfloat("warmup_proportion")
        self.checkpoint_every = config.getint("checkpoint_every")
        self.eval_batch_size = config.getint("eval_batch_size")
        self.model_name = config.get("model_name")



class Trainer(object):
    def __init__(self, config, logger=None):

        if logger:
            self.log = logger
        else:
            self.log = logging.getLogger("bert_train_log")
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
            logging.root.setLevel(level=logging.INFO)

        self.config = config

        self.data_obj = None
        self.model = None
        self.bert_init_checkpoint = config.bert_init_checkpoint

        # 加载数据集
        self.data_obj = DatasetLoader(config, logger=self.log)
        self.train_features = self.load_data(self.data_obj, mode="train")
        self.log.info("train data size: {}".format(len(self.train_features)))
        self.eval_features = self.load_data(self.data_obj, mode="eval")
        self.log.info("eval data size: {}".format(len(self.eval_features)))

        # 加载label
        self.label_map = self.data_obj.label_map
        self.label_list = [value for key, value in self.label_map.items()]
        self.log.info("label numbers: {}".format(len(self.label_list)))

        self.train_epochs = config.num_train_epochs
        self.train_batch_size = config.train_batch_size
        self.eval_batch_size = config.eval_batch_size
        warmup_proportion = config.warmup_proportion
        num_train_steps = int(
            len(self.train_features) / self.train_batch_size * self.train_epochs)
        num_warmup_steps = int(num_train_steps * warmup_proportion)

        # 初始化模型对象
        self.model = self.create_model(num_train_steps, num_warmup_steps)

    def init_config(self, config_file):
        config_ = configparser.ConfigParser()
        config_.read(config_file)
        all_items = config_.items("THUC_NEWS")
        params = {}
        for i in all_items:
            params[i[0]] = i[1]
        self.log.info("*** Init all params ***")
        self.log.info(json.dumps(params, indent=4))

        config = config_["THUC_NEWS"]
        if not config:
            raise Exception("Config file error.")
        return config

    def init_config_v1(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--config_path", help="config path of model")
        args = parser.parse_args()
        with open(args.config_path, "r") as fr:
            config = json.load(fr)
        return config

    def load_data(self, data_obj, mode=None):
        """
        创建数据对象
        :return:
        """
        # 生成训练集对象并生成训练数据

        data_file = os.path.join(self.config.data_dir, "thuc_news.{}.txt".format(mode))
        pkl_file = os.path.join(self.config.data_dir, "thuc_news.{}.pkl".format(mode))
        if not os.path.exists(data_file):
            raise FileNotFoundError

        features = data_obj.gen_data(data_file, pkl_file, mode=mode)

        return features

    def create_model(self, num_train_step, num_warmup_step):
        """
        根据config文件选择对应的模型，并初始化
        :return:
        """
        model = BertClassifier(config=self.config, num_train_step=num_train_step, num_warmup_step=num_warmup_step, logger=self.log)
        return model

    def train(self):
        with tf.Session() as sess:
            tvars = tf.trainable_variables()
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
                tvars, self.bert_init_checkpoint)
            self.log.info("*** Init bert model params ***")
            tf.train.init_from_checkpoint(self.bert_init_checkpoint, assignment_map)
            # self.log.info("*** Init bert model params done ***")
            self.log.info("*** Trainable Variables ***")
            for var in tvars:
                init_string = ""
                if var.name in initialized_variable_names:
                    init_string = ", *INIT_FROM_CKPT*"
                self.log.info("  name = %s, shape = %s%s", var.name, var.shape,
                                init_string)

            sess.run(tf.variables_initializer(tf.global_variables()))

            output_dir = self.config.output_dir
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            self.log.info("*** Start Training ***")
            current_step = 0
            start = time.time()
            for epoch in range(self.train_epochs):
                self.log.info("----- Epoch {}/{} -----".format(epoch + 1, self.train_epochs))

                for batch in self.data_obj.next_batch(self.train_features, self.train_batch_size, mode="train"):
                    loss, predictions = self.model.train(sess, batch)

                    acc, recall, F1 = get_multi_metrics(pred_y=predictions, true_y=batch["label_ids"])
                    self.log.info("train-step: {}, loss: {}, acc: {}, recall: {}, F1_score: {}".format(
                        current_step, loss, acc, recall, F1))

                    current_step += 1
                    if self.data_obj and current_step % self.config.checkpoint_every == 0:

                        eval_losses = []
                        eval_accs = []
                        eval_recalls = []
                        eval_f_betas = []
                        eval_aucs = []
                        eval_precs = []
                        for eval_batch in self.data_obj.next_batch(self.eval_features, self.eval_batch_size, mode="eval"):
                            eval_loss, eval_predictions = self.model.eval(sess, eval_batch)

                            eval_losses.append(eval_loss)

                            acc, recall, F1 = get_multi_metrics(pred_y=eval_predictions,
                                                                          true_y=eval_batch["label_ids"])
                            eval_accs.append(acc)
                            eval_recalls.append(recall)
                            eval_f_betas.append(F1)
                        self.log.info("\n")
                        self.log.info("eval:  loss: {}, acc: {}, recall: {}, F1_score: {}".format(
                            mean(eval_losses), mean(eval_accs), mean(eval_recalls), mean(eval_f_betas)))
                        self.log.info("\n")


                        model_save_path = os.path.join(output_dir, self.config.model_name)
                        self.model.saver.save(sess, model_save_path, global_step=current_step)

            end = time.time()
            self.log.info("total train time: ", end - start)




class Predictor(object):
    def __init__(self, config, logger=None):

        if logger:
            self.log = logger
        else:
            self.log = logging.getLogger("bert_train_log")
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
            logging.root.setLevel(level=logging.INFO)

        self.config = config

        self.data_obj = DatasetLoader(self.config, logger=self.log)
        self.label_map = self.data_obj.label_map
        self.index_to_label = {value: key for key, value in self.label_map.items()}
        self.ckpt_model_path = self.config.output_dir
        # 创建模型
        self.create_model()
        # 加载计算图
        self.load_graph()

    def load_graph(self):
        """
        加载计算图
        :return:
        """
        self.sess = tf.Session()
        ckpt = tf.train.get_checkpoint_state(self.ckpt_model_path)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            self.log.info('*** Reloading model parameters ***')
            self.model.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            raise ValueError('No such file:[{}]'.format(self.ckpt_model_path))

    def create_model(self):
        """
        根据config文件选择对应的模型，并初始化
        :return:
        """
        self.model = BertClassifier(config=self.config, is_training=False, logger=self.log)

    def predict(self, text):
        """
        给定分词后的句子，预测其分类结果
        :param text:
        :return:
        """
        input_ids = []
        input_masks = []
        segment_ids = []
        guid = "predict-1"
        s = time.time()
        feature = self.data_obj.convert_single_example_to_feature(guid, text)
        input_ids.append(feature.input_ids)
        input_masks.append(feature.input_mask)
        segment_ids.append(feature.segment_ids)

        s1 = time.time()
        print("特征转化耗时: {}s".format(s1 - s))

        prediction = self.model.infer(self.sess,
                                      dict(input_ids=input_ids,
                                           input_masks=input_masks,
                                           segment_ids=segment_ids)).tolist()[0]

        label = self.index_to_label[prediction]
        e = time.time()
        print("模型预测耗时:{}s".format(e - s1))
        return label

    def predict_batch(self, sentences):
        input_ids = []
        input_masks = []
        segment_ids = []
        guid = "predict-5"
        # s = time.time()
        for sentence in sentences:
            feature = self.data_obj.convert_single_example_to_feature(guid, sentence)
            input_ids.append(feature.input_ids)
            input_masks.append(feature.input_mask)
            segment_ids.append(feature.segment_ids)
        # s1 = time.time()
        # print("特征转化耗时: {}s".format(s1 - s))

        predictions = self.model.infer(self.sess,
                                      dict(input_ids=input_ids,
                                           input_masks=input_masks,
                                           segment_ids=segment_ids)).tolist()
        out_label = list()
        for label_id in predictions:
            label = self.index_to_label[label_id]
            out_label.append(label)
        # e = time.time()
        # print("模型预测耗时:{}s".format(e - s1))

        return out_label







def train_model():
    """
    训练模型
    :return:
    """
    model_path = os.path.join(DATA_PATH, "model", "thuc_news")
    log_file = os.path.join(model_path, 'bert_train_log')
    log = Logger("bert_train_log", log2console=False, log2file=True, logfile=log_file).get_logger()
    conf_file = os.path.join(CONFIG_PATH, "bert_model_config.ini")
    config = Config(conf_file)
    log.info("*** Init all params ***")
    log.info(json.dumps(config.all_params, indent=4))
    trainer = Trainer(config, logger=log)
    trainer.train()


def predict_demo():
    """
    预测demo
    :return:
    """
    model_path = os.path.join(DATA_PATH, "model", "thuc_news")
    log_file = os.path.join(model_path, 'bert_train_log')
    log = Logger("bert_train_log", log2console=False, log2file=True, logfile=log_file).get_logger()
    conf_file = os.path.join(CONFIG_PATH, "bert_model_config.ini")
    config = Config(conf_file)
    log.info("*** Init all params ***")
    log.info(json.dumps(config.all_params, indent=4))
    s = time.time()
    predictor = Predictor(config, logger=log)
    s1 = time.time()
    print("加载模型耗时: {}s".format(s1 - s))
    # text = "09互联网大会嘉宾：世纪互联副总裁肖峰\n　　简介：\n　　毕业于首都经济贸易大学，拥有12年互联网网络研究与实践、项目管理实践、团队组织与实践等经验。\n　　1996 年加盟世纪互联数据中心有限公司，先后领导公司产品管理、市场销售、客户服务等团队，成功为新浪、搜狐、KDDI、外交部、CCTV等大型门户网站、国际知名企业及政府机构等众多重要客户提供了专业的互联网部署、IT 外包解决方案及实施服务，为世纪互联的快速、稳定发展奠定了坚实的基础，同时也为中国IDC产业的健康发展起到了积极推进作用。\n\n"
    # label = predictor.predict(text)
    text_list = [
        "经典透明翻盖 摩托罗拉全触屏ZN4曝光\n　　北京时间2008年10月13日，我们在网上发现了由国外媒体曝光的摩托罗拉ZN4的真机图片。这款手机的全称为摩托罗>       拉Krave ZN4。据悉摩托罗拉Krave ZN4是一款专为美国本土的运营商Verizon Wireless定制的手机，而它也是继摩托罗拉ZN5之后ZINE系列的又一款新机。\n　　从外观上看摩托罗拉Krave ZN4>       舍弃了摩托罗拉ZN5的直板机身设计，而是采用了\"明\"系列的造型——经典的透明翻盖。据悉摩托罗拉Krave ZN4将配置一个全触摸式屏幕，支持虚拟的QWERTY全功能键盘以及虚拟传统键盘，据>       拿到这款手机的测试者透露这款手机的通话性能优秀，触摸屏的反应速度也很灵敏。\n\n\n\n",
        "《美色江山》端午的终极粽子大礼\n　　端午的粽子香甜糯软，最后一轮的材料收集开始了，粽叶、糯米、红枣统统都拿来！\n　　每天回答问题最多能得到2份>       材料！6月9日可以得到粽叶、6月10日得到糯米、6月11日得到红枣。\n　　集齐三种材料后，可以制作一个粽子，使用后得到随机道具。商城中做好的粽子可以随时享用哦！\n　　吃出易筋经>       、洗髓丹、封侯令什么的都有可能，打开来看看你的粽子包着什么吧！\n\n",
        "美国纽约大客车与卡车相撞30余人死伤\n　　新华网纽约7月22日电 一辆旅游大客车与一辆大型卡车22日凌晨在纽约州滑铁卢市附近相撞，造成1人死亡、30多人>       受伤，其中2人伤势严重。\n　　事故发生在纽约市东北400多公里的90号州际公路上，两车相撞后起火，卡车司机当场死亡，伤者已被送往附近医院救治。目前尚不清楚车上是否有中国人。\n>       　　事故原因尚未确定。当地一家电视台报道说，大客车因机械故障在路肩上临时停车，重新进入车道时被卡车撞上。\n　　目前还没有关于旅游大客车所属公司及具体运行线路的信息。\n欢>       迎发表评论我要评论\n\n",
        "中国政法大学2011年分省市来源计划 \n　　新浪教育讯 中国政法大学本科招生网近日公布了2011年分省计划，特别搜集整理以方便考生阅读，详细内容如下：\n       　　>>中国政法大学2011年分省市来源计划\n\n",
        "东吴新经济基金将于12日起正式发行\n　　新华网北京11月7日电(记者陶俊洁、赵晓辉)记者7日从东吴基金管理有限公司获悉，东吴旗下第6只基金??东吴新经济>       基金将于11月12日起正式发行。\n　　东吴基金有关人士介绍，东吴新经济基金作为一只明确以新经济产业为投资目标的主题型基金，将重点投资于生物技术、信息技术、新材料技术、先进制>       造技术、先进能源技术、海洋技术、激光技术等新经济产业。\n　　与此同时，东吴新经济基金将利用创业板开闸的机遇，投资创业板市场上的科技创新型企业。\n　　据了解，投资者可到建>       行、工行、农行、交行、东吴证券等各大银行券商及东吴基金认购该基金。\n   已有_COUNT_条评论  我要评论\n\n",
        "女子挪用超市资金300万给情夫使用\n　　晨报讯 胡芳利用担任超市财务的工作便利，在情夫邝伟民的指使下，竟疯狂挪用超市营业资金300余万元给其使用。两>       人一审分别被判处有期徒刑6年后，邝伟民提起上诉。上海市检察院第二分院审查认为，他的上诉理由不能成立，建议二审法院维持原判。记者昨晚获悉，二审法院判决驳回上诉，维持原判。 >       　　□记者 赵 磊 通讯员 魏 珉\n\n",
        "破军星一周事业运特别提醒(图)\n　　破军星：★★★☆☆\n　　职业运：\n　　在自我表现方面要特别注意选择好适当的时机，以免表现得不合时宜，反而降低你在>       上司、同事心目中得印象分。\n　　学业运：\n　　自制能力较弱，难以以积极的心态去主动完成某件事，需要依赖于他人的提醒、督促。\n　　压力指数：37%\n　　贵人主星：巨门星\n　小人主星：紫微星\n　　开运色：亮宝蓝色\n　　相关推荐：命格详批升级版 姓名恋爱配对 紫微事业命盘 前程何时会旺 \n\n",
        "组图：衬衫+薄针织衫 绚丽效果够惊喜\n　　导语：春季一来到，大家都在忙着清理衣橱吧？把厚重又严实的羽绒服通通收起来，这一季，抛弃了沉重的衣服，最绚丽夺目的流行服饰立刻上位。衬衫是必备单品之一，用它搭配轻薄的针织衫会有什么效果？一起去看看！\n　　搭配建议：老气的花色也有年轻的穿法，大花针织衫搭配花色老气的衬衫，符 > 合负负得正的原理，看起来少了份沉闷多些青春活力。\n　　搭配建议：天气暖了，衣服穿的薄了，腰部的赘肉就要显出来了，有小肚腩的不用害怕，紧急解救方案来了，宽松开衫＋格子衬衫 >       ，外搭一条宽腰带，不但小肚肚不见了，还显出了性感小蛮腰。\n　　搭配建议：很喜欢北欧风毛衣的美女，不要因为春天来了就急着把衣服收起来，宽松的毛衣搭配一件修身白衬衫，本季最 > 酷的穿法。\n　　搭配建议：纯棉格子衫，大一号当成外套穿，帅气时尚，本季作为内搭款，个性优雅，风格可随你的外搭而改变，优雅的女人也是因为有了优雅的外搭成就的。\n　　搭配建 > 议：同样一款格子衫，搭配糖果色的毛衣，和短裙，就打造了一个俏皮活泼的你。如果换成秀气的高跟鞋，那么又会打造出一个小女人味十足的你。\n\n",
        "胜负彩11071期任九场开奖：一等399注 奖金1万6\n　　北京时间8月1日，胜负彩11071期任九场开奖结果揭晓：一等奖399注，每注奖金16414元，任选九场销量10       233546元。\n　　温馨提示：兑奖有效期60天，本期兑奖截止日为2011年09月29日，逾期作弃奖处理。\n\n",
        "桂纶镁《线人》颠覆自我 气质空灵家居照盘点(6)\n　　桂纶镁在电影《线人》中饰演一位性格暴烈的女悍匪，与之前的清纯形象大相径庭。谈到与片中两位男主角谢霆锋、陆毅的情感纠葛，桂纶镁表示，“我和谢霆锋在交叉路口遇见，两个人因为相知相惜的东西而靠在一起。陆毅就好像是青梅竹马，他像是大哥哥一样和我一起长大。”当被问及更愿意 > 和谁在一起的时候，桂纶镁面露难色：“昨天苗圃说谁在身边我就说谁的好话，这一次两个人都在了，那我就选张家辉吧。”\n　　桂纶镁曾被人形容为“外表很年轻，内心像老奶奶”。对此她无 > 奈地说，“其实我的性格很复杂，有时候很孩子气、很闹、很任性，但谈公事就好象老太太一样按部就班，比较条理化，比较守旧。”\n\n"
        ]
    label = predictor.predict_batch(text_list)
    print("预测类别为:{}".format(label))


def write_file_with_predict():
    """
    写预测结果到文件中
    :return:
    """
    model_path = os.path.join(DATA_PATH, "model", "thuc_news")
    log_file = os.path.join(model_path, 'bert_train_log')
    log = Logger("bert_train_log", log2console=False, log2file=True, logfile=log_file).get_logger()
    conf_file = os.path.join(CONFIG_PATH, "bert_model_config.ini")
    config = Config(conf_file)
    log.info("*** Init all params ***")
    log.info(json.dumps(config.all_params, indent=4))
    predictor = Predictor(config, logger=log)
    files = [os.path.join(DATA_PATH, "corpus", "thuc_news", "thuc_news.{}.txt".format(i)) for i in ["train", "eval", "test"]]
    predict_file = os.path.join(DATA_PATH, "corpus", "thuc_news", "thuc_news.predict.txt")
    batch_size = 20
    with open(predict_file, "w", encoding="utf-8") as wf:
        for file in files:
            file_type = os.path.split(file)[1].split(".")[1]
            with open(file, "r", encoding="utf-8") as f:
                lines = f.readlines()

                num_batches = len(lines) // batch_size
                for i in range(num_batches+1):
                    start = i * batch_size
                    end = start + batch_size
                    text_batch = list()
                    true_labels = list()
                    ids = list()
                    if end > len(lines):
                        _lines = lines[start:]
                    else:
                        _lines = lines[start:end]

                    for i, _line in enumerate(_lines):
                        line = json.loads(_line.strip())
                        ids.append(start+i)
                        true_labels.append(line["label"])
                        text_batch.append(line["text"])

                    predict_labels = predictor.predict_batch(text_batch)
                    for j, _ in enumerate(predict_labels):
                        out = dict()
                        out["guid"] = "{}-{}".format(file_type, ids[j])
                        out["true_label"] = true_labels[j]
                        out["predict_label"] = predict_labels[j]
                        if out:
                            wf.write(json.dumps(out, ensure_ascii=False) + "\n")



def predict_report():
    """
    预测结果评估
    :return:
    """
    from sklearn.metrics import classification_report
    from evaluate.metrics import get_multi_metrics
    import json
    file = "/data/work/dl_project/data/corpus/thuc_news/thuc_news.predict.txt"
    result = dict()
    result["train"] = (list(), list())
    result["eval"] = (list(), list())
    result["test"] = (list(), list())
    with open(file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for _line in lines:
            line = json.loads(_line.strip())
            guid = line["guid"]
            mode = guid.split("-")[0]
            true_y = line["true_label"]
            pred_y = line["predict_label"]
            if mode == "train":
                result["train"][0].append(true_y)
                result["train"][1].append(pred_y)
            elif mode == "eval":
                result["eval"][0].append(true_y)
                result["eval"][1].append(pred_y)
            elif mode == "test":
                result["test"][0].append(true_y)
                result["test"][1].append(pred_y)

    labels_list = ['财经', '彩票', '房产', '股票', '家居', '教育', '科技',
                   '社会', '时尚', '时政', '体育', '星座', '游戏', '娱乐']
    for k, v in result.items():
        print("{}的整体性能:".format(k))
        acc, recall, F1 = get_multi_metrics(v[0], v[1])
        print('\n----模型整体 ----\nacc_score:\t{} \nrecall:\t{} \nf1_score:\t{} '.format(acc, recall, F1))
        print("{}的详细结果:".format(k))
        class_report = classification_report(v[0], v[1], labels=labels_list)
        print('\n----结果报告 ---:\n{}'.format(class_report))




def main():
    # 训练模型
    # train_model()
    # 预测demo
    # predict_demo()
    # 写预测结果到文件中
    # write_file_with_predict()
    # 预测结果评估
    predict_report()


if __name__ == "__main__":
    # 读取用户在命令行输入的信息
    main()