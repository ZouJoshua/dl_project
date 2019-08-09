import json
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, recall_score, classification_report, confusion_matrix
import logging
from preprocess.preprocess_tools import read_json_format_file


class EvaluateModel(object):

    def __init__(self, predict_file, key_name='category', logger=None, label_names=None):
        self.pred_file = predict_file
        self.kn = key_name
        if logger:
            self.log = logger
        else:
            self.log = logging.getLogger("evaluate_model_log")
            self.log.setLevel(logging.INFO)
        self.label_names = label_names

    def evaluate_model(self):
        y_true, y_pred = self.get_pred_result_from_file()
        self.report_result(y_true, y_pred)

    def evaluate_model_v2(self):
        y_true, y_pred = self.get_pred_result_from_file()
        result = self.report_result_v2(y_true, y_pred)
        return


    def get_pred_result_from_file(self):
        y_true = list()
        y_pred = list()
        self.log.info("正在从预测文件获取结果...")
        for line in read_json_format_file(self.pred_file):
            true_category = str(line[self.kn]).lower().strip()
            y_true.append(true_category)
            pred_category = str(line['predict_{}'.format(self.kn)]).lower().strip()
            y_pred.append(pred_category)
        return y_true, y_pred


    def report_result(self, y_true, y_pred):
        """
        自定义报告模型评估结果
        :param y_true: 真实list
        :param y_pred: 预测list
        :return:
        """
        true_labels = list(set(y_true))
        pred_labels = list(set(y_pred))

        A = dict.fromkeys(true_labels, 0)  # 预测正确的各个类的数目
        B = dict.fromkeys(true_labels, 0)  # 测试数据集中实际各个类的数目
        C = dict.fromkeys(pred_labels, 0)  # 测试数据集中预测的各个类的数目
        for i in range(0, len(y_true)):
            B[y_true[i]] += 1
            C[y_pred[i]] += 1
            if y_true[i] == y_pred[i]:
                A[y_true[i]] += 1

        # 计算准确率，召回率，F值
        self.log.info("计算预测结果的准确率、召回率和F值")
        result = dict()
        for key in B:
            try:
                r = float(A[key]) / float(B[key])
                p = float(A[key]) / float(C[key])
                f = p * r * 2 / (p + r)
                _res = "p:%f r:%f f:%f" % (p, r, f)
                result[key] = _res
            except:
                # print("error:", key, "right:", A.get(key, 0), "real:", B.get(key, 0), "predict:", C.get(key, 0))
                _res = "right:{} real:{} predict:{}".format(A.get(key, 0), B.get(key, 0), C.get(key, 0))
                result["error_{}".format(key)] = _res

        self.log.info("\n----结果报告 ---:\n{}".format(json.dumps(result, indent=4)))


    def plot_confusion_matrix(self, cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            self.log.info("Normalized confusion matrix")
        else:
            self.log.info('Confusion matrix, without normalization')

        # print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    def report_result_v2(self, y_true, y_pred, simple=False):

        # 计算检验
        F1_score = f1_score(y_true, y_pred, average='macro')
        if simple:
            return F1_score
        else:
            acc = accuracy_score(y_true, y_pred)
            recall_score_ = recall_score(y_true, y_pred, average='macro')
            confusion_matrix_ = confusion_matrix(y_true, y_pred, labels=self.label_names)
            self.log.debug('\n----混淆矩阵 ---:\n{}'.format(confusion_matrix_))
            class_report = classification_report(y_true, y_pred, target_names=self.label_names)
            self.log.info('\n----模型整体 ----\nf1_score:\t{} \nacc_score:\t{} \nrecall:\t{}'.format(F1_score, acc, recall_score_))
            self.log.info('\n----结果报告 ---:\n{}'.format(class_report))

            # # 画混淆矩阵,画混淆矩阵图
            # plt.figure()
            # self.plot_confusion_matrix(confusion_matrix_, classes=self.label_names,
            #                       title='Confusion matrix, without normalization')
            # plt.show()

            return F1_score, acc, recall_score_, confusion_matrix_, class_report
