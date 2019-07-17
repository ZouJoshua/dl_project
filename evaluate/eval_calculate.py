import json


def evaluate_model(dataPath, model_level='two_level', model_num='national_model_1'):
    """

    :param dataPath:
    :param model_level:
    :param model_num:
    :return:
    """
    labels_right = []
    labels_predict = []
    with open(dataPath, "r") as fr:
        lines = fr.readlines()
        for line in lines:
            line = json.loads(line)
            true_category = str(line[model_level]).lower().strip()
            if true_category in ['shopping', 'aoto', 'world', 'sport']:
                continue
            labels_right.append(true_category)
            labels_predict.append(line['predict_{}'.format(model_level)])
    text_labels = list(set(labels_right))
    text_predict_labels = list(set(labels_predict))

    A = dict.fromkeys(text_labels, 0)  # 预测正确的各个类的数目
    B = dict.fromkeys(text_labels, 0)   # 测试数据集中实际各个类的数目
    C = dict.fromkeys(text_predict_labels, 0)  # 测试数据集中预测的各个类的数目
    for i in range(0, len(labels_right)):
        B[labels_right[i]] += 1
        C[labels_predict[i]] += 1
        if labels_right[i] == labels_predict[i]:
            A[labels_right[i]] += 1
    # print(A)
    # print(B)
    # print(C)
    # 计算准确率，召回率，F值
    print('计算模型{}效果'.format(model_num))
    for key in B:
        try:
            r = float(A[key]) / float(B[key])
            p = float(A[key]) / float(C[key])
            f = p * r * 2 / (p + r)
            print("%s:\t p:%f\t r:%f\t f:%f" % (key, p, r, f))
        except:
            print("error:", key, "right:", A.get(key, 0), "real:", B.get(key, 0), "predict:", C.get(key, 0))
