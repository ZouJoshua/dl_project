#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 19-5-31 下午3:34
@File    : tf_utils.py
@Desc    : tensorflow 工具
"""

import os
import tensorflow as tf




def export_model(session, fast_text, export_path):
    """
    使用SavedModel格式保存pb模型
    :param session: 会话
    :param fast_text: 模型
    :param export_path: 保存路径
    :return:
    """
   #只需要修改这一段，定义输入输出，其他保持默认即可
    prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs={"input": tf.saved_model.utils.build_tensor_info(fast_text.sentence)},
        outputs={"output": tf.saved_model.utils.build_tensor_info(fast_text.logits)},
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

    if os.path.exists(export_path):
        os.system("rm -rf " + export_path)
    print("Export the model to {}".format(export_path))

    try:
        legacy_init_op = tf.group(
            tf.tables_initializer(), name='legacy_init_op')
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)
        builder.add_meta_graph_and_variables(
            session, [tf.saved_model.tag_constants.SERVING],
            clear_devices=True,
            signature_def_map={
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    prediction_signature},
            legacy_init_op=legacy_init_op)

        builder.save()
    except Exception as e:
        print("Fail to export saved model, exception: {}".format(e))



def get_variables(ckpt_path):
    """
    查看所有tensorflow模型变量方法1
    :param ckpt_path: tensorflow ckpt 模型
    :return:
    """
    from tensorflow.python import pywrap_tensorflow as pt
    reader = pt.NewCheckpointReader(ckpt_path)
    # 获取 变量名: 形状
    vars = reader.get_variable_to_shape_map()
    for k in sorted(vars):
        print(k, vars[k])

    # 获取 变量名: 类型
    vars = reader.get_variable_to_dtype_map()
    for k in sorted(vars):
        print(k, vars[k])

    # 获取张量的值
    value = reader.get_tensor("tensor_name")


def get_variables_v2(ckpt_path):
    """
    查看所有tensorflow模型变量方法2
    :param ckpt_path: tensorflow ckpt 模型
    :return:
    """
    from tensorflow.python.tools import inspect_checkpoint as chkp

    # 打印检查点所有的变量
    chkp.print_tensors_in_checkpoint_file(ckpt_path, tensor_name='', all_tensors=True)
    # 仅打印检查点中的 v1 变量
    chkp.print_tensors_in_checkpoint_file(ckpt_path, tensor_name='v1', all_tensors=False)



def tf_confusion_metrics(predict, real, session):
    """
    计算混淆矩阵准确率、召回率、精确率等
    :param predict:
    :param real:
    :param session:
    :return:
    """
    predictions = tf.argmax(predict, 1)
    actuals = tf.argmax(real, 1)

    ones_like_actuals = tf.ones_like(actuals)
    zeros_like_actuals = tf.zeros_like(actuals)
    ones_like_predictions = tf.ones_like(predictions)
    zeros_like_predictions = tf.zeros_like(predictions)

    tp_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, ones_like_actuals),
                tf.equal(predictions, ones_like_predictions)
            ),
            "float"
        )
    )

    tn_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, zeros_like_actuals),
                tf.equal(predictions, zeros_like_predictions)
            ),
            "float"
        )
    )

    fp_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, zeros_like_actuals),
                tf.equal(predictions, ones_like_predictions)
            ),
            "float"
        )
    )

    fn_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, ones_like_actuals),
                tf.equal(predictions, zeros_like_predictions)
            ),
            "float"
        )
    )
    tp, tn, fp, fn = session.run([tp_op, tn_op, fp_op, fn_op])

    tpr = float(tp) / (float(tp) + float(fn))
    fpr = float(fp) / (float(fp) + float(tn))
    fnr = float(fn) / (float(tp) + float(fn))

    accuracy = (float(tp) + float(tn)) / (float(tp) + float(fp) + float(fn) + float(tn))

    recall = tpr
    precision = float(tp) / (float(tp) + float(fp))

    f1_score = (2 * (precision * recall)) / (precision + recall)

    return accuracy, recall, precision, f1_score