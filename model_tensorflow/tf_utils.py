#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 19-5-31 下午3:34
@File    : tf_utils.py
@Desc    : tensorflow 工具
"""

import os
import json
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import _Linear
from tensorflow.python.ops.rnn_cell_impl import RNNCell



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


def load(file_name):
    """
    从文件加载数据
    :param file_name:
    :return:
    """
    with open(file_name) as json_file:
        data = json.load(json_file)
        return data

def load_json_from_file(file_name):
    """
    从文件加载数据
    :param file_name:
    :return:
    """
    with open(file_name) as json_file:
        data = json.load(json_file)
        return data


def get_input_schema_spec(input_schema):
    """
    获取输入信息
    :param input_schema:
    :return:
    """
    feature_map = {}
    for input_tensor,param in input_schema.items():
        if param["feature_type"] == "fixed_len":
            type = tf.string if param['value_type'] == 'string' else tf.int64 if param['value_type'] == 'int' else tf.float32 if param['value_type'] == 'double' else None
            shape = param["value_shape"] if param.has_key("value_shape") else None
            default_value = param["default_value"] if param.has_key("default_value") else None
            if type is None:
                print("no value_type")
            elif shape is not None:
                feature_map[input_tensor] = tf.FixedLenFeature(shape=[int(shape)], dtype=type, default_value=default_value)
            else:
                feature_map[input_tensor] = tf.FixedLenFeature(shape=[], dtype=type, default_value=default_value)
    return feature_map


def PReLU(_x, name=None):
    if name is None:
        name = "alpha"
    _alpha = tf.get_variable(name=name,
                             shape=_x.get_shape()[-1],
                             initializer=tf.constant_initializer(0.25),
                             dtype=_x.dtype)

    return tf.nn.leaky_relu(_x, alpha=_alpha, name=None)


def PReLU2(_x, name=None):
    if name is None:
        name = "alpha"
    _alpha = tf.get_variable(name,
                                shape=_x.get_shape()[-1],
                                initializer=tf.constant_initializer(0.25),
                                dtype=_x.dtype)

    return tf.maximum(_alpha * _x, _x)


def dice(_x, axis=-1, epsilon=0.000000001, name='dice'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        alphas = tf.get_variable('alpha'+name, _x.get_shape()[-1],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        input_shape = list(_x.get_shape())

        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[axis] = input_shape[axis]
    mean = tf.reduce_mean(_x, axis=reduction_axes)
    broadcast_mean = tf.reshape(mean, broadcast_shape)
    std = tf.reduce_mean(tf.square(_x - broadcast_mean) + epsilon, axis=reduction_axes)
    std = tf.sqrt(std)
    broadcast_std = tf.reshape(std, broadcast_shape)
    x_normed = (_x - broadcast_mean) / (broadcast_std + epsilon)
    x_p = tf.sigmoid(x_normed)
    return alphas * (1.0 - x_p) * _x + x_p * _x



class VecAttGRUCell(RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).
    Args:
      num_units: int, The number of units in the GRU cell.
      activation: Nonlinearity to use.  Default: `tanh`.
      reuse: (optional) Python boolean describing whether to reuse variables
       in an existing scope.  If not `True`, and the existing scope already has
       the given variables, an error is raised.
      kernel_initializer: (optional) The initializer to use for the weight and
      projection matrices.
      bias_initializer: (optional) The initializer to use for the bias.
    """

    def __init__(self,
                 num_units,
                 activation=None,
                 reuse=None,
                 kernel_initializer=None,
                 bias_initializer=None):
        super(VecAttGRUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation or math_ops.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._gate_linear = None
        self._candidate_linear = None

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, att_score):
        return self.call(inputs, state, att_score)

    def call(self, inputs, state, att_score=None):
        """Gated recurrent unit (GRU) with nunits cells."""
        if self._gate_linear is None:
            bias_ones = self._bias_initializer
            if self._bias_initializer is None:
                bias_ones = init_ops.constant_initializer(
                    1.0, dtype=inputs.dtype)
            with vs.variable_scope("gates"):  # Reset gate and update gate.
                self._gate_linear = _Linear(
                    [inputs, state],
                    2 * self._num_units,
                    True,
                    bias_initializer=bias_ones,
                    kernel_initializer=self._kernel_initializer)

        value = math_ops.sigmoid(self._gate_linear([inputs, state]))
        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state
        if self._candidate_linear is None:
            with vs.variable_scope("candidate"):
                self._candidate_linear = _Linear(
                    [inputs, r_state],
                    self._num_units,
                    True,
                    bias_initializer=self._bias_initializer,
                    kernel_initializer=self._kernel_initializer)
        c = self._activation(self._candidate_linear([inputs, r_state]))
        u = (1.0 - att_score) * u
        new_h = u * state + (1 - u) * c
        return new_h, new_h





def GET_COLUMNS(input_data=None):


    WIDE_CATE_COLS = []
    DEEP_EMBEDDING_COLS = []
    CONTINUOUS_COLS = []
    DEEP_SHARED_EMBEDDING_COLS = []

    ORIGIN_DEEP_SHARED_EMBEDDING_COLS = []
    for fea in input_data:
        if 'col_type' in fea:
            if isinstance(fea['col_type'], list):
                for col in fea['col_type']:
                    if col == 'WIDE_CATE_COLS':
                        WIDE_CATE_COLS.append((fea['name'], fea['bucket_size']))
                    if col == 'DEEP_EMBEDDING_COLS':
                        DEEP_EMBEDDING_COLS.append(
                            (fea['name'], fea['bucket_size'], fea['embedding_size'], fea['type']))
                    if col == 'CONTINUOUS_COLS':
                        CONTINUOUS_COLS.append(fea['name'])
                    if fea['col_type'] == 'DEEP_SHARED_EMBEDDING_COLS':
                        ORIGIN_DEEP_SHARED_EMBEDDING_COLS.append(
                            (fea['name'], fea['bucket_size'], fea['embedding_size'], fea['type'], fea['shared_flag']))
            else:
                if fea['col_type'] == 'WIDE_CATE_COLS':
                    WIDE_CATE_COLS.append((fea['name'], fea['bucket_size']))
                if fea['col_type'] == 'DEEP_EMBEDDING_COLS':
                    DEEP_EMBEDDING_COLS.append((fea['name'], fea['bucket_size'], fea['embedding_size'], fea['type']))
                if fea['col_type'] == 'CONTINUOUS_COLS':
                    CONTINUOUS_COLS.append(fea['name'])
                if fea['col_type'] == 'DEEP_SHARED_EMBEDDING_COLS':
                    ORIGIN_DEEP_SHARED_EMBEDDING_COLS.append(
                        (fea['name'], fea['bucket_size'], fea['embedding_size'], fea['type'], fea['shared_flag']))

    print("ORIGIN_DEEP_SHARED_EMBEDDING_COLS:", ORIGIN_DEEP_SHARED_EMBEDDING_COLS)
    shared_flags = set()
    for _, _, _, _, flag in ORIGIN_DEEP_SHARED_EMBEDDING_COLS:
        shared_flags.add(flag)

    for c_flag in shared_flags:
        names = []
        bucket_sizes = []
        embedding_sizes = []
        types = []
        for name, bucket_size, embedding_size, type, flag in ORIGIN_DEEP_SHARED_EMBEDDING_COLS:
            if c_flag == flag:
                names.append(name)
                bucket_sizes.append(bucket_size)
                embedding_sizes.append(embedding_size)
                types.append(type)
        DEEP_SHARED_EMBEDDING_COLS.append((names, bucket_sizes[0], embedding_sizes[0], types[0], c_flag))

    print("DEEP_SHARED_EMBEDDING_COLS:", DEEP_SHARED_EMBEDDING_COLS)
    print("WIDE_CATE_COLS:", WIDE_CATE_COLS)
    print('CONTINUOUS_COLS:', CONTINUOUS_COLS)
    print('DEEP_EMBEDDING_COLS:', DEEP_EMBEDDING_COLS)

    WIDE_CROSS_COLS = (('pv', 'created_time', 140),
                       ('pv', 'g_created_time', 140),
                       ('class2', 'class2', 250))
    return WIDE_CATE_COLS,DEEP_EMBEDDING_COLS,CONTINUOUS_COLS,DEEP_SHARED_EMBEDDING_COLS,WIDE_CROSS_COLS