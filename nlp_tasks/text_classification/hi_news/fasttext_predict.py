#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 19-3-14 下午11:28
@File    : fasttext_predict.py
@Desc    : process--->
            1.load data(X:list of lint,y:int).
            2.create session. 3.feed data. 4.predict
"""



import tensorflow as tf
import numpy as np
import os
import sys
import codecs

current_work_dir = os.path.realpath(__file__)
root_dir = os.path.dirname(os.path.dirname(current_work_dir))
sys.path.append(root_dir)


import warnings
warnings.filterwarnings('ignore')

from tf_model.fasttext_model import FastText
from tflearn.data_utils import pad_sequences
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from gensim.models import KeyedVectors
from preprocess.preprocess_data_hi import DataSet


data_dir = os.path.join(root_dir, "data", "hi_news")
predict_data_file = os.path.join(data_dir, "corpus_predict")
predict_result_file = os.path.join(data_dir, "corpus_predict_result")
word2vec_file = os.path.join(data_dir, "word2vec.bin")
model_checkpoint = os.path.join(data_dir, "fasttext_checkpoint")
model_saved = os.path.join(data_dir, "pb_model")
cache_file_h5py = os.path.join(data_dir, "data.h5")
cache_file_pickle = os.path.join(data_dir, "vocab_label.pik")
output_dir = os.path.join(data_dir, "summarys")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("cache_file_h5py", cache_file_h5py, "path of training/validation/test data.")
tf.flags.DEFINE_string("cache_file_pickle", cache_file_pickle, "path of vocabulary and label files")

tf.flags.DEFINE_integer("label_size", 9, "number of label")
tf.flags.DEFINE_float("learning_rate", 0.01, "learning rate")
tf.flags.DEFINE_integer("batch_size", 256, "batch size for training/evaluating")  # 批处理的大小 32-->128
tf.flags.DEFINE_integer("decay_steps", 10000, "how many steps before decay learning rate")
tf.flags.DEFINE_float("decay_rate", 0.96, "Rate of decay for learning rate")  # 一次衰减多少
tf.flags.DEFINE_integer("num_sampled", 100, "number of noise sampling")
tf.flags.DEFINE_float("dropout_keep_prob", 0.8, "to control the activation level of neurons")
tf.flags.DEFINE_string("ckpt_dir", model_checkpoint, "checkpoint location for the model")
tf.flags.DEFINE_integer("sentence_len", 200, "max sentence length")
tf.flags.DEFINE_integer("embed_size", 300, "embedding size")
tf.flags.DEFINE_boolean("is_training", False, "true:training, false:testing/inference")
tf.flags.DEFINE_integer("num_epochs", 20, "epoch times")
tf.flags.DEFINE_integer("validate_every", 1, "validate every validate_every epochs")  # 每1轮做一次验证
tf.flags.DEFINE_boolean("use_embedding", True, "whether to use embedding or not")
tf.flags.DEFINE_string("predict_source_file", predict_data_file, "target file path for final prediction")
tf.flags.DEFINE_string("predict_target_file", predict_result_file, "target file path for final prediction")
tf.flags.DEFINE_boolean("load_pb_model_flag", True, "if load pb model or load ckpt model")


def predict_from_pb_model(testX2, id_list, idx2label, session, model_path, savedmodel=False):
    if savedmodel:
        output_graph_def = tf.GraphDef()
        with open(model_path + "/frozen_model.pb", "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tensors = tf.import_graph_def(output_graph_def, name="")
        init = tf.global_variables_initializer()
        session.run(init)
        print("load model finish!")
        input = session.graph.get_tensor_by_name("sentence:0")
        output = session.graph.get_tensor_by_name("accuracy/predictions:0")
        # 测试pb模型
        predict_target_file_f = codecs.open(FLAGS.predict_target_file, 'a', 'utf8')
        for i, test_x in enumerate(testX2):
            feed_dict_map = {input: [test_x]}
            predict = session.run(output, feed_dict=feed_dict_map)
            _id = id_list[i]
            # write_labels_file(_id, predict_y, predict_target_file_f)
            label = idx2label.get(predict[0], None)
            write_labels_file(_id, label, predict_target_file_f)
            # print("{},{}".format(_id, label))
        predict_target_file_f.close()
    else:
        meta_graph = tf.saved_model.loader.load(session, [tf.saved_model.tag_constants.SERVING], model_path)
        model_graph_signature = list(meta_graph.signature_def.items())[0][1]
        output_tensor_names = []
        output_op_names = []
        for output_item in model_graph_signature.outputs.items():
            output_op_name = output_item[0]
            print(output_op_name)
            output_op_names.append(output_op_name)
            output_tensor_name = output_item[1].name
            output_tensor_names.append(output_tensor_name)
        print("load model finish!")
        sentences = {}
        # 测试pb模型
        predict_target_file_f = codecs.open(FLAGS.predict_target_file, 'a', 'utf8')
        for i, test_x in enumerate(testX2):
            sentences["input"] = [test_x]
            feed_dict_map = {}
            for input_item in model_graph_signature.inputs.items():
                input_op_name = input_item[0]
                input_tensor_name = input_item[1].name
                feed_dict_map[input_tensor_name] = sentences[input_op_name]
            logits = session.run(output_tensor_names, feed_dict=feed_dict_map)
            label = get_label_from_logits(logits[0], idx2label)
            _id = id_list[i]
            write_labels_file(_id, label, predict_target_file_f)
            # print("predict y:", label)
        predict_target_file_f.close()

def predict_from_ckpt_model(testX2, id_list, idx2label, session, vocab_size):
    # Instantiate Model
    fast_text = FastText(FLAGS.label_size, FLAGS.learning_rate, FLAGS.decay_rate, FLAGS.decay_steps,
                         FLAGS.batch_size, FLAGS.num_sampled, FLAGS.dropout_keep_prob,
                         FLAGS.sentence_len, vocab_size, FLAGS.embed_size, FLAGS.is_training)

    saver = tf.train.Saver()
    if os.path.exists(FLAGS.ckpt_dir):
        print("Restoring Variables from Checkpoint")
        saver.restore(session, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
    else:
        print("Can't find the checkpoint.going to stop")
        return
    # feed data, to get logits
    number_of_training_data = len(testX2)
    print("number_of_predict_data:", number_of_training_data)
    batch_size = 1
    predict_target_file_f = codecs.open(FLAGS.predict_target_file, 'a', 'utf8')
    for start, end in zip(range(0, number_of_training_data, batch_size),
                          range(batch_size, number_of_training_data + 1, batch_size)):
        logits = session.run(fast_text.logits,
                          feed_dict={fast_text.sentence: testX2[start:end]})  # 'shape of logits:', ( 1, 1999)
        label = get_label_from_logits(logits[0], idx2label)
        _id = id_list[start]
        write_labels_file(_id, label, predict_target_file_f)
    predict_target_file_f.close()


# get label using logits
def get_label_from_logits(logits, index2label):
    y_pred = np.argmax(logits)
    label = index2label.get(y_pred, None)
    return label


# write question id and labels to file system.
def write_labels_file(_id, label, f):
    f.write(_id + "," + label + "\n")
    sys.stdout.flush()
    f.flush()



def main(_):
    """导入数据 -> 创建session -> 喂数据 -> 训练 -> (验证) ->（预测）"""

    # step1 -> load data
    ds = DataSet(data_dir, word2vec_file, embedding_dims=FLAGS.embed_size)
    word2idx = ds.word2index
    idx2label = ds.index2label
    data = ds.load_data_predict(FLAGS.predict_source_file, word2idx)
    predict_data = list()
    id_list = list()
    vocab_size = len(word2idx)
    print("fasttext_model.vocab_size:", vocab_size)
    for doc in data:
        doc_id, pred_data = doc
        id_list.append(doc_id)
        predict_data.append(pred_data)

    # 2.Data preprocessing: Sequence padding
    print("start padding....")
    testX2 = pad_sequences(predict_data, maxlen=FLAGS.sentence_len, value=0.)  # padding to max length
    print("end padding...")

    # 3.create session.
    with tf.Graph().as_default():
        config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # 定义会话
        sess = tf.Session(config=config)
        with sess.as_default():
            if FLAGS.load_pb_model_flag:
                model_path = "/home/zoushuai/algoproject/tf_project/data/hi_news/frozen_model"
                predict_from_pb_model(testX2, id_list, idx2label, sess, model_path, savedmodel=True)
            else:
                predict_from_ckpt_model(testX2, id_list, idx2label, sess, vocab_size)





if __name__ == "__main__":
    tf.app.run()