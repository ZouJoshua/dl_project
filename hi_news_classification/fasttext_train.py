#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 19-3-14 下午11:27
@File    : fasttext_train.py
@Desc    : 

"""

import tensorflow as tf
import numpy as np
import os
import sys

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
training_data_file = os.path.join(data_dir, "top_category_corpus")
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
tf.flags.DEFINE_float("learning_rate", 0.05, "learning rate")
tf.flags.DEFINE_integer("batch_size", 256, "batch size for training/evaluating")  # 批处理的大小 32-->128
tf.flags.DEFINE_integer("decay_steps", 100, "how many steps before decay learning rate")
tf.flags.DEFINE_float("decay_rate", 0.96, "Rate of decay for learning rate")  # 一次衰减多少
tf.flags.DEFINE_integer("num_sampled", 100, "number of noise sampling")
tf.flags.DEFINE_float("dropout_keep_prob", 0.8, "to control the activation level of neurons")
tf.flags.DEFINE_string("ckpt_dir", model_checkpoint, "checkpoint location for the model")
tf.flags.DEFINE_integer("sentence_len", 200, "max sentence length")
tf.flags.DEFINE_integer("embed_size", 300, "embedding size")
tf.flags.DEFINE_boolean("is_training", True, "true:training, false:testing/inference")
tf.flags.DEFINE_integer("num_epochs", 15, "epoch times")
tf.flags.DEFINE_integer("validate_every", 1, "validate every validate_every epochs")  # 每1轮做一次验证
tf.flags.DEFINE_boolean("use_embedding", True, "whether to use embedding or not")


def next_batch(x, y, batch_size):
    """
    生成batch数据集(随机打散)，用生成器的方式输出
    """

    perm = np.arange(len(x))
    np.random.shuffle(perm)
    x = [x[i] for i in perm]
    y = [y[i] for i in perm]

    num_batches = len(x) // batch_size

    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        batch_x = x[start: end]
        batch_y = y[start: end]

        yield batch_x, batch_y


def batch_train(sess, fast_text, batch_x, batch_y, summary_op, train_summary_writer):
    """
    训练函数
    """
    curr_loss, curr_acc, step, summary, train_op = sess.run(
        [fast_text.loss_val, fast_text.accuracy, fast_text.global_step, summary_op, fast_text.train_op],
        feed_dict={fast_text.sentence: batch_x, fast_text.label: batch_y})
    train_summary_writer.add_summary(summary, step)
    return curr_loss, curr_acc



# 定义性能指标函数


def gen_metrics(y_true, y_pred, logits):
    """
    生成acc和auc值
    """
    auc = roc_auc_score(y_true, logits)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    return round(accuracy, 4), round(auc, 4), round(precision, 4), round(recall, 4), round(f1, 4)


# 在验证集上做验证，计算损失、精确度
def do_eval(sess, fast_text, eval_x, eval_y, summary_op, eval_summary_writer, index2label):
    number_examples = len(eval_x)
    print("number_examples for validation:", number_examples)
    eval_loss, eval_acc, eval_counter = 0.0, 0.0, 0
    batch_size = 4096

    for batch_eval in next_batch(eval_x, eval_y, batch_size):
        curr_eval_loss, logits, step, real_labels, pred_labels, summary = sess.run(
            [fast_text.loss_val, fast_text.logits, fast_text.epoch_increment, fast_text.y_true, fast_text.y_pred, summary_op],
            feed_dict={fast_text.sentence: batch_eval[0], fast_text.label: batch_eval[1]})
        eval_summary_writer.add_summary(summary, step)

        labels_one_hot = tf.one_hot(real_labels, 9).eval()
        pred_labels_one_hot = tf.one_hot(pred_labels, 9).eval()
        acc, auc, prec, recall, f1 = gen_metrics(real_labels, pred_labels, logits)

        eval_loss, eval_counter, eval_acc = eval_loss + curr_eval_loss, eval_counter + 1, eval_acc + acc

        print("validation acc:{}, prec:{}, f1:{}, recall:{}, auc:{}".format(acc, prec, f1, recall, auc))


    return eval_loss/float(eval_counter), eval_acc/float(eval_counter)




def tf_confusion_metrics(predict, real, session):
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



#保存为pb模型
def export_model(session, fast_text, export_path):

   #只需要修改这一段，定义输入输出，其他保持默认即可
    prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs={"input": tf.saved_model.utils.build_tensor_info(fast_text.sentence)},
        outputs={"output": tf.saved_model.utils.build_tensor_info(fast_text.label)},
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




def main(_):
    """step1 -> 导入数据(从h5py缓存文件导入或重新处理数据导入)
        step2 -> 创建session
        step3 -> 喂数据
        step4 -> 训练
        step5 -> (验证)
        step6 ->（预测）"""
    # step1 -> load data
    ds = DataSet(data_dir, word2vec_file, training_data_file, embedding_dims=FLAGS.embed_size)
    train, test, _ = ds.load_data(use_embedding=True, valid_portion=0.2)
    index2label = ds.index2label
    vocab_embedding = ds.embedding
    vocab_size = len(ds.word2index)
    print("fasttext_model.vocab_size:", vocab_size)
    # num_classes = len(ds.label2index)
    # print("num_classes:", num_classes)
    print("num_classes:", FLAGS.label_size)
    trainX, trainY = train
    testX, testY = test

    print("testX.shape:", np.array(testX).shape)
    print("testY.shape:", np.array(testY).shape)
    print("testX[0]:", testX[0])  # [17, 25, 10, 406, 26, 14, 56, 61, 62, 323, 4]
    print("testX[1]:", testX[1])
    print("testY[0]:", testY[0])  # 0
    print("testY[1]:", testY[1])   # 0

    # Sequence padding
    print("start padding ...")
    trainX = pad_sequences(trainX, maxlen=FLAGS.sentence_len, value=0.)  # padding to max length
    testX = pad_sequences(testX, maxlen=FLAGS.sentence_len, value=0.)    # padding to max length
    ###############################################################################################
    # with open(FLAGS.cache_path, 'w') as data_f: #save data to cache file, so we can use it next time quickly.
    #    pickle.dump((trainX,trainY,testX,testY,vocabulary_index2word),data_f)
    ###############################################################################################
    print("testX[0]:", testX[0])
    print("testX[1]:", testX[1])  # [17, 25, 10, 406, 26, 14, 56, 61, 62, 323, 4]
    print("testY[0]:", testY[0])  # 0
    print("testY[1]:", testY[1])  # 0
    print("end padding ...")
    num_examples, FLAGS.sentence_len = trainX.shape
    print("num_examples of training:", num_examples, "\nsentence_len:", FLAGS.sentence_len)

    # word2index, label2index, trainX, trainY, vaildX, vaildY, testX, testY = ds.load_data_from_h5py(FLAGS.cache_file_h5py, FLAGS.cache_file_pickle)

    # step2 -> create session
    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # run_metadata = tf.RunMetadata

        # Instantiate Model
        fast_text = FastText(FLAGS.label_size, FLAGS.learning_rate, FLAGS.decay_rate, FLAGS.decay_steps,
                             FLAGS.batch_size, FLAGS.num_sampled, FLAGS.dropout_keep_prob,
                             FLAGS.sentence_len, vocab_size, FLAGS.embed_size, FLAGS.is_training)

        print("writing to {}\n".format(output_dir))

        # 用summary绘制tensorBoard
        tf.summary.scalar("loss", fast_text.loss_val)
        tf.summary.scalar('acc', fast_text.accuracy)
        summary_op = tf.summary.merge_all()

        train_summary_dir = os.path.join(output_dir, "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
        eval_summary_dir = os.path.join(output_dir, "eval")
        eval_summary_writer = tf.summary.FileWriter(eval_summary_dir, sess.graph)

        # Initialize Save
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)


        if os.path.exists(FLAGS.ckpt_dir):
            print("restoring variables from checkpoint")
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            os.makedirs(model_checkpoint)
            print("created model checkpoint dir: {}".format(FLAGS.ckpt_dir))
            print('initializing variables...')
            sess.run(tf.global_variables_initializer())
            if FLAGS.use_embedding:  # load pre-trained word vectors
                word_embedding = tf.constant(vocab_embedding, dtype=tf.float32)  # convert to tensor
                t_assign_embedding = tf.assign(fast_text.embedding,
                                               word_embedding)
                sess.run(t_assign_embedding)

        curr_epoch = sess.run(fast_text.epoch_step)

        # step3 -> feed data and train the model
        number_of_training_data = len(trainX)
        batch_size = FLAGS.batch_size

        # 训练模型
        print("start training model...")
        for epoch in range(curr_epoch, FLAGS.num_epochs):
            loss, acc, counter = 0.0, 0.0, 0
            # print(trainY[:5])
            # print(trainX[:1])
            for batch in next_batch(trainX, trainY, batch_size):
                # if epoch == 0 and counter == 0:
                #     print("train_x[start:end]:", batch[0])
                #     print("train_y[start:end]:", batch[1])
                batch_loss, batch_acc = batch_train(sess, fast_text, batch[0], batch[1], summary_op, train_summary_writer)
                loss, acc, counter = loss + batch_loss, acc + batch_acc, counter + 1

                if counter % 20 == 0:
                    print("epoch %d\tbatch %d\ttrain loss:%.3f\ttrain accuracy:%.3f" % (epoch, counter, loss / float(counter),
                                                                                        acc / float(counter)))
                # if counter % 122 == 0:
                #     eval_loss, eval_accuracy = do_eval(sess, fast_text, testX, testY, index2label)
                #     print(eval_loss, eval_accuracy)
            # 验证模型

            # epoch increment
            print("going to increment epoch counter....")
            sess.run(fast_text.epoch_increment)

            # step4 -> validation
            print("validation epoch:{} validate_every:{}".format(epoch, FLAGS.validate_every))
            if epoch % FLAGS.validate_every == 0:
                eval_loss, eval_acc = do_eval(sess, fast_text, testX, testY, summary_op, eval_summary_writer, index2label)
                print("epoch %d validation loss:%.3f \t validation accuracy: %.3f" % (epoch, eval_loss, eval_acc))
                # save model to checkpoint
                save_path = os.path.join(FLAGS.ckpt_dir, "model.ckpt")
                path = saver.save(sess, save_path, global_step=epoch)
                print("saved model checkpoint to {}\n".format(path))

        # step5 -> 最后在测试集上测试
        test_loss, test_acc = do_eval(sess, fast_text, testX, testY, summary_op, eval_summary_writer, index2label)
        print("test loss: %2.4f, test accruacy: %2.4f" % (test_loss, test_acc))

        # 保存模型的一种方式，保存为pb文件
        export_model(sess, fast_text, model_saved)




if __name__ == "__main__":
    tf.app.run()
    # print(current_work_dir)
    # print(root_dir)