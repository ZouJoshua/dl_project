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
from tf_model.fasttext_model import fastText
from tflearn.data_utils import pad_sequences
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from gensim.models import KeyedVectors
from preprocess.preprocess_data_hi import DataSet

current_work_dir = os.path.realpath(__file__)
root_dir = os.path.dirname(os.path.dirname(current_work_dir))
data_dir = os.path.join(root_dir, "data", "hi_news")
training_data_file = os.path.join(data_dir, "top_category_corpus")
word2vec_file = os.path.join(data_dir, "word2vec.bin")
model_checkpoint = os.path.join(data_dir, "fasttext_checkpoint")
model_saved = os.path.join(data_dir, "fasttext_model_saved")
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
tf.flags.DEFINE_integer("batch_size", 128, "batch size for training/evaluating")  # 批处理的大小 32-->128
tf.flags.DEFINE_integer("decay_steps", 20000, "how many steps before decay learning rate")
tf.flags.DEFINE_float("decay_rate", 0.8, "Rate of decay for learning rate")  # 一次衰减多少
tf.flags.DEFINE_integer("num_sampled", 100, "number of noise sampling")
tf.flags.DEFINE_float("dropout_keep_prob", 0.8, "to control the activation level of neurons")
tf.flags.DEFINE_string("ckpt_dir", model_checkpoint, "checkpoint location for the model")
tf.flags.DEFINE_integer("sentence_len", 200, "max sentence length")
tf.flags.DEFINE_integer("embed_size", 300, "embedding size")
tf.flags.DEFINE_boolean("is_training", True, "true:training, false:testing/inference")
tf.flags.DEFINE_integer("num_epochs", 15, "epoch times")
tf.flags.DEFINE_integer("validate_every", 1, "validate every validate_every epochs")  # 每10轮做一次验证
tf.flags.DEFINE_boolean("use_embedding", True, "whether to use embedding or not")


def next_batch(x, y, batch_size):
    """
    生成batch数据集(随机打散)，用生成器的方式输出
    """

    perm = np.arange(len(x))
    np.random.shuffle(perm)
    x = x[perm]
    y = y[perm]

    num_batches = len(x) // batch_size

    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        batchX = x[start: end]
        batchY = y[start: end]

        yield batchX, batchY


def batch_train(sess, fast_text, batchX, batchY, summary_op, train_summary_writer):
    """
    训练函数
    """
    curr_loss, curr_acc, step, summary, train_op = sess.run(
        [fast_text.loss_val, fast_text.accuracy, fast_text.global_step, summary_op, fast_text.train_op],
        feed_dict={fast_text.sentence: batchX, fast_text.label: batchY})
    train_summary_writer.add_summary(summary, step)

    loss, acc, counter = loss + curr_loss, acc + curr_acc, counter + 1
    if counter % 100 == 0:
        print("epoch %d\tbatch %d\ttrain loss:%.3f\ttrain accuracy:%.3f" % (
        epoch, counter, loss / float(counter), acc / float(counter)))


# 定义性能指标函数

def mean(item):
    return sum(item) / len(item)


def gen_metrics(trueY, predY, binaryPredY):
    """
    生成acc和auc值
    """
    auc = roc_auc_score(trueY, predY)
    accuracy = accuracy_score(trueY, binaryPredY)
    precision = precision_score(trueY, binaryPredY)
    recall = recall_score(trueY, binaryPredY)

    return round(accuracy, 4), round(auc, 4), round(precision, 4), round(recall, 4)


# 在验证集上做验证，报告损失、精确度
def do_eval(sess, fast_text, evalX, evalY, summary_op, eval_summary_writer):
    number_examples = len(evalX)
    eval_loss, eval_acc, eval_counter = 0.0, 0.0, 0
    batch_size = 1
    for start, end in zip(range(0, number_examples, batch_size), range(batch_size, number_examples, batch_size)):
        curr_eval_loss, curr_eval_acc, step, summary = sess.run(
            [fast_text.loss_val, fast_text.accuracy, fast_text.global_step, summary_op],
            feed_dict={fast_text.sentence: evalX[start:end], fast_text.label: evalY[start: end]})
        eval_loss, eval_acc, eval_counter = eval_loss+curr_eval_loss, eval_acc+curr_eval_acc, eval_counter+1

        eval_summary_writer.add_summary(summary, step)
    return eval_loss/float(eval_counter), eval_acc/float(eval_counter)




# 预测的准确率
def calculate_accuracy(labels_predicted, labels,eval_counter):
    if eval_counter < 10:
        print("labels_predicted:", labels_predicted, " ;labels:", labels)
    count = 0
    label_dict = {x: x for x in labels}
    for label_predict in labels_predicted:
        flag = label_dict.get(label_predict, None)
    if flag is not None:
        count = count + 1
    return count / len(labels)



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
        fast_text = fastText(FLAGS.label_size, FLAGS.learning_rate, FLAGS.decay_rate, FLAGS.decay_steps,
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
        # 保存模型的一种方式，保存为pb文件
        builder = tf.saved_model.builder.SavedModelBuilder(model_saved)

        if os.path.exists(FLAGS.ckpt_dir):
            print("Restoring variables from checkpoint")
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            os.makedirs(model_checkpoint)
            print("created model checkpoint dir: {}".format(FLAGS.ckpt_dir))
            print('Initializing Variables')
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
        for epoch in range(curr_epoch, FLAGS.num_epochs):  # range(start,stop,step_size)
            loss, acc, counter = 0.0, 0.0, 0
            for start, end in zip(range(0, number_of_training_data, batch_size), range(batch_size, number_of_training_data, batch_size)):
                if epoch == 0 and counter == 0:
                    print("trainX[start:end]:", trainX[start:end])
                    print("trainY[start:end]:", trainY[start:end])
                curr_loss, curr_acc, step, summary, train_op = sess.run([fast_text.loss_val, fast_text.accuracy, fast_text.global_step, summary_op, fast_text.train_op],
                                                  feed_dict={fast_text.sentence: trainX[start:end], fast_text.label: trainY[start:end]})
                train_summary_writer.add_summary(summary, step)

                loss, acc, counter = loss+curr_loss, acc+curr_acc, counter+1
                if counter % 100 == 0:
                    print("epoch %d\tbatch %d\ttrain loss:%.3f\ttrain accuracy:%.3f" % (epoch, counter, loss/float(counter), acc/float(counter)))

                # if fast_text.global_step % 100 == 0:
                #     eval_loss, eval_accuracy = do_eval(sess, fast_text, testX, testY, batch_size, summary_op, eval_summary_writer)
                #     print("epoch %d validation loss:%.3f \t validation accuracy: %.3f" % (
                #     epoch, eval_loss, eval_accuracy))  # ,\tValidation Accuracy: %.3f--->eval_acc
                #     # save model to checkpoint
                #     if start % (450 * FLAGS.batch_size) == 0:
                #         print("going to save checkpoint.")
                #         save_path = os.path.join(FLAGS.ckpt_dir, "model.ckpt")
                #         path = saver.save(sess, save_path, global_step=epoch)  # fast_text.epoch_step
                #         print("saved model checkpoint to {}\n".format(path))

            # epoch increment
            print("going to increment epoch counter....")
            sess.run(fast_text.epoch_increment)

            # 4.validation
            print(epoch, FLAGS.validate_every, (epoch % FLAGS.validate_every == 0))
            if epoch % FLAGS.validate_every == 0:
                eval_loss, eval_acc = do_eval(sess, fast_text, testX, testY, summary_op, eval_summary_writer)
                print("epoch %d validation loss:%.3f \t validation accuracy: %.3f" % (epoch, eval_loss, eval_acc))
                # save model to checkpoint
                save_path = os.path.join(FLAGS.ckpt_dir, "model.ckpt")
                path = saver.save(sess, save_path, global_step=epoch)
                print("saved model checkpoint to {}\n".format(path))

        # 5.最后在测试集上做测试，并报告测试准确率 Test
        test_loss, test_acc = do_eval(sess, fast_text, testX, testY, summary_op, eval_summary_writer)
        print("test loss: %2.4f, test accruacy: %2.4f" % (test_loss, test_acc))
    pass



if __name__ == "__main__":
    tf.app.run()
    # print(current_work_dir)
    # print(root_dir)