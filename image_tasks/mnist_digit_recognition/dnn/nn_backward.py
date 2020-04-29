#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 12/2/19 8:21 PM
@File    : nn_backward.py
@Desc    : 单层反向传播

"""


"""
网络结构：
输入层 -> 输出层
Layer1:
input[None,784] -> softmax[784,10] -> out[None, 10]
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from image_tasks.mnist_digit_recognition.dnn import nn_forward
import os


BATCH_SIZE = 200
LEARNING_RATE_BASE = 0.01
REGULARIZER = 0.0001
STEPS = 50000
LOG_PATH = "/data/work/dl_project/logs/mnist"
MODEL_SAVE_PATH = "./model_nn/"
MODEL_NAME = "mnist_model"
train_num_examples = 60000  # mnist.train.num_examples


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def backward(mnist):
    x = tf.placeholder(tf.float32, [None, nn_forward.INPUT_NODE])
    y_ = tf.placeholder(tf.float32, [None, nn_forward.OUTPUT_NODE])
    y = nn_forward.forward(x, None)
    global_step = tf.Variable(0, trainable=False)


    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    loss = tf.reduce_mean(ce)
    # loss = cem + tf.add_n(tf.get_collection("losses"))

    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE_BASE).minimize(loss, global_step=global_step)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver(max_to_keep=3)
    # img_batch, label_batch = mnist_generateds.get_tfrecord(BATCH_SIZE, isTrain=True)

    gpu_config = tf.ConfigProto()
    gpu_config.allow_soft_placement = True
    gpu_config.gpu_options.allow_growth = True

    tf.summary.scalar("loss", loss)
    tf.summary.scalar('accuracy', accuracy)
    merged = tf.summary.merge_all()


    with tf.Session(config=gpu_config) as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # 断点续训
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        # 创建writer
        train_writer = tf.summary.FileWriter(LOG_PATH, sess.graph)
        # coord = tf.train.Coordinator()
        # threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            # xs, ys = sess.run([img_batch, label_batch])
            # _, loss_value, step = sess.run([train_step, loss, global_step], feed_dict={x:xs, y_:ys})
            _, loss_value, step, summary = sess.run([train_step, loss, global_step, merged], feed_dict={x:xs, y_:ys})
            train_writer.add_summary(summary, i)
            train_writer.flush()
            if i % 1000 == 0:
                accuracy_score = sess.run(accuracy, feed_dict={x: mnist.validation.images, y_: mnist.validation.labels})
                print("After %d training step(s), loss on training batch is %g, acc on validation is %g" % (step, loss_value, accuracy_score))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
        train_writer.close()
        # coord.request_stop()
        # coord.join(threads)


def main():
    mnist = input_data.read_data_sets("/data/work/dl_project/data/mnist", one_hot=True)
    backward(mnist)


if __name__ == "__main__":
    main()


