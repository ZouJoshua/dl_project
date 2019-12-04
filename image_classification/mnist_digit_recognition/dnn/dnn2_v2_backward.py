#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 11/18/19 11:13 PM
@File    : dnn2_v2_backward.py
@Desc    : 双隐藏层反向传播过程

"""


"""
网络结构：
输入层 -> 隐藏层1（600神经元） -> 隐藏层2(480神经元) -> 输出层
Input:
input[None,784]
Layer1:
relu[784, 600] -> dropout[784, 600]
Layer2:
relu[600,480] -> dropout[600,480] 
Output:
softmax[480,10] -> out[None, 10]

"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from image_classification.mnist_digit_recognition.dnn import dnn2_v2_forward
import os


BATCH_SIZE = 200
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 0.0001
STEPS = 50000
MOVING_AVERAGE_DECAY = 0.99
KEEP_PROB = 0.8
LOG_PATH = "/data/work/dl_project/logs/mnist"
MODEL_SAVE_PATH = "./model_dnn/"
MODEL_NAME = "mnist_model"
train_num_examples = 60000  # mnist.train.num_examples


os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def backward(mnist):
    x = tf.placeholder(tf.float32, [None, dnn2_v2_forward.INPUT_NODE])
    y_ = tf.placeholder(tf.float32, [None, dnn2_v2_forward.OUTPUT_NODE])
    y = dnn2_v2_forward.forward(x, REGULARIZER, KEEP_PROB)
    global_step = tf.Variable(0, trainable=False)

    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection("losses"))

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True
    )

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())

    # 在训练神经网络模型时，每过一遍数据既需要通过反向传播来更新神经网络中的参数，又要更新每一个参数的轮滑平均值。
    # 为了一次完成多个操作，Tensorflow 提供了 tf.control_dependencies 和 tf.group 两种机制。下面两行程序和
    # tf.group(train_step, variables_averages_op) 是等价的
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name="train")

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
            _, loss_value, step, summary = sess.run([train_op, loss, global_step, merged], feed_dict={x:xs, y_:ys})
            train_writer.add_summary(summary, i)
            train_writer.flush()
            if i % 1000 == 0:
                # accuracy_score = sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels})
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