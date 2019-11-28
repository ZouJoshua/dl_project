#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 11/19/19 10:45 PM
@File    : lenet5_backward.py
@Desc    : lenet5 反向传播

"""

import tensorflow as tf
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data
from image_classification.mnist_digit_recognition import lenet5_forward

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.005
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 0.0001
STEPS = 50000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "mnist_model"
train_num_examples = 60000  # mnist.train.num_examples

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def backward(mnist):
    x = tf.placeholder(tf.float32, shape=[
        BATCH_SIZE,
        lenet5_forward.IMAGE_SIZE,
        lenet5_forward.IMAGE_SIZE,
        lenet5_forward.NUM_CHANNELS
    ])

    y_ = tf.placeholder(tf.float32, shape=[None, lenet5_forward.OUTPUT_NODE])
    y = lenet5_forward.forward(x, True, REGULARIZER)
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

    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name="train")

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver(max_to_keep=3)

    gpu_config = tf.ConfigProto()
    gpu_config.allow_soft_placement = True
    gpu_config.gpu_options.allow_growth = True

    with tf.Session(config=gpu_config) as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        for i in range(STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs, (
                BATCH_SIZE,
                lenet5_forward.IMAGE_SIZE,
                lenet5_forward.IMAGE_SIZE,
                lenet5_forward.NUM_CHANNELS
            ))
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_:ys})
            if i % 100 == 0:
                print("After %d training step(s), loss on train batch is %g" % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

def main():
    mnist_data_dir = "/data/work/dl_project/data/mnist"
    mnist = input_data.read_data_sets(mnist_data_dir, one_hot=True)
    backward(mnist)


if __name__ == "__main__":
    main()