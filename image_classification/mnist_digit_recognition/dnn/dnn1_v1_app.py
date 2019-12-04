#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 11/19/19 12:24 AM
@File    : dnn1_v1_app.py
@Desc    : 预测应用

"""


import tensorflow as tf
import numpy as np
from PIL import Image
from image_classification.mnist_digit_recognition.dnn import dnn1_v1_forward, dnn1_v1_backward


def restore_model(testPicArr):
    with tf.Graph().as_default() as tg:
        x = tf.placeholder(tf.float32, [None, dnn1_v1_forward.INPUT_NODE])
        y = dnn1_v1_forward.forward(x, None)

        preValue = tf.argmax(y, 1)
        variable_average = tf.train.ExponentialMovingAverage(dnn1_v1_backward.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_average.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(dnn1_v1_backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                preValue = sess.run(preValue, feed_dict={x: testPicArr})
                return preValue
            else:
                print("No checkpoint file found")
                return -1

def pre_pic(picName):
    img = Image.open(picName)
    reIm = img.resize((28, 28), Image.ANTIALIAS)
    im_arr = np.array(reIm.convert("L"))
    threshold = 50
    for i in range(28):
        for j in range(28):
            im_arr[i][j] = 255 - im_arr[i][j]
            if (im_arr[i][j] < threshold):
                im_arr[i][j] = 0
            else:
                im_arr[i][j] = 255
    nm_arr = im_arr.reshape([1, 784])
    nm_arr = nm_arr.astype(np.float32)
    im_ready = np.multiply(nm_arr, 1.0/255.0)

    return im_ready



def application():
    testNum = int(input("input the number of pictures:"))
    for i in range(testNum):
        testPic = input("the path of test picture:")
        testPicArr = pre_pic(testPic)
        preValue = restore_model(testPicArr)
        print("The prediction number is:", preValue)

def main():
    application()