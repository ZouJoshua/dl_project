#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 19-9-10 下午2:41
@File    : extract_feature_main.py
@Desc    : 
"""

import os
import sys
import logging


# curr_path = os.path.dirname(os.path.realpath(__file__))
# vc_path = os.path.dirname(curr_path)
# root_path = os.path.dirname(os.path.dirname(vc_path))
# sys.path.append(vc_path)
# sys.path.append(root_path)
# print(sys.path)

import cv2
from video_tasks.server.feature_extractor import YouTube8MFeatureExtractor
# from predict.predict_main import load_flags_config
import numpy
import tensorflow as tf
from tensorflow import app
from tensorflow import flags

# In OpenCV3.X, this is available as cv2.CAP_PROP_POS_MSEC
# In OpenCV2.X, this is available as cv2.cv.CV_CAP_PROP_POS_MSEC
CAP_PROP_POS_MSEC = 0

class ExtractFeature(object):

    def __init__(self, extract_flags, logger=None):
        self.flags = extract_flags
        if logger:
            self.log = logger
        else:
            self.log = logging.getLogger("yt8m_video_classification")
            self.log.setLevel(logging.INFO)
        self.log.info("Loading youtube8m feature extractor...")
        self.extractor = YouTube8MFeatureExtractor(extract_flags.extractor_model_dir)
        self.log.info("Successful load extractor")


    def frame_iterator(self, filename, every_ms=1000, max_num_frames=300):
      """Uses OpenCV to iterate over all frames of filename at a given frequency.

      Args:
        filename: Path to video file (e.g. mp4)
        every_ms: The duration (in milliseconds) to skip between frames.
        max_num_frames: Maximum number of frames to process, taken from the
          beginning of the video.

      Yields:
        RGB frame with shape (image height, image width, channels)
      """
      video_capture = cv2.VideoCapture()
      if not video_capture.open(filename):
        # print >> sys.stderr, 'Error: Cannot open video file ' + filename
        self.log.error("Error: Cannot open video file {}".format(filename))
        return
      last_ts = -99999  # The timestamp of last retrieved frame.
      num_retrieved = 0

      while num_retrieved < max_num_frames:
        # Skip frames
        while video_capture.get(CAP_PROP_POS_MSEC) < every_ms + last_ts:
          if not video_capture.read()[0]:
            return

        last_ts = video_capture.get(CAP_PROP_POS_MSEC)
        has_frames, frame = video_capture.read()
        if not has_frames:
          break
        yield frame
        num_retrieved += 1


    def _int64_list_feature(self, int64_list):
      return tf.train.Feature(int64_list=tf.train.Int64List(value=int64_list))


    def _bytes_feature(self, value):
      return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


    def _make_bytes(self, int_array):
      if bytes == str:  # Python2
        return ''.join(map(chr, int_array))
      else:
        return bytes(int_array)


    def quantize(self, features, min_quantized_value=-2.0, max_quantized_value=2.0):
      """Quantizes float32 `features` into string."""
      assert features.dtype == 'float32'
      assert len(features.shape) == 1  # 1-D array
      features = numpy.clip(features, min_quantized_value, max_quantized_value)
      quantize_range = max_quantized_value - min_quantized_value
      features = (features - min_quantized_value) * (255.0 / quantize_range)
      features = [int(round(f)) for f in features]

      return self._make_bytes(features)


    def extract(self, video_file):

        rgb_features = []
        for rgb in self.frame_iterator(video_file, every_ms=1000.0/self.flags.frames_per_second):
            features = self.extractor.extract_rgb_frame_features(rgb[:, :, ::-1])
            rgb_features.append(self._bytes_feature(self.quantize(features)))

        if not rgb_features:
            # print >> sys.stderr, 'Could not get features for ' + video_file
            self.log.error("Could not get features for {}".format(video_file))

        # Create SequenceExample proto and write to output.
        feature_list = {
            self.flags.image_feature_key: tf.train.FeatureList(feature=rgb_features),
        }
        if self.flags.insert_zero_audio_features:
            feature_list['audio'] = tf.train.FeatureList(
              feature=[self._bytes_feature(self._make_bytes([0] * 128))] * len(rgb_features))

        example = tf.train.SequenceExample(
            context=tf.train.Features(feature={
                self.flags.labels_feature_key:
                    self._int64_list_feature([0]),
                self.flags.video_file_key_feature_key:
                    self._bytes_feature(self._make_bytes(map(ord, video_file))),
                # "predictions": tf.train.Feature(float_list=tf.train.FloatList(value=video_prediction))
            }),
            feature_lists=tf.train.FeatureLists(feature_list=feature_list))
        # print('Successfully encoded %i out of %i videos' % (
        #   total_written, total_written + total_error))
        return example.SerializeToString()



# import time
# flags = load_flags_config()
# ef = ExtractFeature(flags)
# video_file = "/home/zoushuai/Downloads/videoplayback.mp4"
# s = time.time()
# feature = ef.extract(video_file)
# print(feature)
# e = time.time()
#
# print(">> 抽取视频特征耗时{}s".format(e -s))