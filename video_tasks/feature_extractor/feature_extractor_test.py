#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 19-9-5 下午6:18
@File    : feature_extractor_test.py
@Desc    : Tests for feature_extractor
"""


import json
import os
from video_tasks.feature_extractor import feature_extractor
import numpy
from PIL import Image
from six.moves import cPickle
from tensorflow.python.platform import googletest


def _FilePath(filename):
  return os.path.join('testdata', filename)


def _MeanElementWiseDifference(a, b):
  """Calculates element-wise percent difference between two numpy matrices."""
  difference = numpy.abs(a - b)
  denominator = numpy.maximum(numpy.abs(a), numpy.abs(b))

  # We dont care if one is 0 and another is 0.01
  return (difference / (0.01 + denominator)).mean()


class FeatureExtractorTest(googletest.TestCase):

  def setUp(self):
    self._extractor = feature_extractor.YouTube8MFeatureExtractor()

  def testPCAOnFeatureVector(self):
    sports_1m_test_data = cPickle.load(open(_FilePath('sports1m_frame.pkl')))
    actual_pca = self._extractor.apply_pca(sports_1m_test_data['original'])
    expected_pca = sports_1m_test_data['pca']
    self.assertLess(_MeanElementWiseDifference(actual_pca, expected_pca), 1e-5)


if __name__ == '__main__':
  googletest.main()
