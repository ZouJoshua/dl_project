#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 19-9-5 上午11:34
@File    : models.py
@Desc    : Contains the base class for models.
"""


class BaseModel(object):
  """Inherit from this class when implementing new models."""

  def create_model(self, unused_model_input, **unused_params):
    raise NotImplementedError()