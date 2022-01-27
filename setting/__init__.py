#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 19-7-17 上午11:19
@File    : __init__.py
@Desc    : 
"""

import sys
import os
from os.path import dirname
try:
    import configparser
except:
    from six.moves import configparser


#####
"""
current file and current path setting
"""
curr_file = os.path.realpath(__file__)
curr_path = dirname(curr_file)

#####
"""
project root path setting
"""
PROJECT_ROOT_PATH = dirname(curr_path)
sys.path.append(PROJECT_ROOT_PATH)

#####
"""
config path setting
"""
CONFIG_PATH = os.path.join(PROJECT_ROOT_PATH, 'config')
DEFAULT_CONFIG_FILE = os.path.join(CONFIG_PATH, 'default.conf')

#####
"""
read config file
"""
config = configparser.ConfigParser()
config.read(DEFAULT_CONFIG_FILE, encoding='utf-8')

#####
"""
data path setting
"""
DATA_PATH_NAME = config.get('DEFAULT.data', 'DEFAULT_DATA_PATH')
DATA_PATH = os.path.join(PROJECT_ROOT_PATH, DATA_PATH_NAME)

#####
"""
log path setting
"""
LOG_PATH_NAME = config.get('DEFAULT.log', 'DEFAULT_lOGGING_PATH')
LOG_PATH = os.path.join(PROJECT_ROOT_PATH, LOG_PATH_NAME)
if not os.path.exists(LOG_PATH):
    os.mkdir(LOG_PATH)
DEFAULT_LOGGING_LEVEL = config.getint('DEFAULT.log', 'DEFAULT_LOGGING_LEVEL')
DEFAULT_LOGGING_FILE_NAME = config.get('DEFAULT.log', 'DEFAULT_LOGGING_FILE_NAME')
DEFAULT_LOGGING_FILE = os.path.join(LOG_PATH, DEFAULT_LOGGING_FILE_NAME)

