#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 19-7-17 上午11:43
@File    : test_log.py
@Desc    : 测试日志功能
"""

import os
from unittest import main, TestCase
from utils.logger import Logger
from setting import LOG_PATH

log_file1 = os.path.join(LOG_PATH, "test1")
log_file2 = os.path.join(LOG_PATH, "test2")

class TestLog(TestCase):

    @classmethod
    def setUpClass(cls):
        print(">>>>>>>>>>测试环境已准备好！")
        print(">>>>>>>>>>即将测试 Case ...")

    @classmethod
    def tearDownClass(cls):
        print(">>>>>>>>>>Case 用例已测试完成 ...")
        print(">>>>>>>>>>测试环境已清理完成！")

    def test_file_logger(self):
        flogger = Logger('flogger', log2console=False, log2file=True, logfile=log_file1).get_logger()
        flogger.debug('debug')
        flogger.info('info')
        flogger.warning('warn')
        flogger.error("error")

    def test_console_logger(self):
        clogger = Logger('clogger', log2console=True, log2file=False).get_logger()
        clogger.debug('debug')
        clogger.info('info')
        clogger.warning('warn')
        clogger.error("error")


    def test_file_console_logger(self):
        fclogger = Logger('fclogger', log2console=True, log2file=True, logfile=log_file2).get_logger()
        fclogger.debug('debug')
        fclogger.info('info')
        fclogger.warning('warn')
        fclogger.error('error')


if __name__ == '__main__':
    main()

