#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 12/5/19 1:57 PM
@File    : test_language_detection.py
@Desc    : 

"""

from collections import defaultdict
from feature_extractor.multilanguage_keywords_extractor.model.language_detection.ld_polyglot import \
    LanguageDetectionPolyglot

languageDetection = LanguageDetectionPolyglot()
num = defaultdict(lambda: 0)
for line in open('test.txt', mode='r', encoding='utf-8'):
    language = languageDetection.detect(line.split(',')[0][1:])
    num[str(language)] += 1
    if language.code not in ['en', 'ko']:
        num['-KO-EN'] += 1
    num['all'] += 1
for k, v in sorted(num.items(), key=lambda d: d[1]):
    print(k, v)