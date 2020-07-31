#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 12/5/19 1:10 PM
@File    : base.py
@Desc    : 

"""

import os


pwd_dir = os.path.split(os.path.realpath(__file__))[0]
# print(pwd_dir)
data_dir = pwd_dir[0:pwd_dir.rfind('core')]
# print(data_dir)

data_path = data_dir + 'data/'
# print(data_path)
language_path = data_path + 'support_language/language'
# with open(language_path, "w", encoding="utf-8") as f:
#     f.write("zh\t中文\n")


tokenizationPosPolyglotSupportLanguagePath = data_path + 'support_language/tokenization_pos_polyglot'

stemNLTKSupportLanguagePath = data_path + 'support_language/stem_nltk'

punctuationStopwordPath = data_path + 'stopword/punctuation'

englishStopwordPath = data_path + 'stopword/EnglishStoplist.txt'

modelPath = data_path + 'model/'