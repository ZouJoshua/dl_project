#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 12/5/19 1:40 PM
@File    : stem_nltk.py
@Desc    : 

"""

from nltk.stem import SnowballStemmer

from feature_extractor.multilanguage_keywords_extractor.core import base
from feature_extractor.multilanguage_keywords_extractor.model.stem.stem_base import Stem
from feature_extractor.multilanguage_keywords_extractor.util import file_utils

class StemNLTK(Stem):
    def __init__(self):
        self.__support_languages = set(file_utils.readLanguages(base.stemNLTKSupportLanguagePath))

    def stem(self, words, language):
        assert language in self.__support_languages, 'StemPolyglot do not support {} language.'.format(language)
        stemmer = SnowballStemmer(language.name)
        return [stemmer.stem(word) if type(word) == str else (stemmer.stem(word[0]), word[1]) for word in words]

    def supportLanguages(self):
        return self.__support_languages
